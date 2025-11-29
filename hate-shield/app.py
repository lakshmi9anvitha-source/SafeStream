# app.py
import os
import re
import json
import time
import threading
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import emoji
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pytchat

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, FileResponse

# =========================
# CONFIG
# =========================
YOUTUBE_URL = "https://www.youtube.com/live/b_o38Nn2kps?si=a-778BiADjWgZ1ei"
MODEL_PATH = "./model"
BADWORDS_DATASET = "hate.csv"   # optional; used for masking only
STATE_PATH = "hs_state.json"
UNCERT_PATH = "uncertain_queue.jsonl"

THRESHOLDS = {
    "toxic": 0.36,
    "severe_toxic": 0.40,
    "obscene": 0.33,
    "threat": 0.33,
    "insult": 0.40,
    "identity_hate": 0.12
}
UNCERT_LO, UNCERT_HI = 0.35, 0.65
SAFE_USERS_LIMIT = 3

BUILTIN_BAD = ["bitch", "stupid", "idiot", "hell", "damn", "asshole", "shit", "fuck", "faggot", "bastard"]

# block when warnings > BLOCK_AFTER
BLOCK_AFTER = 3

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# UTILITIES
# =========================
def extract_video_id(url: str):
    if not url:
        return None
    m = re.search(r"(?:v=|\/live\/|youtu\.be\/|\/watch\?v=|\/)([0-9A-Za-z_-]{11})", url)
    return m.group(1) if m else None

# emoji list for lightweight signal
HATE_EMOJIS = [
    "🤬","😡","😠","😤","😾","👿","💢","😈","👺","👹",
    "🤯","😖","😣","😫","😩","🥵","🥶","🙄","😏","😒"
]
def contains_hate_emoji(text: str):
    return [e for e in HATE_EMOJIS if e in (text or "")]

# normalization helpers
LEET_MAP = {'0':'o','1':'i','3':'e','4':'a','5':'s','7':'t','@':'a','$':'s','!':'i','|':'i'}
def replace_leet(s: str) -> str:
    return ''.join(LEET_MAP.get(ch, ch) for ch in s)

def collapse_repeated_letters(s: str) -> str:
    return re.sub(r'([a-z])\1{2,}', r'\1\1', s, flags=re.IGNORECASE)

def deobfuscate_spaces(s: str) -> str:
    # if token is like "f u c k" or "f u c k i n g", remove spaces inside continuous letter sequences
    # but preserve legitimate spaces between words
    return re.sub(r'\b(?:[A-Za-z]\s+){1,}[A-Za-z]\b', lambda m: re.sub(r'\s+','',m.group(0)), s)

def normalize_text_for_model(s: str) -> str:
    if not s:
        return ""
    t = s.lower()
    t = emoji.demojize(t)
    t = re.sub(r'http\S+','', t)
    t = re.sub(r'@\w+','', t)
    t = deobfuscate_spaces(t)
    t = replace_leet(t)
    t = collapse_repeated_letters(t)
    t = re.sub(r'[^a-z0-9\s:]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# =========================
# BAD WORDS loader (masking)
# =========================
def load_bad_words(path: str):
    words = []
    if not path or not os.path.exists(path):
        return list(dict.fromkeys([w.lower() for w in BUILTIN_BAD]))
    try:
        import pandas as pd
        df = pd.read_csv(path, dtype=str, encoding='utf-8', keep_default_na=False)
        if df.shape[1] == 1:
            series = df.iloc[:,0].astype(str)
        else:
            candidates = [c for c in df.columns if str(c).lower() in ("text","word","comment","badword","phrase")]
            series = df[candidates[0]].astype(str) if candidates else df.astype(str).apply(lambda r: " ".join(r.values), axis=1)
        for v in series.tolist():
            parts = [p.strip() for p in re.split(r'[,\n;|/]+', str(v)) if p.strip()]
            words.extend(parts)
    except Exception:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                for line in fh:
                    parts = [p.strip() for p in re.split(r'[,\n;|/]+', line) if p.strip()]
                    words.extend(parts)
        except Exception:
            words = []
    norm = []
    for w in words:
        w2 = normalize_text_for_model(str(w))
        if w2 and w2 not in ("word","badword"):
            norm.append(w2)
    for b in BUILTIN_BAD:
        if b not in norm:
            norm.append(b)
    return list(dict.fromkeys(norm))

BAD_WORDS = load_bad_words(BADWORDS_DATASET)
print(f"Loaded {len(BAD_WORDS)} bad words (masking list).")

# =========================
# Highlight / masking improvement:
# - match tokens in the original text,
# - when replacing for display we remove internal spaces so "f u c k" becomes "fuck" when revealed,
# - mask placeholder shown by default; reveal shows deobfuscated token (spaces removed)
# =========================
def highlight_hate(text: str, bad_words_normalized: List[str], hate_emojis: List[str]):
    """
    Return a string where offensive tokens are wrapped in a custom marker using **...** (same as previous),
    but replace the revealed token with a deobfuscated (spaces removed) version.
    e.g. original: "f u c k you" -> highlighted: "**fuck** you"  (stored in returned string)
    The UI will mask '**fuck**' by default and reveal the deobfuscated token.
    """
    if not text:
        return ""
    out = text
    # mark emojis first
    for e in (hate_emojis or []):
        out = out.replace(e, f"**{e}**")

    # iterate tokens in the original text, try to match normalized forms against bad_words list
    tokens = re.findall(r"\S+", text)
    for tok in tokens:
        normalized_tok = normalize_text_for_model(tok)
        for bad in bad_words_normalized:
            if not bad:
                continue
            if bad in normalized_tok or normalized_tok in bad:
                # create a deobfuscated display token (remove internal spaces)
                display_tok = re.sub(r'\s+', '', tok)
                # keep original case/spaces removed for display; wrap with ** for frontend parsing
                try:
                    # replace the exact token occurrence with deobfuscated bolded token
                    out = re.sub(re.escape(tok), f"**{display_tok}**", out, count=1)
                except re.error:
                    out = out.replace(tok, f"**{display_tok}**", 1)
                break
    return out

# =========================
# Load model/tokenizer
# =========================
LABEL_NAMES = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
tokenizer = None
model = None
MODEL_READY = False
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device).eval()
    MODEL_READY = True
    print("Loaded model from", MODEL_PATH)
except Exception as e:
    print("Failed to load local model:", e)
    try:
        tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
        model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
        model.to(device).eval()
        MODEL_READY = True
        print("Loaded fallback unitary/toxic-bert")
    except Exception as e2:
        print("Failed fallback:", e2)
        MODEL_READY = False

# =========================
# classify_texts
# =========================
def classify_texts(texts: List[str]):
    if not MODEL_READY:
        return [{"probs": {n:0.0 for n in LABEL_NAMES}, "flagged": False, "reasons": []} for _ in texts]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits
        probs_batch = torch.sigmoid(logits).cpu().tolist()
    results = []
    for text, probs in zip(texts, probs_batch):
        prob_map = {n: float(p) for n, p in zip(LABEL_NAMES, probs)}
        reasons = []
        flagged = False
        for label in ("toxic","obscene","insult"):
            if prob_map.get(label, 0.0) >= THRESHOLDS.get(label, 0.5):
                reasons.append(label)
                flagged = True
        for label,p in prob_map.items():
            if UNCERT_LO < p < UNCERT_HI:
                try:
                    with open(UNCERT_PATH, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"text": text, "label": label, "prob": p, "ts": time.time()}) + "\n")
                except Exception:
                    pass
                break
        results.append({"probs": prob_map, "flagged": bool(flagged), "reasons": reasons})
    return results

def predict_batch(texts: List[str]):
    res = classify_texts(texts)
    out = []
    for r in res:
        out.append((1 if r["flagged"] else 0, r["probs"].get("toxic", 0.0)))
    return out

# =========================
# State & persistence
# =========================
CHAT = None
CHAT_CREATED = False
EVENT_LOOP = None
STOP_EVENT = threading.Event()
monitor_thread = None
subscribers = set()

user_warnings = {}
blocked_users = set()
safe_users = set()
STATE_LOCK = threading.Lock()

def load_state():
    global user_warnings, blocked_users, safe_users
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as fh:
                s = json.load(fh)
                user_warnings = s.get("user_warnings", {})
                blocked_users = set(s.get("blocked_users", []))
                safe_users = set(s.get("safe_users", []))
                print("Loaded state:", STATE_PATH)
        except Exception as e:
            print("Failed to load state:", e)

def save_state():
    with STATE_LOCK:
        try:
            with open(STATE_PATH, "w", encoding="utf-8") as fh:
                json.dump({"user_warnings": user_warnings, "blocked_users": list(blocked_users), "safe_users": list(safe_users)}, fh, indent=2)
        except Exception as e:
            print("Failed to save state:", e)

load_state()

# =========================
# pytchat helpers
# =========================
CHAT_LOCK = threading.Lock()
def create_chat(video_id: str):
    global CHAT, CHAT_CREATED
    try:
        CHAT = pytchat.create(video_id=video_id)
        CHAT_CREATED = True
        print(f"CHAT created for video_id={video_id}")
    except Exception as e:
        CHAT = None
        CHAT_CREATED = False
        print("CHAT creation failed:", e)

def terminate_chat():
    global CHAT, CHAT_CREATED
    try:
        if CHAT:
            try:
                CHAT.terminate()
            except Exception:
                pass
            CHAT = None
        CHAT_CREATED = False
    except Exception as e:
        print("Error terminating chat:", e)

# =========================
# Monitor
# =========================
def monitor_chat():
    global CHAT
    if CHAT is None:
        print("CHAT not created. Exiting thread.")
        return

    print("Monitor thread running...")
    while CHAT.is_alive() and not STOP_EVENT.is_set():
        try:
            for item in CHAT.get().sync_items():
                if STOP_EVENT.is_set():
                    break

                raw_author = getattr(item.author, "name", "unknown") or "unknown"
                author = raw_author.strip()
                text = getattr(item, "message", "") or ""

                # skip blocked users immediately
                if author in blocked_users:
                    continue

                # preserve raw text for infractions; normalized version for model
                normalized = normalize_text_for_model(text)
                hate_emoji_list = contains_hate_emoji(text)

                ml_res = classify_texts([normalized])[0]
                label = 1 if ml_res.get("flagged") else 0
                ml_reasons = ml_res.get("reasons", [])

                # safe users exempt
                if author in safe_users:
                    label = 0
                    ml_reasons = []

                # build reason list for UI
                reason_list = [r.replace('_',' ').capitalize() for r in ml_reasons]
                if hate_emoji_list:
                    reason_list.append("Hate Emoji")

                # increment warnings if any model reason or hate emoji (unless safe user)
                should_increment = False
                if ml_reasons:
                    should_increment = True
                if hate_emoji_list:
                    should_increment = True
                if author in safe_users:
                    should_increment = False

                if should_increment:
                    prev = user_warnings.get(author, 0)
                    user_warnings[author] = prev + 1
                    save_state()
                    print(f"[warnings] {author} -> {user_warnings[author]} (prev {prev})")
                    # block if warnings > BLOCK_AFTER
                    if user_warnings.get(author, 0) > BLOCK_AFTER:
                        blocked_users.add(author)
                        user_warnings.pop(author, None)
                        save_state()
                        print(f"[blocked] {author} (warnings exceeded {BLOCK_AFTER})")
                        try:
                            broadcast_event({"type": "blocked_update", "blocked_users": list(blocked_users)})
                        except Exception:
                            pass

                # highlighted text — use deobfuscated display tokens (so revealed text is compact)
                highlighted = ""
                if ml_reasons or hate_emoji_list:
                    highlighted = highlight_hate(text, BAD_WORDS, hate_emoji_list)

                prob_toxic = ml_res.get("probs", {}).get("toxic", 0.0)

                payload = {
                    "author": author,
                    "text": text,
                    "highlighted_text": highlighted,
                    "pred_label": label,
                    "pred_prob": round(float(prob_toxic), 3),
                    "hate_emojis": ",".join(hate_emoji_list),
                    "warnings_for_user": user_warnings.get(author, 0),
                    "blocked_users": list(blocked_users),
                    "safe_users": list(safe_users),
                    "reason_list": reason_list,
                    "fetched_at": datetime.now(timezone.utc).isoformat()
                }

                if EVENT_LOOP:
                    EVENT_LOOP.call_soon_threadsafe(broadcast_event, payload)

        except Exception as e:
            print("Chat error:", e)
            import time as _t
            _t.sleep(0.5)

    print("Monitor thread stopped.")

# broadcast SSE
def broadcast_event(data):
    msg = json.dumps(data)
    for q in list(subscribers):
        try:
            q.put_nowait(msg)
        except Exception:
            pass

# =========================
# FastAPI app & endpoints (unchanged)
# =========================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    path = Path("static/frontend.html")
    if not path.exists():
        path = Path("static/index.html")
    return path.read_text(encoding="utf-8") if path.exists() else "frontend missing!"

@app.get("/status")
def status():
    return JSONResponse({
        "model_ready": bool(MODEL_READY),
        "bad_words_count": len(BAD_WORDS),
        "video_id": extract_video_id(YOUTUBE_URL) if YOUTUBE_URL else None,
        "chat_created": bool(CHAT_CREATED),
        "blocked_users_count": len(blocked_users),
        "warnings_total": sum(user_warnings.values()),
        "safe_users": list(safe_users),
        "safe_users_limit": SAFE_USERS_LIMIT
    })

@app.post("/set_video")
async def set_video(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    url = data.get("url") or data.get("video") or data.get("video_url")
    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url'")
    vid = extract_video_id(url)
    if not vid:
        raise HTTPException(status_code=400, detail="Invalid youtube url")
    with CHAT_LOCK:
        try:
            stop_monitor_thread(timeout=2.0)
        except:
            pass
        terminate_chat()
        global CHAT, CHAT_CREATED, YOUTUBE_URL
        YOUTUBE_URL = url
        create_chat(vid)
        start_monitor_thread()
    return JSONResponse({"ok": True, "message": f"Switched to {vid}", "video_id": vid})

@app.get("/stream")
async def stream(request: Request):
    q = asyncio.Queue()
    subscribers.add(q)
    async def event_gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                msg = await q.get()
                yield f"data: {msg}\n\n"
        finally:
            subscribers.discard(q)
    return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.post("/block")
async def block_user(request: Request):
    try:
        data = await request.json()
        user = data.get("user")
        if not user:
            raise HTTPException(status_code=400, detail="Missing 'user'")
        blocked_users.add(user.strip())
        user_warnings.pop(user.strip(), None)
        save_state()
        broadcast_event({"type": "blocked_update", "blocked_users": list(blocked_users)})
        return JSONResponse({"ok": True, "blocked_users": list(blocked_users)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid request")

@app.post("/unblock")
async def unblock(request: Request):
    try:
        data = await request.json()
        user = data.get("user")
        if not user:
            raise HTTPException(status_code=400, detail="Missing 'user'")
        blocked_users.discard(user.strip())
        user_warnings.pop(user.strip(), None)
        save_state()
        broadcast_event({"type": "blocked_update", "blocked_users": list(blocked_users)})
        return JSONResponse({"ok": True, "blocked_users": list(blocked_users)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid request")

@app.get("/blocked")
def get_blocked():
    return JSONResponse({"blocked_users": list(blocked_users)})

@app.get("/safe")
def get_safe():
    return JSONResponse({"safe_users": list(safe_users)})

@app.post("/safe_add")
async def safe_add(request: Request):
    try:
        data = await request.json()
        user = data.get("user")
        if not user:
            raise HTTPException(status_code=400, detail="Missing 'user'")
        if len(safe_users) >= SAFE_USERS_LIMIT:
            return JSONResponse({"ok": False, "error": "safe list full", "safe_users": list(safe_users)})
        safe_users.add(user.strip())
        save_state()
        broadcast_event({"type": "safe_update", "safe_users": list(safe_users)})
        return JSONResponse({"ok": True, "safe_users": list(safe_users)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid request")

@app.post("/safe_remove")
async def safe_remove(request: Request):
    try:
        data = await request.json()
        user = data.get("user")
        if not user:
            raise HTTPException(status_code=400, detail="Missing 'user'")
        safe_users.discard(user.strip())
        save_state()
        broadcast_event({"type": "safe_update", "safe_users": list(safe_users)})
        return JSONResponse({"ok": True, "safe_users": list(safe_users)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid request")

@app.post("/fp_report")
async def fp_report(request: Request):
    try:
        j = await request.json()
        with open("fp_reports.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"text": j.get("text"), "author": j.get("author"), "ts": time.time()}) + "\n")
        return JSONResponse({"ok": True})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid request")

@app.get("/uncertain_export")
def uncertain_export():
    if not os.path.exists(UNCERT_PATH):
        return JSONResponse({"ok": True, "count": 0})
    out_csv = "uncertain_export.csv"
    try:
        import csv
        with open(UNCERT_PATH, "r", encoding="utf-8") as fh, open(out_csv, "w", newline='', encoding="utf-8") as fo:
            w = csv.writer(fo)
            w.writerow(["text","label","prob","ts"])
            for line in fh:
                try:
                    o = json.loads(line)
                    w.writerow([o.get("text",""), o.get("label",""), o.get("prob",""), o.get("ts","")])
                except:
                    pass
        return FileResponse(out_csv, media_type="text/csv", filename=out_csv)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})

# monitor thread helpers
def start_monitor_thread():
    global monitor_thread, STOP_EVENT
    STOP_EVENT.clear()
    monitor_thread = threading.Thread(target=monitor_chat, daemon=True)
    monitor_thread.start()
    print("Monitor thread started.")

def stop_monitor_thread(timeout=3.0):
    global monitor_thread, STOP_EVENT
    STOP_EVENT.set()
    if monitor_thread and monitor_thread.is_alive():
        monitor_thread.join(timeout=timeout)
    STOP_EVENT.clear()

# startup/shutdown
@app.on_event("startup")
async def startup():
    global CHAT, CHAT_CREATED, EVENT_LOOP, monitor_thread
    EVENT_LOOP = asyncio.get_running_loop()
    print("Starting up HateShield server...")
    with CHAT_LOCK:
        try:
            vid = extract_video_id(YOUTUBE_URL) if YOUTUBE_URL else None
            if vid:
                create_chat(vid)
        except Exception as e:
            print("CHAT creation failed at startup:", e)
    STOP_EVENT.clear()
    monitor_thread = threading.Thread(target=monitor_chat, daemon=True)
    monitor_thread.start()
    print("Monitor thread started (startup).")

@app.on_event("shutdown")
async def shutdown():
    print("Shutting down...")
    STOP_EVENT.set()
    if monitor_thread and monitor_thread.is_alive():
        monitor_thread.join(timeout=5)
    try:
        terminate_chat()
    except:
        pass
    save_state()
    print("Shutdown complete.")
