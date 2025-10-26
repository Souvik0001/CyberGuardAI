import os
import io
import csv
import json
import time
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import pytesseract

# ML / training-related imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib


# ─────────────────────────────────────────
# CONFIG / PATHS
# ─────────────────────────────────────────

DATA_DIR = "data"
MODELS_DIR = "models"

HARASSMENT_MODEL_PATH = os.path.join(MODELS_DIR, "harassment_model.pkl")
HARASSMENT_VECTORIZER_PATH = os.path.join(
    MODELS_DIR, "harassment_vectorizer.pkl"
)
TAMPER_MODEL_PATH = os.path.join(MODELS_DIR, "tamper_model.pkl")

HARASSMENT_DATA_PATH = os.path.join(DATA_DIR, "harassment_samples.jsonl")
TAMPER_DATA_PATH = os.path.join(DATA_DIR, "tamper_samples.jsonl")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Tesseract path setup (Windows default; harmless on Linux/Mac if missing)
CUSTOM_TESS = os.getenv(
    "TESSERACT_PATH",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)
if os.path.exists(CUSTOM_TESS):
    pytesseract.pytesseract.tesseract_cmd = CUSTOM_TESS
# otherwise pytesseract will try system default


# ─────────────────────────────────────────
# FASTAPI APP / CORS
# ─────────────────────────────────────────

app = FastAPI(title="CyberGuard Backend API")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# IN-MEMORY GLOBAL MODELS
# ─────────────────────────────────────────

harassment_model: Optional[LogisticRegression] = None
harassment_vectorizer: Optional[TfidfVectorizer] = None
tamper_model: Optional[LogisticRegression] = None


def load_models():
    """
    Load trained models from disk if available.
    If not, globals stay None and we fall back to heuristics.
    """
    global harassment_model, harassment_vectorizer, tamper_model

    if os.path.exists(HARASSMENT_MODEL_PATH) and os.path.exists(
        HARASSMENT_VECTORIZER_PATH
    ):
        try:
            harassment_model = joblib.load(HARASSMENT_MODEL_PATH)
            harassment_vectorizer = joblib.load(HARASSMENT_VECTORIZER_PATH)
        except Exception:
            harassment_model = None
            harassment_vectorizer = None

    if os.path.exists(TAMPER_MODEL_PATH):
        try:
            tamper_model = joblib.load(TAMPER_MODEL_PATH)
        except Exception:
            tamper_model = None


# load once on startup
load_models()


# ─────────────────────────────────────────
# OCR
# ─────────────────────────────────────────

def ocr_extract_text(pil_img: Image.Image) -> str:
    """
    Run OCR on the given PIL image and return extracted text.
    """
    try:
        text = pytesseract.image_to_string(pil_img)
        return text.strip()
    except Exception as e:
        return f"[OCR ERROR: {e}]"


# ─────────────────────────────────────────
# HARASSMENT / COERCION DETECTION (SCREENSHOT ANALYZER)
# ─────────────────────────────────────────

# Rule-based keyword list for abuse, coercion, sexual pressure, blackmail.
ABUSE_PATTERNS = [
    "kill yourself",
    "kms",
    "i own you",
    "your tits are mine",
    "send nudes",
    "i'll leak",
    "i will leak",
    "i will expose",
    "i'll expose",
    "slut",
    "whore",
    "bitch",
    "shut up and do what i say",
    "then your tits are mine",
    "you are mine",
    "you belong to me",
    "i'll ruin you",
    "i'll destroy you",
    "i will ruin you",
    "i will destroy you",
    "fuck you",
    "getting your ass fucked",
    "piss off",
]


def analyze_harassment_rule(text: str) -> Dict[str, Any]:
    """
    Simple string-based coercion / abuse detector.
    Returns:
      harassment (bool)
      harassment_score (int)
      harassment_phrases (List[str])
      harassment_probability (float)
    """
    lower = text.lower()
    hits = [p for p in ABUSE_PATTERNS if p in lower]
    harassment_score = len(hits)
    harassment_flag = harassment_score > 0

    return {
        "harassment": harassment_flag,
        "harassment_score": harassment_score,
        "harassment_phrases": hits,
        "harassment_probability": 1.0 if harassment_flag else 0.0,
    }


def analyze_harassment_ml(text: str) -> Dict[str, Any]:
    """
    If we have a trained harassment model, use it.
    Otherwise fall back to the rule-based detection above.
    """
    global harassment_model, harassment_vectorizer

    if harassment_model is None or harassment_vectorizer is None:
        return analyze_harassment_rule(text)

    vec = harassment_vectorizer.transform([text])
    try:
        proba = harassment_model.predict_proba(vec)[0][1]  # probability abusive
    except Exception:
        return analyze_harassment_rule(text)

    harassment_flag = proba >= 0.5
    rule_info = analyze_harassment_rule(text)

    return {
        "harassment": harassment_flag,
        "harassment_score": rule_info["harassment_score"],
        "harassment_phrases": rule_info["harassment_phrases"],
        "harassment_probability": float(round(proba, 3)),
    }


# ─────────────────────────────────────────
# SCAM / FRAUD DETECTION (SCREENSHOT ANALYZER)
# ─────────────────────────────────────────

SCAM_PATTERNS = [
    "part-time online opportunities",
    "part time online opportunities",
    "we are currently offering part-time online opportunities",
    "hiring instagram follow",
    "your position is a data provider",
    "for each following you will earn",
    "you will immediately receive",
    "simple trial",
    "first reward",
    "follow the given instagram accounts",
    "send the money via crypto",
    "bank server is down",
    "usdt",
    "instant payout",
    "work from home task",
    "₹",
    "$",
    "earn money quickly",
    "earn extra money",
]


def analyze_scam_text(text: str) -> Dict[str, Any]:
    """
    Lightweight scam lure detector:
    money bait, instant payout, crypto excuse, etc.
    """
    lower = text.lower()
    hits = [p for p in SCAM_PATTERNS if p in lower]

    scam_score = len(hits)
    scam_flag = scam_score > 0
    scam_prob = min(1.0, scam_score / 3.0)

    reason = (
        "Language consistent with recruitment / payout / crypto-style scam offer."
        if scam_flag
        else "No strong scam indicators detected."
    )

    return {
        "scam": scam_flag,
        "scam_score": scam_score,
        "matched_phrases": hits,
        "scam_probability": float(round(scam_prob, 3)),
        "scam_reason": reason,
    }


# ─────────────────────────────────────────
# TAMPER / FORGERY DETECTION (SCREENSHOT ANALYZER)
# ─────────────────────────────────────────

def ela_score(pil_img: Image.Image, quality: int = 90) -> float:
    """
    Error Level Analysis heuristic.

    1. Save as JPEG (quality=90)
    2. Compare vs original
    3. 95th percentile of amplified difference

    High score → parts compress differently → possible paste/splice.
    BUT chat UIs (WhatsApp etc.) naturally trigger this sometimes.
    """
    img_rgb = pil_img.convert("RGB")

    buf = io.BytesIO()
    img_rgb.save(buf, "JPEG", quality=quality)
    buf.seek(0)

    jpeg_again = Image.open(buf).convert("RGB")

    diff = ImageChops.difference(img_rgb, jpeg_again)
    enhancer = ImageEnhance.Brightness(diff)
    diff_enhanced = enhancer.enhance(10)

    diff_arr = np.array(diff_enhanced).astype("float32")
    max_channel = diff_arr.max(axis=2)
    p95 = float(np.percentile(max_channel, 95))
    return p95


def detect_redaction_blocks(pil_img: Image.Image) -> Dict[str, Any]:
    """
    Detect obvious manual censor bars:
    - big white paint blocks
    - big black bars
    """
    w, h = pil_img.size
    if w == 0 or h == 0:
        return {
            "redacted": False,
            "redaction_reason": "Invalid image size",
            "redaction_ratio_white": 0.0,
            "redaction_ratio_black": 0.0,
        }

    scale_w = max(1, w // 4)
    scale_h = max(1, h // 4)

    small = pil_img.convert("L").resize((scale_w, scale_h))
    arr = np.array(small, dtype="float32")

    mask_white = arr >= 240  # very bright, like a painted white box
    mask_black = arr <= 15   # very dark, like a blackout box

    white_ratio = float(mask_white.mean())
    black_ratio = float(mask_black.mean())

    REDACT_THRESHOLD = 0.02  # 2% of downscaled frame

    redacted_flag = False
    reason = "No obvious redaction blocks found."

    if white_ratio > REDACT_THRESHOLD:
        redacted_flag = True
        reason = "White block/overlay detected that likely hides content."
    elif black_ratio > REDACT_THRESHOLD:
        redacted_flag = True
        reason = "Black bar/overlay detected that likely hides content."

    return {
        "redacted": redacted_flag,
        "redaction_reason": reason,
        "redaction_ratio_white": round(white_ratio, 4),
        "redaction_ratio_black": round(black_ratio, 4),
    }


def analyze_tampering_rule(pil_img: Image.Image) -> Dict[str, Any]:
    """
    Baseline splice suspicion using only ELA.
    """
    try:
        score = ela_score(pil_img)
    except Exception as e:
        return {
            "tampered_splice": False,
            "tamper_score": -1.0,
            "tamper_reason": f"ELA_ERROR: {e}",
            "tamper_probability": 0.0,
        }

    threshold = 45.0
    splice_flag = score >= threshold

    return {
        "tampered_splice": splice_flag,
        "tamper_score": round(score, 2),
        "tamper_reason": (
            "Compression anomalies suggest possible pasted/edited regions."
            if splice_flag
            else "No strong localized compression anomalies detected."
        ),
        "tamper_probability": float(
            round(min(max(score / 100.0, 0.0), 1.0), 3)
        ),
    }


def looks_like_chat_screenshot(pil_img: Image.Image, ocr_text: str) -> bool:
    """
    Heuristic: guess if this is a normal phone chat screenshot.
    We soften 'tampered' for real chat screenshots, because UI elements
    (gradients, emojis, stickers, header bars) trigger high ELA naturally.
    """
    w, h = pil_img.size

    portraitish = False
    if h > 0 and (h / max(w, 1.0)) >= 1.4:
        portraitish = True

    lower = ocr_text.lower()
    has_chat_language = any(
        key in lower
        for key in [
            "am",
            "pm",
            "last seen",
            "october",
            "today",
            "yesterday",
            "hi ",
            "i'm the hr manager",
            "would you like to get started",
        ]
    )

    return portraitish and has_chat_language


def analyze_tampering_ml(pil_img: Image.Image, ocr_text: str) -> Dict[str, Any]:
    """
    Merge:
    - ELA splice suspicion
    - Redaction block detection
    - Optional ML tamper model
    - Chat screenshot false-positive suppression
    """
    global tamper_model

    img_w, img_h = pil_img.size

    # (1) splice suspicion
    splice_info = analyze_tampering_rule(pil_img)
    splice_flag = splice_info["tampered_splice"]

    # (2) redaction suspicion
    redact_info = detect_redaction_blocks(pil_img)
    redaction_flag = redact_info["redacted"]

    # (3) optional ML
    ml_prob = splice_info["tamper_probability"]
    if tamper_model is not None:
        feat_vec = np.array(
            [[splice_info["tamper_score"], float(img_w), float(img_h)]],
            dtype="float32",
        )
        try:
            proba = tamper_model.predict_proba(feat_vec)[0][1]
            ml_prob = float(round(proba, 3))
        except Exception:
            pass

    # (4) chat false-positive mitigation
    chat_like = looks_like_chat_screenshot(pil_img, ocr_text)

    if redaction_flag:
        # definitely edited (info was covered)
        final_tampered = True
        final_reason = redact_info["redaction_reason"]

    elif splice_flag:
        if chat_like:
            # looks like a real chat screenshot;
            # high ELA alone shouldn't autocall it fake
            final_tampered = False
            final_reason = (
                "High compression differences found. These can occur in normal "
                "chat screenshots (UI elements, stickers, emojis). "
                "No obvious manual redaction detected."
            )
        else:
            final_tampered = True
            final_reason = splice_info["tamper_reason"]
    else:
        # clean-ish ELA, no redaction
        final_tampered = False
        final_reason = splice_info["tamper_reason"]

    return {
        "tampered": final_tampered,
        "tamper_score": splice_info["tamper_score"],
        "tamper_reason": final_reason,
        "tamper_probability": ml_prob,
        "redacted": redaction_flag,
        "redaction_reason": redact_info["redaction_reason"],
        "chat_like": chat_like,
    }


# ─────────────────────────────────────────
# STALKER / CYBERSTALKING DETECTION HELPERS
# (used by /analyze_chat)
# ─────────────────────────────────────────

def _split_lines_to_messages(text: str) -> List[Dict[str, Any]]:
    """
    Break a blob of text into per-line pseudo-messages.
    Output: [{"text": "..."}]
    """
    msgs = []
    if not text:
        return msgs
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        msgs.append({"text": s})
    return msgs


def _csv_to_messages(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Reads a CSV export of chat logs.
    Tries columns like "message","text","content" for message body,
    plus "timestamp"/"time"/"date" and "sender"/"from".
    """
    msgs: List[Dict[str, Any]] = []
    try:
        s = file_bytes.decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(s))
        for row in reader:
            msg = (
                row.get("message")
                or row.get("text")
                or row.get("content")
                or ""
            )
            ts = row.get("timestamp") or row.get("time") or row.get("date")
            sender = row.get("sender") or row.get("from")
            item = {
                "text": str(msg).strip(),
                "timestamp": ts,
                "sender": sender,
            }
            if item["text"]:
                msgs.append(item)
    except Exception:
        # fallback: naive line split
        try:
            fallback = file_bytes.decode("utf-8", errors="ignore")
            msgs = _split_lines_to_messages(fallback)
        except Exception:
            msgs = []
    return msgs


# Buckets for stalker / obsessive behavior:
STALKING_BOUNDARY_PHRASES = [
    "leave me alone",
    "stop texting",
    "stop calling",
    "do not contact",
    "don't contact",
    "i'm not interested",
    "no means no",
    "please stop",
    "go away",
    "stop messaging",
    "block you",
    "i will block",
    "restraining order",
    "i'll call the police",
]

STALKING_PERSISTENCE_PHRASES = [
    "hi",
    "hello",
    "are you there",
    "answer me",
    "why not responding",
    "please",
    "talk to me",
    "pick up",
    "call me",
    "text me",
    "where are you",
    "reply",
    "good morning",
    "good night",
]

STALKING_MONITORING_PHRASES = [
    "last seen",
    "i saw you",
    "i'm outside",
    "im outside",
    "i am outside",
    "outside your",
    "i'm near your",
    "i know where you live",
    "i know your address",
    "share location",
    "turn on location",
    "waiting outside",
    "come outside",
    "follow you",
    "i'm watching",
    "i'm tracking",
    "find your location",
    "gps",
    "location on",
]

STALKING_THREAT_PHRASES = [
    "i will find you",
    "i'll find you",
    "i will come",
    "i'm coming",
    "come to your house",
    "hurt you",
    "kill you",
    "i'll kill",
    "beat you",
    "break your",
    "burn",
    "stab",
    "ruin your life",
    "destroy you",
    "i'll leak",
    "i will leak",
    "post your photos",
    "revenge porn",
]

STALKING_EVASION_PHRASES = [
    "new number",
    "my other account",
    "backup account",
    "alt account",
    "you blocked me",
    "unblock me",
    "this is my new",
    "why did you block me",
    "you can't ignore me",
]

STALKING_DOXX_PHRASES = [
    "your address",
    "address is",
    "your phone number",
    "your parents",
    "your work",
    "your office",
    "your school",
    "i told your",
    "i will tell your",
    "send your location",
    "share your live location",
]


def analyze_stalker_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Heuristic stalker risk analysis across a set of messages.
    Returns a dict with:
      classification: "normal" | "concerning" | "stalking"
      risk_score: 0..1
      indicators: list of buckets triggered
      counts, matched_phrases
      advice
      meta
    """
    text_all = " \n ".join(m.get("text", "") for m in messages).lower()

    counts = {
        "boundary": 0,
        "persistence": 0,
        "monitoring": 0,
        "threats": 0,
        "evasion": 0,
        "doxxing": 0,
    }
    matched = {k: [] for k in counts.keys()}

    def scan(phrases: List[str], key: str):
        total = 0
        for p in phrases:
            if p in text_all:
                total += text_all.count(p)
                matched[key].append(p)
        counts[key] = total

    # scan each category
    scan(STALKING_BOUNDARY_PHRASES, "boundary")
    scan(STALKING_PERSISTENCE_PHRASES, "persistence")
    scan(STALKING_MONITORING_PHRASES, "monitoring")
    scan(STALKING_THREAT_PHRASES, "threats")
    scan(STALKING_EVASION_PHRASES, "evasion")
    scan(STALKING_DOXX_PHRASES, "doxxing")

    # If user said "stop / leave me alone" but messages keep coming,
    # that's worse.
    persistence_after_no = counts["boundary"] > 0 and counts["persistence"] > 0

    # crude risk score 0..100
    score = 0
    score += min(30, counts["threats"] * 15)        # threats weigh heavy
    score += min(20, counts["monitoring"] * 10)
    score += min(20, counts["doxxing"] * 10)
    score += min(15, counts["persistence"] * 5)
    score += min(10, counts["evasion"] * 5)
    score += min(10, counts["boundary"] * 5)

    if persistence_after_no:
        score += 15

    score = max(0, min(100, score))

    # classification buckets
    if score >= 60:
        classification = "stalking"
    elif score >= 35:
        classification = "concerning"
    else:
        classification = "normal"

    # advice for user safety
    advice = []
    if classification != "normal":
        advice.extend([
            "Preserve evidence (screenshots, export chat).",
            "Avoid responding; block and report on the platform.",
            "If threats mention physical harm, contact local authorities.",
        ])
        if persistence_after_no:
            advice.append(
                "User continued contact after a clear 'stop' — document this for escalation."
            )

    # indicators for UI
    indicators = []
    for key, phrases in matched.items():
        if phrases:
            indicators.append({
                "type": key,
                "phrases": sorted(list(set(phrases))),
                "count": counts[key],
            })

    return {
        "classification": classification,
        "risk_score": round(score / 100.0, 2),  # 0..1
        "indicators": indicators,
        "counts": counts,
        "matched_phrases": matched,
        "advice": advice,
        "meta": {
            "messages_analyzed": len(messages),
        },
    }


# ─────────────────────────────────────────
# EXTRA THREAT RULES FOR STALKER DETECTION
# (physical proximity, blackmail, payout scams)
# ─────────────────────────────────────────

def high_level_behavior_scan(text: str) -> Dict[str, Any]:
    """
    Scan raw text blob for high-severity behaviors:
    - physical proximity / surveillance
    - boundary violation
    - blackmail / coercion
    - payout / crypto scam

    Returns:
      {
        "anomalies": [ {type,severity,message,timestamp}, ... ],
        "recommendations": [...],
        "risk_components": [float, ...]  # each 0..1-ish
      }
    """
    lower = text.lower()
    anomalies: List[Dict[str, Any]] = []
    recs: List[str] = []
    risk_components: List[float] = []

    # Physical proximity / surveillance
    if any(
        phrase in lower
        for phrase in [
            "i know where you work",
            "i know where you live",
            "i know where u live",
            "i'm outside",
            "im outside",
            "i am outside",
            "i'll wait for you outside",
            "ill wait for you outside",
            "i'm waiting outside",
            "i am waiting outside",
        ]
    ):
        anomalies.append({
            "type": "Physical proximity / surveillance",
            "severity": "high",
            "message": "Possible real-world presence / tracking ('i'll wait for you outside', 'i know where you work').",
            "timestamp": "Just now",
        })
        recs.append("Do not meet this person in real life. Tell a trusted friend and consider escalating if they claim to be nearby.")
        recs.append("Document these messages (screenshots + timestamps).")
        risk_components.append(0.9)

    # Boundary violation / obsessive persistence
    if any(
        phrase in lower
        for phrase in [
            "i've told him to leave me alone",
            "i told him to leave me alone",
            "i told her to leave me alone",
            "stop messaging me",
            "leave me alone",
            "why are you ignoring me",
            "if you ignore me again",
            "i'll keep messaging you",
            "ill keep messaging you",
            "answer me now",
            "why aren't you answering",
            "why arent you answering",
        ]
    ):
        anomalies.append({
            "type": "Boundary violation / stalking persistence",
            "severity": "medium",
            "message": "Repeated unwanted contact or refusal to respect 'stop'.",
            "timestamp": "Just now",
        })
        recs.append("Block or mute the account if it is safe to do so.")
        recs.append("Preserve the conversation history in case escalation continues.")
        risk_components.append(0.6)

    # Threats / coercion / blackmail
    if any(
        phrase in lower
        for phrase in [
            "i'll leak",
            "i will leak",
            "i'll expose",
            "i will expose",
            "i'll post this everywhere",
            "i will post this everywhere",
            "i'll ruin you",
            "i will ruin you",
            "send me pics or i'll",
            "send me pictures or i'll",
            "send me nudes or i'll",
            "your tits are mine",
            "you are mine",
        ]
    ):
        anomalies.append({
            "type": "Threat / Intimidation / Blackmail",
            "severity": "high",
            "message": "Coercion or blackmail detected ('i'll leak', 'i'll ruin you', forced sexual pressure).",
            "timestamp": "Just now",
        })
        recs.append("Do not send photos, money, or personal info. Capture screenshots immediately.")
        recs.append("If sexual coercion or underage content is involved, escalate immediately to a trusted adult / relevant authority.")
        risk_components.append(0.85)

    # Scam / payout lure / crypto excuse
    if any(
        phrase in lower
        for phrase in [
            "part-time online opportunities",
            "part time online opportunities",
            "hiring instagram follow",
            "your position is a data provider",
            "for each following you will earn",
            "you will immediately receive",
            "send the money via crypto",
            "bank server is down",
            "easy payout",
            "first reward",
        ]
    ):
        anomalies.append({
            "type": "Payment scam / social engineering",
            "severity": "medium",
            "message": "Likely scam / payout lure / crypto excuse.",
            "timestamp": "Just now",
        })
        recs.append("Do not send money or crypto. Stop engaging with this account.")
        recs.append("Do not move to off-platform payment methods.")
        risk_components.append(0.5)

    return {
        "anomalies": anomalies,
        "recommendations": recs,
        "risk_components": risk_components,
    }


def merge_stalker_and_behavior(
    stalker_res: Dict[str, Any],
    behavior_scan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Combine:
     - stalker_res (from analyze_stalker_messages)
     - behavior_scan (from high_level_behavior_scan)

    and produce final fields for /analyze_chat output.
    """
    # Prepare anomaly list for UI:
    # 1) high-level anomalies (physical proximity / blackmail / scam etc.)
    ui_anomalies: List[Dict[str, Any]] = list(behavior_scan["anomalies"])

    # 2) bucket indicators from stalker_res -> convert to anomalies
    for ind in stalker_res.get("indicators", []):
        bucket_type = ind.get("type", "behavior")
        phrases = ind.get("phrases", [])
        count = ind.get("count", 0)

        severity = "low"
        if bucket_type in ["threats", "monitoring", "doxxing"]:
            severity = "high"
        elif bucket_type in ["persistence", "boundary", "evasion"]:
            severity = "medium"

        msg_preview = f"{bucket_type} x{count}; phrases: {', '.join(phrases[:4])}"
        ui_anomalies.append({
            "type": bucket_type,
            "severity": severity,
            "message": msg_preview,
            "timestamp": "Just now",
        })

    # Merge recommendations (dedupe)
    recs = list(stalker_res.get("advice", [])) + list(behavior_scan["recommendations"])
    deduped_recs: List[str] = []
    seen = set()
    for r in recs:
        if r not in seen:
            deduped_recs.append(r)
            seen.add(r)

    # Risk calculation:
    # - stalker_res["risk_score"] is already 0..1
    # - behavior_scan["risk_components"] is list of ~0..1 pieces
    stalker_risk_component = float(stalker_res.get("risk_score", 0.0))
    behavior_max_component = max(behavior_scan["risk_components"], default=0.1)
    # Take the max — "the scariest thing wins"
    overall_risk = max(stalker_risk_component, behavior_max_component)
    overall_risk = round(float(overall_risk), 3)

    # Map numeric risk to label
    if overall_risk >= 0.9:
        risk_level = "critical"
    elif overall_risk >= 0.7:
        risk_level = "high"
    elif overall_risk >= 0.4:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "risk": overall_risk,
        "risk_level": risk_level,
        "anomalies": ui_anomalies,
        "recommendations": deduped_recs,
    }


# ─────────────────────────────────────────
# RESPONSE MODELS (FOR FASTAPI SCHEMA / DOCS)
# ─────────────────────────────────────────

class ScreenshotAnalysisResult(BaseModel):
    filename: str
    verdict: str
    confidence: float
    ocr_text: str

    harassment: bool
    harassment_score: int
    harassment_phrases: List[str]
    harassment_probability: float

    scam: bool
    scam_score: int
    scam_phrases: List[str]
    scam_probability: float
    scam_reason: str

    tampered: bool
    tamper_score: float
    tamper_reason: str
    tamper_probability: float


class ScreenshotAnalysisResponse(BaseModel):
    results: List[ScreenshotAnalysisResult]


class ChatAnomaly(BaseModel):
    type: str            # e.g. "Physical proximity / surveillance"
    severity: str        # "low" | "medium" | "high"
    message: str         # threat summary or behavior summary
    timestamp: Optional[str] = None  # e.g. "Just now"


class ChatAnalysisResponse(BaseModel):
    risk: float                 # 0..1
    risk_level: str             # "low" | "medium" | "high" | "critical"
    anomalies: List[ChatAnomaly]
    recommendations: List[str]
    analysis_time_ms: int


class TrainModelsResponse(BaseModel):
    harassment_samples_used: int
    tamper_samples_used: int
    harassment_model_trained: bool
    tamper_model_trained: bool
    note: str


# ─────────────────────────────────────────
# /analyze_screenshot
# ─────────────────────────────────────────

@app.post("/analyze_screenshot", response_model=ScreenshotAnalysisResponse)
async def analyze_screenshot(files: List[UploadFile] = File(...)):
    """
    Accepts 1+ screenshots (PNG/JPG/etc). For each:
      - OCR text extraction
      - Harassment / coercion analysis
      - Scam / fraud lure analysis
      - Tampering / redaction / splice analysis (with chat false-positive reduction)
      - Verdict string + confidence

    Verdict:
      "Real"        = looks consistent
      "Suspicious"  = high ELA but looks like normal chat (treat with caution)
      "Edited"      = redacted or clearly spliced (manual modification)

    We ALSO cap tamper score for clean chat screenshots so the UI
    doesn't scream red bars at victims for normal WhatsApp captures.
    """

    results: List[ScreenshotAnalysisResult] = []

    for f in files:
        raw_bytes = await f.read()
        pil_img = Image.open(io.BytesIO(raw_bytes))

        # 1. OCR
        ocr_text = ocr_extract_text(pil_img)

        # 2. Harassment
        har_info = analyze_harassment_ml(ocr_text)

        # 3. Scam detection
        scam_info = analyze_scam_text(ocr_text)

        # 4. Tampering / redaction / splice
        tamp_info = analyze_tampering_ml(pil_img, ocr_text)

        # 5. Verdict + base confidence before OCR penalty
        suspicious_but_not_redacted = (
            (not tamp_info["tampered"])
            and tamp_info["tamper_score"] >= 45.0
            and tamp_info["chat_like"]
        )

        if tamp_info["tampered"]:
            verdict = "Edited"
            base_conf = 0.4
        elif suspicious_but_not_redacted:
            verdict = "Suspicious"
            base_conf = 0.5
        else:
            verdict = "Real"
            base_conf = 0.7

        # downgrade confidence if OCR failed
        if not ocr_text or ocr_text.strip() == "":
            base_conf = min(base_conf, 0.5)

        final_conf = round(base_conf, 3)

        # 6. UI-friendly tamper score / probability
        # If we decided it's not "Edited", AND it looks like chat,
        # AND there's no manual redaction,
        # calm the score down for display.
        tamper_score_out = tamp_info["tamper_score"]
        tamper_prob_out = float(tamp_info["tamper_probability"])

        if verdict in ["Real", "Suspicious"] and tamp_info["chat_like"] and not tamp_info["tampered"]:
            tamper_score_out = min(tamper_score_out, 30.0)
            tamper_prob_out = min(tamper_prob_out, 0.3)

        result_obj = ScreenshotAnalysisResult(
            filename=f.filename,
            verdict=verdict,
            confidence=final_conf,
            ocr_text=ocr_text,

            harassment=har_info["harassment"],
            harassment_score=har_info["harassment_score"],
            harassment_phrases=har_info["harassment_phrases"],
            harassment_probability=float(har_info["harassment_probability"]),

            scam=scam_info["scam"],
            scam_score=scam_info["scam_score"],
            scam_phrases=scam_info["matched_phrases"],
            scam_probability=float(scam_info["scam_probability"]),
            scam_reason=scam_info["scam_reason"],

            tampered=tamp_info["tampered"],
            tamper_score=tamper_score_out,
            tamper_reason=tamp_info["tamper_reason"],
            tamper_probability=tamper_prob_out,
        )

        results.append(result_obj)

    return ScreenshotAnalysisResponse(results=results)


# ─────────────────────────────────────────
# /analyze_chat
# ─────────────────────────────────────────
# This powers the "Stalker Detection" page.
# It returns richer data (risk_level, anomalies, recommendations)
# that the frontend dashboard expects.

@app.post("/analyze_chat", response_model=ChatAnalysisResponse)
async def analyze_chat_endpoint(
    text: Optional[str] = Form(None),
    csv_file: Optional[UploadFile] = File(None),
):
    """
    Analyze stalking / coercion / proximity threats / scams in conversation data.

    Accepts multipart form-data:
      - text: pasted description or chat log
      - csv_file: optional CSV export of chat logs
        (columns like message/text/content/timestamp/sender)

    Returns:
      {
        "risk": 0.82,
        "risk_level": "high",
        "anomalies": [...],
        "recommendations": [...],
        "analysis_time_ms": 142
      }
    """

    t0 = time.time()

    # Step 1: collect messages from text and/or CSV
    messages: List[Dict[str, Any]] = []

    if text:
        messages.extend(_split_lines_to_messages(text))

    raw_joined_text = text or ""

    if csv_file is not None:
        try:
            data = await csv_file.read()
            msgs_from_csv = _csv_to_messages(data)
            messages.extend(msgs_from_csv)

            # also fold CSV messages into raw_joined_text for rule scan
            joined_csv_lines = "\n".join(m.get("text", "") for m in msgs_from_csv)
            raw_joined_text = (raw_joined_text + "\n" + joined_csv_lines).strip()
        except Exception:
            pass

    # If no input, just return "low"
    if not messages and not raw_joined_text:
        t1 = time.time()
        return ChatAnalysisResponse(
            risk=0.1,
            risk_level="low",
            anomalies=[],
            recommendations=["No data provided. Please paste messages or upload a chat export."],
            analysis_time_ms=int((t1 - t0) * 1000),
        )

    # Step 2: baseline stalker-style analysis (boundaries, tracking, etc.)
    stalker_res = analyze_stalker_messages(messages if messages else _split_lines_to_messages(raw_joined_text))

    # Step 3: high level behavior scan (physical proximity, blackmail, scam)
    behavior_scan = high_level_behavior_scan(raw_joined_text)

    # Step 4: merge both into final risk, anomalies, recs
    merged = merge_stalker_and_behavior(stalker_res, behavior_scan)

    t1 = time.time()
    analysis_ms = int((t1 - t0) * 1000)

    # Step 5: build final response object
    anomalies_for_model = [
        ChatAnomaly(
            type=a["type"],
            severity=a["severity"],
            message=a["message"],
            timestamp=a.get("timestamp", "Just now"),
        )
        for a in merged["anomalies"]
    ]

    return ChatAnalysisResponse(
        risk=merged["risk"],
        risk_level=merged["risk_level"],
        anomalies=anomalies_for_model,
        recommendations=merged["recommendations"],
        analysis_time_ms=analysis_ms,
    )


# ─────────────────────────────────────────
# TRAINING HELPERS
# ─────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Read newline-delimited JSON objects from a file.

    harassment_samples.jsonl lines:
      {"text": "...", "label": 1}

    tamper_samples.jsonl lines:
      {"ela":63.2,"res_w":1080,"res_h":1920,"label":1}
    """
    items: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                items.append(obj)
            except json.JSONDecodeError:
                continue
    return items


def train_harassment_model(samples: List[Dict[str, Any]]) -> bool:
    """
    Train harassment/coercion classifier using TF-IDF + LogisticRegression.
    Saves:
      models/harassment_model.pkl
      models/harassment_vectorizer.pkl
    """
    global harassment_model, harassment_vectorizer

    if not samples:
        return False

    texts = [s.get("text", "") for s in samples]
    labels = [int(s.get("label", 0)) for s in samples]

    if len(set(labels)) < 2:
        # need both classes 0 and 1
        return False

    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=5000,
    )
    X = vec.fit_transform(texts)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X, labels)

    joblib.dump(clf, HARASSMENT_MODEL_PATH)
    joblib.dump(vec, HARASSMENT_VECTORIZER_PATH)

    harassment_model = clf
    harassment_vectorizer = vec

    return True


def train_tamper_model(samples: List[Dict[str, Any]]) -> bool:
    """
    Train tamper classifier (splicing suspicion) on numeric features.

    samples like:
      {"ela":63.2,"res_w":1080,"res_h":1920,"label":1}

    Feature vector = [ela_score, width, height]
    """
    global tamper_model

    if not samples:
        return False

    X_list = []
    y_list = []

    for s in samples:
        try:
            ela_val = float(s.get("ela", 0.0))
            rw = float(s.get("res_w", 0))
            rh = float(s.get("res_h", 0))
            label = int(s.get("label", 0))
        except (TypeError, ValueError):
            continue

        X_list.append([ela_val, rw, rh])
        y_list.append(label)

    if not X_list:
        return False
    if len(set(y_list)) < 2:
        return False

    X_arr = np.array(X_list, dtype="float32")
    y_arr = np.array(y_list, dtype="int32")

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_arr, y_arr)

    joblib.dump(clf, TAMPER_MODEL_PATH)
    tamper_model = clf

    return True


# ─────────────────────────────────────────
# /train_models
# ─────────────────────────────────────────

@app.post("/train_models", response_model=TrainModelsResponse)
async def train_models():
    """
    Retrain:
      - harassment / coercion text model
      - tamper (splicing) model

    using labeled data under:
      data/harassment_samples.jsonl
      data/tamper_samples.jsonl

    After training:
      /analyze_screenshot uses new models automatically.
    """

    harassment_samples = load_jsonl(HARASSMENT_DATA_PATH)
    tamper_samples = load_jsonl(TAMPER_DATA_PATH)

    trained_har = train_harassment_model(harassment_samples)
    trained_tamp = train_tamper_model(tamper_samples)

    note_parts = []
    if trained_har:
        note_parts.append("harassment model updated")
    else:
        note_parts.append(
            "harassment model not updated (insufficient or invalid data)"
        )

    if trained_tamp:
        note_parts.append("tamper model updated")
    else:
        note_parts.append(
            "tamper model not updated (insufficient or invalid data)"
        )

    return TrainModelsResponse(
        harassment_samples_used=len(harassment_samples),
        tamper_samples_used=len(tamper_samples),
        harassment_model_trained=trained_har,
        tamper_model_trained=trained_tamp,
        note="; ".join(note_parts),
    )
