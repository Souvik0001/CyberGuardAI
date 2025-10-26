import os
import io
import json
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File
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
# HARASSMENT / COERCION DETECTION
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
# SCAM / FRAUD DETECTION
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
# TAMPER / FORGERY DETECTION
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
# RESPONSE MODELS (for FastAPI schema / docs)
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


class ChatRequest(BaseModel):
    messages: List[str]


class ChatResponse(BaseModel):
    classification: str
    risk_score: float


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
    Accepts 1+ screenshots (PNG/JPG/etc).
    For each:
      - OCR text extraction
      - Harassment / coercion analysis
      - Scam / fraud lure analysis
      - Tampering / redaction / splice analysis (with chat false-positive reduction)
      - Verdict string + confidence

    Verdict:
      "Real"        = looks consistent
      "Suspicious"  = high ELA but looks like normal chat (treat with caution)
      "Edited"      = redacted or clearly spliced (manual modification)

    We ALSO clamp the tamper score for clean chat screenshots so the UI
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

        # downgrade confidence if OCR failed (super blurry / weird crop)
        if not ocr_text or ocr_text.strip() == "":
            base_conf = min(base_conf, 0.5)

        final_conf = round(base_conf, 3)

        # 6. UI-friendly tamper score / probability
        # If we decided it's not "Edited", AND it looks like chat,
        # AND there's no manual redaction, calm the score down for display.
        tamper_score_out = tamp_info["tamper_score"]
        tamper_prob_out = float(tamp_info["tamper_probability"])

        if verdict in ["Real", "Suspicious"] and tamp_info["chat_like"] and not tamp_info["tampered"]:
            # Cap them so the UI bar doesn't scream "50" on honest chats.
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
            scam_reason=scam_info["scam_reason"] ,

            tampered=tamp_info["tampered"],
            tamper_score=tamper_score_out,
            tamper_reason=tamp_info["tamper_reason"],
            tamper_probability=tamper_prob_out,
        )

        results.append(result_obj)

    return ScreenshotAnalysisResponse(results=results)


# ─────────────────────────────────────────
# /analyze_chat  (stub)
# ─────────────────────────────────────────

@app.post("/analyze_chat", response_model=ChatResponse)
async def analyze_chat(req: ChatRequest):
    """
    Simple stub for chat-level stalker threat.
    You can upgrade this to also call harassment/scam detectors on full convo.
    """
    full_text = " ".join(req.messages).lower()
    risk = 0.1
    if (
        "follow you" in full_text
        or "i am outside" in full_text
        or "i'll leak" in full_text
        or "i will leak" in full_text
        or "send the money via crypto" in full_text
    ):
        risk = 0.8

    return ChatResponse(
        classification="harassment" if risk > 0.5 else "normal",
        risk_score=risk,
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
        # we need both classes present (0 and 1)
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
