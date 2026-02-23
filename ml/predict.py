"""
predict.py — Fake Profile Predictor
=====================================
Can be called two ways (server.js uses args):
  1. python predict.py "https://instagram.com/username"   ← used by server
  2. echo '{"url":"..."}' | python predict.py             ← stdin fallback

Optional second arg — JSON profile metadata (all fields optional):
  python predict.py "https://x.com/john" '{"followers":120,"following":80,"posts":30,"has_pic":1,"verified":0,"bio_len":45}'

Outputs JSON to stdout:
  {"prediction":"Fake"|"Real", "confidence":0.xx, "method":"...", "username":"...", "features":{...}}

Priority:
  1. Hard rules  — catches obvious cases instantly, runs BEFORE the ML model
  2. ML model    — uses model.pkl + scaler.pkl if both classes are present
  3. Heuristic   — calibrated fallback, always available
"""

import sys
import json
import re
import math
import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent
MODEL_PATH    = SCRIPT_DIR / "model.pkl"
SCALER_PATH   = SCRIPT_DIR / "scaler.pkl"
FEATURES_PATH = SCRIPT_DIR / "features.json"

# ── Try to load trained model ─────────────────────────────────────────────────
_model         = None
_scaler        = None
_feature_names = None
_model_metrics = {}
_using_ml      = False

try:
    import joblib
    import numpy as np

    if MODEL_PATH.exists() and SCALER_PATH.exists():
        _model  = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)

        if hasattr(_model, "classes_") and 0 in _model.classes_ and 1 in _model.classes_:
            if FEATURES_PATH.exists():
                with open(FEATURES_PATH) as f:
                    _meta          = json.load(f)
                    _feature_names = _meta.get("features", [])
                    _model_metrics = _meta.get("metrics", {})
            _using_ml = True
        else:
            _model = _scaler = None   # broken / single-class model
except Exception:
    pass


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS  (kept in sync with train_model.py)
# ══════════════════════════════════════════════════════════════════════════════
SUSPICIOUS_KW = {
    "bot", "fake", "scam", "spam", "hack", "temp", "xyz", "anon",
    "xxx", "adult", "promo", "giveaway", "f4f", "l4l", "onlyfan",
    "follow4follow", "like4like", "viral", "offi", "official2",
}
VOWELS = set("aeiouAEIOU")


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def _entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: dict = {}
    for c in s.lower():
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def extract_username(url: str) -> str:
    """Pull the last meaningful path segment from a URL or plain username."""
    url   = url.strip().rstrip("/")
    url   = re.split(r"[?#]", url)[0]
    parts = [p for p in url.split("/")
             if p and not p.startswith("http") and "." not in p]
    return parts[-1] if parts else url.split("/")[-1]


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
#  22 features — superset of the 20 used during training.
#  The ML path selects exactly the names stored in features.json.
#  The heuristic path uses all 22.
# ══════════════════════════════════════════════════════════════════════════════
def build_features(username: str, profile: dict = None) -> dict:
    """
    Return a dict of numeric features.

    profile keys (all optional):
        followers, following, posts, has_pic, verified, bio_len
    """
    if profile is None:
        profile = {}

    u        = username.strip()
    n        = len(u)
    lower    = u.lower()
    digits   = sum(c.isdigit() for c in u)
    alpha    = sum(c.isalpha() for c in u)
    vowels_n = sum(c in VOWELS for c in u)
    d_ratio  = digits / (n + 1e-5)

    followers = float(profile.get("followers", 0) or 0)
    following = float(profile.get("following", 0) or 0)
    posts     = float(profile.get("posts",     0) or 0)

    return {
        # ── username shape ────────────────────────────────────────────────
        "username_len":             min(n, 40),
        "digit_count":              digits,
        "digit_ratio":              round(d_ratio, 4),
        "special_count":            sum(c in "._-" for c in u),
        "alpha_count":              alpha,
        "vowel_ratio":              round(vowels_n / (n + 1e-5), 4),
        "entropy":                  round(_entropy(u), 4),

        # ── boolean username signals ──────────────────────────────────────
        "all_digits":               int(u.isdigit()),
        "no_alpha":                 int(alpha == 0),
        "very_short":               int(n <= 3),
        "only_one_char":            int(n == 1),
        "no_vowels":                int(n >= 5 and vowels_n == 0),
        "trailing_digits4":         int(bool(re.search(r"\d{4,}$", u))),
        "digit_block":              int(bool(re.search(r"[A-Za-z]{2,}\d{4,}", u))),
        "repeating_chars":          int(bool(re.search(r"(.)\1{3,}", u))),
        "suspicious_keyword":       int(any(kw in lower for kw in SUSPICIOUS_KW)),
        "uuid_like":                int(bool(re.match(r"^[a-f0-9]{8,}$", lower))),

        # ── profile metadata ──────────────────────────────────────────────
        "has_profile_pic":          float(profile.get("has_pic",  1)),
        "verified":                 float(profile.get("verified", 0)),
        "bio_length":               min(float(profile.get("bio_len", 0)), 500),
        "follower_following_ratio": round(min(followers / (following + 1), 50), 4),
        "posts_per_follower":       round(min(posts / (followers + 1), 50), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  HARD-RULE PRE-CHECK
#  Runs BEFORE the ML model — catches cases where username signals are
#  unambiguous regardless of what the model was trained on.
# ══════════════════════════════════════════════════════════════════════════════
def hard_rules(username: str):
    """
    Returns (is_fake: bool, confidence: float, reason: str).
    Returns (None, 0.0, '') if no rule fires.
    """
    u     = username.strip()
    lower = u.lower()
    n     = len(u)

    digits   = sum(c.isdigit() for c in u)
    vowels_n = sum(c in VOWELS for c in u)
    d_ratio  = digits / (n + 1e-5)

    # ── Absolute rules ────────────────────────────────────────────────────
    if n <= 1:
        return True, 0.97, "single_char_username"

    if u.isdigit():
        return True, 0.95, "pure_digit_username"

    if n == 2 and digits == 2:
        return True, 0.92, "two_digit_username"

    if all(c in "._-" for c in u):
        return True, 0.96, "only_special_chars"

    # UUID-like hex string (e.g. "a3f9c2e1b7d04582")
    if re.match(r"^[a-f0-9]{8,}$", lower):
        return True, 0.93, "uuid_like_username"

    # ── Compound rules — suspicious keyword + at least one digit signal ──
    # These catch patterns like: botaccount9274, fakeperson123456, spamuser88
    has_sus_kw      = any(kw in lower for kw in SUSPICIOUS_KW)
    has_trailing4   = bool(re.search(r"\d{4,}$", u))
    has_digit_block = bool(re.search(r"[A-Za-z]{2,}\d{4,}", u))

    if has_sus_kw and (has_trailing4 or has_digit_block or d_ratio > 0.30):
        return True, 0.91, "suspicious_keyword_plus_digits"

    # Suspicious keyword with no vowels (e.g. "btkr_xyz", "spmbrt")
    if has_sus_kw and n >= 5 and vowels_n == 0:
        return True, 0.89, "suspicious_keyword_no_vowels"

    # Suspicious keyword alone — still fairly strong signal
    if has_sus_kw:
        return True, 0.82, "suspicious_keyword"

    return None, 0.0, ""


# ══════════════════════════════════════════════════════════════════════════════
#  ML MODEL PATH
# ══════════════════════════════════════════════════════════════════════════════
def predict_with_model(username: str, profile: dict = None) -> dict:
    """Use the trained RandomForest model for prediction."""
    fv  = build_features(username, profile)

    # Select and order features exactly as seen during training
    if _feature_names:
        vec = [fv.get(name, 0.0) for name in _feature_names]
    else:
        vec = list(fv.values())

    import numpy as np
    X   = np.array(vec, dtype=float).reshape(1, -1)
    X_s = _scaler.transform(X)

    pred  = int(_model.predict(X_s)[0])
    proba = float(_model.predict_proba(X_s)[0][pred])

    return {
        "prediction": "Fake" if pred == 1 else "Real",
        "confidence": round(min(max(proba, 0.51), 0.98), 2),
        "method":     "ml_model",
        "username":   username,
        "features":   {k: round(float(v), 3) for k, v in fv.items()},
    }


# ══════════════════════════════════════════════════════════════════════════════
#  HEURISTIC FALLBACK
#  Used when no trained model is available.
# ══════════════════════════════════════════════════════════════════════════════
WEIGHTS = {
    "only_one_char":        0.40,
    "very_short":           0.25,
    "all_digits":           0.25,
    "no_alpha":             0.22,
    "trailing_digits4":     0.20,
    "uuid_like":            0.20,
    "suspicious_keyword":   0.30,
    "no_vowels":            0.14,
    "digit_block":          0.16,
    "repeating_chars":      0.10,
    "high_digit_ratio":     0.14,
    "high_entropy":         0.09,
}
MAX_W       = sum(WEIGHTS.values())
FAKE_THRESH = 0.12


def predict_heuristic(username: str, profile: dict = None) -> dict:
    fv = build_features(username, profile)

    signals = dict(fv)
    signals["high_digit_ratio"] = (
        min(fv["digit_ratio"] / 0.35, 1.0) if fv["digit_ratio"] > 0.30 else 0.0
    )
    signals["high_entropy"] = max(0.0, min((fv["entropy"] - 3.4) / 0.8, 1.0))

    raw  = sum(WEIGHTS.get(k, 0) * float(v) for k, v in signals.items())
    norm = raw / MAX_W

    x         = (norm - FAKE_THRESH) * 12
    fake_prob = 1.0 / (1.0 + math.exp(-x))
    is_fake   = norm >= FAKE_THRESH

    if is_fake:
        confidence = 0.52 + fake_prob * 0.43
    else:
        confidence = 0.52 + (1 - fake_prob) * 0.43
    confidence = round(min(max(confidence, 0.52), 0.97), 2)

    return {
        "prediction": "Fake" if is_fake else "Real",
        "confidence": confidence,
        "method":     "heuristic",
        "username":   username,
        "features":   {k: round(float(v), 3) for k, v in fv.items()},
        "score":      round(norm, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════
def predict(url: str, profile: dict = None) -> dict:
    """
    Full prediction pipeline:
      1. Extract username from URL
      2. Apply hard rules  ← always runs, even if ML model is loaded
      3. ML model (if available)
      4. Heuristic fallback
    """
    username = extract_username(url)

    if not username:
        return {
            "prediction": "Real",
            "confidence": 0.55,
            "method":     "default",
            "username":   "",
        }

    # ── 1. Hard rules — run BEFORE ML model ───────────────────────────────
    is_fake, conf, reason = hard_rules(username)
    if is_fake is not None:
        fv = build_features(username, profile)
        return {
            "prediction": "Fake",
            "confidence": conf,
            "method":     f"hard_rule:{reason}",
            "username":   username,
            "features":   {k: round(float(v), 3) for k, v in fv.items()},
        }

    # ── 2. ML model ───────────────────────────────────────────────────────
    if _using_ml:
        return predict_with_model(username, profile)

    # ── 3. Heuristic fallback ─────────────────────────────────────────────
    return predict_heuristic(username, profile)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            url     = sys.argv[1]
            profile = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        else:
            raw = sys.stdin.read().strip()
            if not raw:
                raise ValueError("No input received")
            data    = json.loads(raw)
            url     = data.get("url", "")
            profile = data.get("profile", {})

        result = predict(url, profile)
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({
            "prediction": "Real",
            "confidence": 0.55,
            "error":      str(e),
            "method":     "error_fallback",
        }))
