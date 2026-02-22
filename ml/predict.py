"""
predict.py — Fake Profile Predictor
=====================================
Can be called two ways (server.js uses args):
  1. python predict.py "https://instagram.com/username"   ← used by server
  2. echo '{"url":"..."}' | python predict.py             ← stdin fallback

Outputs JSON to stdout:
  {"prediction":"Fake"|"Real", "confidence":0.xx, "method":"...", ...}

Priority:
  1. Use trained model.pkl + scaler.pkl if both classes present
  2. Fall back to calibrated heuristic engine (always available)
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
                    _feature_names = json.load(f).get("features", [])
            _using_ml = True
        else:
            _model = _scaler = None   # broken / single-class model
except Exception:
    pass


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
SUSPICIOUS_KW = {
    "bot","fake","scam","spam","hack","temp","xyz","anon",
    "xxx","adult","promo","giveaway","f4f","l4l","onlyfan",
    "follow4follow","like4like","viral","offi","official2",
}
VOWELS = set("aeiouAEIOU")


def _entropy(s):
    if not s:
        return 0.0
    freq = {}
    for c in s.lower():
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def extract_username(url: str) -> str:
    """Pull last meaningful path segment from a URL."""
    url = url.strip().rstrip("/")
    url = re.split(r"[?#]", url)[0]
    parts = [p for p in url.split("/")
             if p and not p.startswith("http") and "." not in p]
    return parts[-1] if parts else url.split("/")[-1]


def build_features(username: str, profile: dict = None) -> dict:
    """Return 22-feature dict. All values are numeric."""
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
        "very_short":               int(n <= 3),           # ← catches "1","12","ab"
        "only_one_char":            int(n == 1),           # ← strongest signal
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
#  HARD-RULE PRE-CHECK  — catches obvious cases before any scoring
# ══════════════════════════════════════════════════════════════════════════════
def hard_rules(username: str) -> tuple:
    """
    Returns (is_fake: bool, confidence: float, reason: str) or (None,…) if
    no hard rule fires.
    """
    u = username.strip()
    n = len(u)

    # Single character usernames are never real social media profiles
    if n <= 1:
        return True, 0.97, "single_char_username"

    # Pure digits of any length
    if u.isdigit():
        return True, 0.95, "pure_digit_username"

    # 2-char all-digits  e.g. "12"
    if n == 2 and sum(c.isdigit() for c in u) == 2:
        return True, 0.92, "two_digit_username"

    # Username is just special chars
    if all(c in "._-" for c in u):
        return True, 0.96, "only_special_chars"

    return None, 0.0, ""


# ══════════════════════════════════════════════════════════════════════════════
#  ML MODEL PATH
# ══════════════════════════════════════════════════════════════════════════════
def predict_with_model(username: str, profile: dict = None) -> dict:
    fv  = build_features(username, profile)
    vec = [fv.get(name, 0.0) for name in _feature_names] if _feature_names else list(fv.values())

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
# ══════════════════════════════════════════════════════════════════════════════
WEIGHTS = {
    "only_one_char":        0.40,   # strongest — single char = almost always fake
    "very_short":           0.25,   # 2-3 chars
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
MAX_W       = sum(WEIGHTS.values())   # ≈ 2.45
FAKE_THRESH = 0.12                    # normalised score above this → Fake


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
    username = extract_username(url)

    if not username:
        return {"prediction": "Real", "confidence": 0.55,
                "method": "default", "username": ""}

    # ── Hard rules first (catches single-char, pure-digit, etc.) ──────────
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

    # ── ML model (if trained & valid) ──────────────────────────────────────
    if _using_ml:
        return predict_with_model(username, profile)

    # ── Heuristic fallback ─────────────────────────────────────────────────
    return predict_heuristic(username, profile)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT  — accepts URL as CLI arg OR JSON on stdin
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        # Primary: URL passed as first command-line argument (used by server.js)
        if len(sys.argv) > 1:
            url     = sys.argv[1]
            profile = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        else:
            # Fallback: read JSON from stdin
            raw     = sys.stdin.read().strip()
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