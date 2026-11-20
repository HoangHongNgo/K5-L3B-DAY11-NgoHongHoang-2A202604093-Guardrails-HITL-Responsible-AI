"""
Microsoft Presidio PII layer for Blue's output (open-source, MIT).

Presidio adds checks the hand-written regexes cannot do well:
  * CREDIT_CARD with Luhn checksum (no false hits on random 16-digit numbers)
  * IBAN with checksum, international PHONE_NUMBER via Google's ``phonenumbers``
    (VN, US, GB, JP, KR, CN, FR, DE, SG, TH, …), EMAIL, IP, crypto wallets
  * custom recognizers: VN phone, VN CCCD/CMND, VinBank lab secrets, generic
    API keys / bearer tokens / private keys

Only pattern-based recognizers are enabled (no PERSON/LOCATION NER), because a
bank reply legitimately mentions branch names and people; this keeps false
positives low and works on text in any language.
"""
from __future__ import annotations

import os
from functools import lru_cache

# Official numbers the assistant is allowed to show (not PII).
ALLOW_LIST = ["1900 545 413", "1900545413", "support@vinbank.example"]

ENTITIES = [
    "CREDIT_CARD", "IBAN_CODE", "PHONE_NUMBER", "EMAIL_ADDRESS", "IP_ADDRESS",
    "CRYPTO", "VN_PHONE", "VN_NATIONAL_ID", "VINBANK_SECRET", "API_SECRET",
]
SCORE_THRESHOLD = float(os.environ.get("BLUE_PII_THRESHOLD", "0.4"))  # PhoneRecognizer base score is 0.4
PHONE_REGIONS = ("VN", "US", "GB", "JP", "KR", "CN", "FR", "DE", "SG", "TH",
                 "IN", "AU", "ID", "MY", "PH", "RU", "ES", "IT", "BR")


@lru_cache(maxsize=1)
def get_analyzer():
    """Build the Presidio AnalyzerEngine once; None if Presidio is not installed."""
    try:
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_analyzer.predefined_recognizers import (
            CreditCardRecognizer, CryptoRecognizer, EmailRecognizer,
            IbanRecognizer, IpRecognizer, PhoneRecognizer,
        )
    except ImportError:
        return None

    try:
        nlp_engine = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }).create_engine()
    except Exception:
        return None

    registry = RecognizerRegistry(supported_languages=["en"])
    for rec in (CreditCardRecognizer(), CryptoRecognizer(), EmailRecognizer(),
                IbanRecognizer(), IpRecognizer(),
                PhoneRecognizer(supported_regions=PHONE_REGIONS)):
        registry.add_recognizer(rec)

    custom = [
        ("VN_PHONE", [Pattern("vn_mobile", r"(?<!\d)(?:\+?84|0)[\s.-]?(?:3|5|7|8|9)\d[\s.-]?\d{3}[\s.-]?\d{3,4}(?!\d)", 0.7)]),
        ("VN_NATIONAL_ID", [Pattern("cccd", r"(?<!\d)0\d{11}(?!\d)", 0.6),
                            Pattern("cmnd", r"(?<!\d)\d{9}(?!\d)", 0.3)]),
        ("VINBANK_SECRET", [Pattern("admin_pw", r"(?i)\badmin\W{0,3}123\b", 1.0),
                            Pattern("sk_key", r"(?i)\bsk-[a-z0-9_-]{6,}", 1.0),
                            Pattern("db_host", r"(?i)\b[\w-]+(?:\.[\w-]+)*\.internal(?::\d+)?\b", 0.9)]),
        ("API_SECRET", [Pattern("aws", r"\bAKIA[0-9A-Z]{16}\b", 0.9),
                        Pattern("github", r"\bgh[pousr]_[A-Za-z0-9]{30,}\b", 0.9),
                        Pattern("bearer", r"(?i)\bbearer\s+[a-z0-9._~+/-]{20,}=*", 0.8),
                        Pattern("jwt", r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b", 0.9),
                        Pattern("pem", r"-----BEGIN [A-Z ]*PRIVATE KEY-----", 1.0)]),
    ]
    for entity, patterns in custom:
        registry.add_recognizer(PatternRecognizer(supported_entity=entity, patterns=patterns))

    return AnalyzerEngine(registry=registry, nlp_engine=nlp_engine, supported_languages=["en"])


def presidio_available() -> bool:
    return get_analyzer() is not None


def analyze_pii(text: str) -> list[dict]:
    """Return a list of ``{entity, start, end, score}`` findings (empty if none / unavailable)."""
    analyzer = get_analyzer()
    if analyzer is None or not text:
        return []
    results = analyzer.analyze(
        text=text, language="en", entities=ENTITIES,
        score_threshold=SCORE_THRESHOLD, allow_list=ALLOW_LIST,
    )
    return [
        {"entity": r.entity_type, "start": r.start, "end": r.end, "score": round(r.score, 2)}
        for r in results
    ]


def redact_pii(text: str, replacement: str = "[REDACTED]") -> tuple[str, list[str]]:
    """Replace every Presidio finding with ``replacement``; returns (text, entity types)."""
    findings = analyze_pii(text)
    if not findings:
        return text, []
    # merge overlapping spans, replace right-to-left
    spans = sorted((f["start"], f["end"]) for f in findings)
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out = text
    for s, e in reversed(merged):
        out = out[:s] + replacement + out[e:]
    return out, sorted({f["entity"] for f in findings})
