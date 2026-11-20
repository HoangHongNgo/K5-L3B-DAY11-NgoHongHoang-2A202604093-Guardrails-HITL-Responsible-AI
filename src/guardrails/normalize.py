"""
Shared de-obfuscation helpers for Blue's input + output guardrails.

Attackers hide payloads so a single regex over the raw text misses them.
Instead of matching one string, we build several *views* of the text — each
one undoing a common obfuscation trick — and run the policy over every view:

  - Unicode tricks:  NFKC (full-width), invisible / format chars, homoglyphs
                     (Cyrillic/Greek look-alikes), Vietnamese accents
  - Leetspeak:       1gn0r3 → ignore, p@ssw0rd → password
  - Spaced letters:  "i g n o r e", "a-p-i k-e-y"  → squashed view
  - Encodings:       base64 / hex / ROT13 / reversed text
"""
from __future__ import annotations

import base64
import binascii
import codecs
import re
import unicodedata

# Cyrillic / Greek characters that render like Latin letters.
_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ј": "j", "ѕ": "s", "к": "k", "м": "m", "т": "t", "н": "h",
    "в": "b", "ԁ": "d", "ɡ": "g", "ⅼ": "l",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S",
    "α": "a", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ο": "o", "ρ": "p",
    "τ": "t", "υ": "u", "χ": "x", "Α": "A", "Β": "B", "Ε": "E", "Ι": "I",
    "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X",
})

_LEET = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
    "@": "a", "$": "s", "!": "i", "|": "l",
})

_B64_TOKEN = re.compile(r"[A-Za-z0-9+/_-]{8,}={0,2}")
_HEX_TOKEN = re.compile(r"(?:[0-9a-fA-F]{2}[\s:-]?){6,}")
# "i g n o r e", "a-p-i", "p.a.s.s" → runs of single characters joined by separators
_SPACED_RUN = re.compile(r"\b(?:\w[\s\-_.*/|]{1,3}){3,}\w\b")


def strip_invisible(text: str) -> str:
    """Remove zero-width / bidi / soft-hyphen and other format (Cf) characters."""
    return "".join(c for c in text if unicodedata.category(c) != "Cf")


def normalize(text: str) -> str:
    """NFKC + strip invisible + homoglyph fold + collapse whitespace."""
    text = unicodedata.normalize("NFKC", text or "")
    text = strip_invisible(text).translate(_HOMOGLYPHS)
    return re.sub(r"\s+", " ", text).strip()


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def squash(text: str) -> str:
    """Lowercase alnum only — defeats spacing/punctuation tricks."""
    return re.sub(r"[^a-z0-9]", "", (text or "").casefold())


def _printable(s: str) -> bool:
    if len(s) < 4:
        return False
    ok = sum(1 for c in s if c.isprintable() and ord(c) < 0x2000)
    return ok / len(s) > 0.85


def decoded_payloads(text: str) -> list[str]:
    """Decode base64 / hex blobs found inside the text."""
    out: list[str] = []
    for token in _B64_TOKEN.findall(text):
        padded = token.replace("-", "+").replace("_", "/")
        padded += "=" * (-len(padded) % 4)
        try:
            decoded = base64.b64decode(padded, validate=False).decode("utf-8", errors="ignore")
        except (binascii.Error, ValueError):
            continue
        if _printable(decoded):
            out.append(decoded)
    for token in _HEX_TOKEN.findall(text):
        digits = re.sub(r"[^0-9a-fA-F]", "", token)
        digits = digits[: len(digits) - (len(digits) % 2)]
        try:
            decoded = bytes.fromhex(digits).decode("utf-8", errors="ignore")
        except ValueError:
            continue
        if _printable(decoded):
            out.append(decoded)
    return out


def _collapse_spaced(text: str) -> str:
    return _SPACED_RUN.sub(lambda m: re.sub(r"[\s\-_.*/|]", "", m.group(0)), text)


def text_views(text: str, *, include_reversed: bool = False) -> list[str]:
    """All de-obfuscated views of ``text`` (deduplicated, order preserved)."""
    base = normalize(text)
    folded = strip_accents(base)
    views = [base, folded, folded.translate(_LEET), _collapse_spaced(folded),
             codecs.decode(folded, "rot_13")]
    for decoded in decoded_payloads(base):
        d = strip_accents(normalize(decoded))
        views.extend([d, d.translate(_LEET)])
    if include_reversed:
        views.extend([v[::-1] for v in list(views)])
    seen, unique = set(), []
    for v in views:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique
