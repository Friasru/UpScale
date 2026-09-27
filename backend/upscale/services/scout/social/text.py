"""Deterministic text utilities: reference extraction, duplicate fingerprints, and opaque
author keys. No network, no models, no per-coin rules."""

import hashlib
import hmac
import re
from collections.abc import Iterable
from urllib.parse import urlparse

from upscale.services.chains import CHAIN_MENTIONS, EVM_CHAINS, SOLANA

_EVM_ADDRESS = re.compile(r"(?<![0-9A-Za-z])0x[0-9a-fA-F]{40}(?![0-9A-Za-z])")
_SOLANA_ADDRESS = re.compile(r"(?<![0-9A-Za-z])[1-9A-HJ-NP-Za-km-z]{32,44}(?![0-9A-Za-z])")
_CASHTAG = re.compile(r"(?<![\w$])\$([A-Za-z][A-Za-z0-9]{0,14})(?![A-Za-z0-9])")
_URL = re.compile(r"https?://\S+")
_WORDS = re.compile(r"[a-z0-9]+")
# Chain names that are unambiguous as standalone words (plus the phrases in CHAIN_MENTIONS).
_CHAIN_WORDS: dict[str, str] = {
    "solana": SOLANA,
    "ethereum": "ethereum",
    "arbitrum": "arbitrum",
    "polygon": "polygon",
    "avalanche": "avalanche",
    "optimism": "optimism",
    "bsc": "bsc",
    "bnb": "bsc",
    "erc20": "ethereum",
    "spl": SOLANA,
}


def extract_addresses(text: str) -> list[tuple[str, str]]:
    """(family, address) found in the text: family "evm" (lowercased) or "solana"."""
    found: list[tuple[str, str]] = []
    for match in _EVM_ADDRESS.findall(text):
        item = ("evm", match.lower())
        if item not in found:
            found.append(item)
    stripped = _EVM_ADDRESS.sub(" ", text)
    for match in _SOLANA_ADDRESS.findall(stripped):
        item = ("solana", match)
        if item not in found:
            found.append(item)
    return found


def extract_cashtags(text: str) -> set[str]:
    """Lowercased $TICKER references ("$100" is a price, not a ticker)."""
    return {m.lower() for m in _CASHTAG.findall(text)}


def is_cashtag_symbol(symbol: str) -> bool:
    """Whether `$symbol` would be recognized as a cashtag by `extract_cashtags`."""
    return _CASHTAG.fullmatch(f"${symbol}") is not None


def chain_phrases(chain: str) -> list[str]:
    """Every word / phrase `mentioned_chains` recognizes as naming `chain`."""
    phrases = {p for p, c in CHAIN_MENTIONS.items() if c == chain}
    phrases |= {w for w, c in _CHAIN_WORDS.items() if c == chain}
    return sorted(phrases)


def mentioned_chains(text: str) -> set[str]:
    lowered = text.lower()
    chains = {chain for phrase, chain in CHAIN_MENTIONS.items() if phrase in lowered}
    words = set(_WORDS.findall(lowered))
    chains |= {chain for word, chain in _CHAIN_WORDS.items() if word in words}
    return chains


def contains_phrase(text: str, phrase: str) -> bool:
    words = _WORDS.findall(text.lower())
    target = _WORDS.findall(phrase.lower())
    if not target:
        return False
    n = len(target)
    return any(words[i : i + n] == target for i in range(len(words) - n + 1))


def same_words(a: str, b: str) -> bool:
    """Whether two strings are the same words ("Degen" and "DEGEN", "Pepe Coin" and
    "pepe-coin"), ignoring case and punctuation."""
    return _WORDS.findall(a.lower()) == _WORDS.findall(b.lower())


def domains(urls: Iterable[str]) -> set[str]:
    out = set()
    for url in urls:
        host = (urlparse(url).hostname or "").lower()
        if host:
            out.add(host.removeprefix("www."))
    return out


def urls_in(text: str) -> list[str]:
    return _URL.findall(text)


def normalize_text(text: str) -> str:
    """Lowercase words only: links, addresses, numbers and punctuation removed, so the same
    message posted with another link or contract still fingerprints the same."""
    text = _URL.sub(" ", text)
    text = _EVM_ADDRESS.sub(" ", text)
    text = _SOLANA_ADDRESS.sub(" ", text)
    return " ".join(w for w in _WORDS.findall(text.lower()) if not w.isdigit())


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()[:16]


def simhash(text: str) -> str:
    """64-bit SimHash over word pairs: near-identical texts differ in few bits."""
    words = normalize_text(text).split()
    features = [" ".join(words[i : i + 2]) for i in range(len(words) - 1)] or words or [""]
    totals = [0] * 64
    for feature in features:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            totals[bit] += 1 if h >> bit & 1 else -1
    value = sum(1 << bit for bit in range(64) if totals[bit] > 0)
    return f"{value:016x}"


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def author_key(salt: bytes, platform: str, author_id: str) -> str:
    """Opaque, salted key: stable enough to count distinct authors, not usable for profiling
    (the salt never leaves the local store)."""
    digest = hmac.new(salt, f"{platform}:{author_id}".encode(), hashlib.sha256).hexdigest()
    return digest[:20]


def address_family(chain: str) -> str | None:
    if chain == SOLANA:
        return "solana"
    if chain in EVM_CHAINS:
        return "evm"
    return None
