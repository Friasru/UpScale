"""Blockchains UpScale understands: address formats, provider network ids, and the quote
assets that make a pool's USD liquidity trustworthy.

Infrastructure depends on the chain, not on individual coins: adding a chain here (plus a
provider that serves it) extends coverage without touching the decision engine.
"""

import re
from typing import Literal

SOLANA = "solana"
EVM_CHAINS = frozenset({"ethereum", "base", "bsc", "arbitrum", "polygon", "avalanche", "optimism"})
# Chains whose DEX pools UpScale can read (DEX Screener uses these same chain ids).
DEX_CHAINS = frozenset({SOLANA, "ethereum", "base", "bsc", "arbitrum", "polygon"})
# GeckoTerminal network ids, for pool candles.
GECKOTERMINAL_NETWORKS: dict[str, str] = {
    SOLANA: "solana",
    "ethereum": "eth",
    "base": "base",
    "bsc": "bsc",
    "arbitrum": "arbitrum",
    "polygon": "polygon_pos",
}
CHAIN_LABELS: dict[str, str] = {
    SOLANA: "Solana",
    "ethereum": "Ethereum",
    "base": "Base",
    "bsc": "BNB Chain",
    "arbitrum": "Arbitrum",
    "polygon": "Polygon",
}
# Phrases in a request that name a chain (used to pick the chain of an EVM address).
CHAIN_MENTIONS: dict[str, str] = {
    "on solana": SOLANA,
    "on ethereum": "ethereum",
    "on eth": "ethereum",
    "erc20": "ethereum",
    "erc-20": "ethereum",
    "on base": "base",
    "base chain": "base",
    "on bsc": "bsc",
    "bnb chain": "bsc",
    "bep20": "bsc",
    "on arbitrum": "arbitrum",
    "on polygon": "polygon",
}

QuoteKind = Literal["SOL", "USDC", "USDT", "WETH", "WBNB", "other"]
# Quote assets per chain, by (normalized) token address.
KNOWN_QUOTES: dict[str, dict[str, QuoteKind]] = {
    SOLANA: {
        "So11111111111111111111111111111111111111112": "SOL",
        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "USDC",
        "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": "USDT",
    },
    "ethereum": {
        "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "WETH",
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "USDC",
        "0xdac17f958d2ee523a2206206994597c13d831ec7": "USDT",
    },
    "base": {
        "0x4200000000000000000000000000000000000006": "WETH",
        "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913": "USDC",
    },
    "bsc": {
        "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c": "WBNB",
        "0x55d398326f99059ff775485246999027b3197955": "USDT",
        "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d": "USDC",
    },
}

# Solana addresses are base58-encoded 32-byte keys: 32 to 44 characters, no 0/O/I/l.
_SOLANA_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")
_EVM_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


def is_solana_address(value: str) -> bool:
    return bool(_SOLANA_RE.fullmatch(value))


def is_evm_address(value: str) -> bool:
    return bool(_EVM_RE.fullmatch(value))


def is_valid_address(chain: str, value: str) -> bool:
    if chain == SOLANA:
        return is_solana_address(value)
    if chain in EVM_CHAINS:
        return is_evm_address(value)
    return False


def normalize_address(chain: str | None, address: str | None) -> str | None:
    """EVM addresses are case-insensitive hex (compared lowercased); Solana's base58 is
    case-sensitive and kept as is."""
    if address is None:
        return None
    address = address.strip()
    return address.lower() if chain in EVM_CHAINS else address


def same_address(chain: str, a: str | None, b: str | None) -> bool:
    return (
        a is not None
        and b is not None
        and normalize_address(chain, a) == normalize_address(chain, b)
    )


def quote_kind(chain: str, address: str) -> QuoteKind:
    return KNOWN_QUOTES.get(chain, {}).get(normalize_address(chain, address) or "", "other")


def chain_label(chain: str | None) -> str:
    return CHAIN_LABELS.get(chain or "", chain or "unknown chain")
