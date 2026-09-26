"""Discovery providers: where Scout finds tokens, normalized into ScoutCandidates.

`DiscoveryProvider` is the interface; Scout's service only sees normalized candidates, so
providers can be added or replaced without touching anything downstream (Risk and
Opportunity never see providers at all). Each provider declares which discovery kinds and
chains it serves; a kind it doesn't serve is never faked.

Providers today (both free public APIs, no key):

* **GeckoTerminal** (`api.geckoterminal.com/api/v2`): new pools
  (`/networks/{net}/new_pools`), most-traded pools (`/networks/{net}/pools?sort=
  h24_tx_count_desc`), trending pools (`/networks/{net}/trending_pools`), and exact tokens
  in batches of 30 (`/networks/{net}/tokens/multi/{addresses}?include=top_pools`). It
  reports 5m / 15m / 30m / 1h / 6h / 24h windows, distinct buyers and sellers, FDV, and a
  market cap only when one is known (often null for new tokens: it stays null here).
* **DEX Screener** (`api.dexscreener.com`): newly created token profiles
  (`/token-profiles/latest/v1`) and exact tokens in batches of 30
  (`/tokens/v1/{chain}/{addresses}`). Its "boosts" listings are paid promotion and are
  deliberately not used as a discovery source.

Every HTTP request goes through the provider's `RequestGate` (cache, deduplication, rate
limit, concurrency cap, timeout).
"""

import asyncio
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

import httpx2

from upscale.services.chains import (
    DEX_CHAINS,
    GECKOTERMINAL_NETWORKS,
    chain_label,
    is_valid_address,
    normalize_address,
)
from upscale.services.dexscreener import (
    _count,
    _dict,
    _non_negative,
    _number,
    _positive,
    parse_pair,
)
from upscale.services.market_data import MarketDataError, MarketDataUnavailableError
from upscale.services.scout.config import ScoutConfig
from upscale.services.scout.gate import RequestGate
from upscale.services.scout.models import SCOUT_WINDOWS, DiscoveryKind
from upscale.services.scout.normalize import (
    DiscoveryResult,
    Listing,
    build_candidates,
    canonical_id,
)
from upscale.services.solana_dex import DexPool, TokenRef, WindowStats


class DiscoveryNotSupportedError(MarketDataError):
    """The provider doesn't offer this kind of discovery (or this chain)."""


class DiscoveryProvider(Protocol):
    name: str
    kinds: frozenset[DiscoveryKind]
    chains: frozenset[str]

    async def discover_new_tokens(self, chain: str, limit: int) -> DiscoveryResult:
        """Tokens with recently created pools."""
        ...

    async def discover_active_tokens(self, chain: str, limit: int) -> DiscoveryResult:
        """Tokens whose pools are trading the most right now."""
        ...

    async def discover_trending_tokens(self, chain: str, limit: int) -> DiscoveryResult:
        """Tokens the provider lists as trending (its own, activity-based, definition)."""
        ...

    async def lookup_exact_tokens(self, chain: str, addresses: Sequence[str]) -> DiscoveryResult:
        """Every pool the provider knows for exact token addresses (batched)."""
        ...

    async def lookup_exact_token(self, chain: str, address: str) -> DiscoveryResult:
        """One exact token (chain + contract / mint)."""
        ...


class _HttpDiscoveryProvider:
    name = "provider"
    kinds: frozenset[DiscoveryKind] = frozenset()
    chains: frozenset[str] = frozenset()

    def __init__(
        self,
        base_url: str,
        gate: RequestGate,
        config: ScoutConfig,
        transport: httpx2.AsyncBaseTransport | None,
        now: Callable[[], datetime],
        headers: dict[str, str],
    ):
        self.base_url = base_url
        self.gate = gate
        self.config = config
        self._transport = transport
        self.now = now
        self._headers = headers

    def _require(self, kind: DiscoveryKind, chain: str) -> None:
        if kind not in self.kinds:
            raise DiscoveryNotSupportedError(f"{self.name} doesn't offer {kind} discovery")
        if chain not in self.chains:
            raise DiscoveryNotSupportedError(f"{self.name} doesn't cover {chain_label(chain)}")

    async def lookup_exact_tokens(self, chain: str, addresses: Sequence[str]) -> DiscoveryResult:
        raise NotImplementedError

    async def lookup_exact_token(self, chain: str, address: str) -> DiscoveryResult:
        return await self.lookup_exact_tokens(chain, [address])

    async def _get(self, path: str, params: dict[str, str] | None = None) -> Any:
        key = path + "?" + "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
        return await self.gate.run(key, lambda: self._request(path, params))

    async def _request(self, path: str, params: dict[str, str] | None) -> Any:
        try:
            async with httpx2.AsyncClient(
                base_url=self.base_url,
                headers=self._headers,
                timeout=self.gate.limits.timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.get(path, params=params)
        except httpx2.TimeoutException as exc:
            raise MarketDataUnavailableError(f"{self.name} request timed out") from exc
        except httpx2.HTTPError as exc:
            raise MarketDataUnavailableError(f"could not reach {self.name}") from exc
        if response.status_code == 429:
            raise MarketDataUnavailableError(f"{self.name} rate limit reached")
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise MarketDataUnavailableError(f"{self.name} returned HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError as exc:
            raise MarketDataUnavailableError(f"{self.name} returned invalid JSON") from exc

    async def _batched(
        self, chain: str, addresses: Sequence[str], fetch_batch: Any
    ) -> list[DiscoveryResult]:
        unique = list(dict.fromkeys(normalize_address(chain, a) or a for a in addresses))
        valid = [a for a in unique if is_valid_address(chain, a)]
        size = self.config.lookup_batch_size
        batches = [valid[i : i + size] for i in range(0, len(valid), size)]
        results: list[DiscoveryResult] = await asyncio.gather(*(fetch_batch(b) for b in batches))
        return results


def _json_list(body: Any, provider: str) -> list[Any]:
    """A JSON list body; None (HTTP 404) is an empty list."""
    if body is None:
        return []
    if not isinstance(body, list):
        raise MarketDataUnavailableError(f"{provider} returned an unexpected response")
    return body


def _combine(results: Sequence[DiscoveryResult]) -> DiscoveryResult:
    out = DiscoveryResult()
    for r in results:
        out.candidates.extend(r.candidates)
        out.rejected.extend(r.rejected)
    return out


# --- GeckoTerminal --------------------------------------------------------------------------

GT_BASE_URL = "https://api.geckoterminal.com/api/v2"
GT_INCLUDE = "base_token,quote_token,dex"


class GeckoTerminalDiscoveryProvider(_HttpDiscoveryProvider):
    name = "GeckoTerminal"
    kinds: frozenset[DiscoveryKind] = frozenset({"new", "active", "trending", "lookup"})
    chains = frozenset(GECKOTERMINAL_NETWORKS)

    def __init__(
        self,
        config: ScoutConfig | None = None,
        gate: RequestGate | None = None,
        base_url: str = GT_BASE_URL,
        transport: httpx2.AsyncBaseTransport | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        trending_duration: str = "1h",
    ):
        config = config or ScoutConfig()
        super().__init__(
            base_url,
            gate or RequestGate(self.name, config.geckoterminal),
            config,
            transport,
            now,
            {"accept": "application/json;version=20230302"},
        )
        self.trending_duration = trending_duration

    async def discover_new_tokens(self, chain: str, limit: int) -> DiscoveryResult:
        self._require("new", chain)
        return await self._pool_listing(chain, "new", "new_pools", {}, limit)

    async def discover_active_tokens(self, chain: str, limit: int) -> DiscoveryResult:
        self._require("active", chain)
        return await self._pool_listing(
            chain, "active", "pools", {"sort": "h24_tx_count_desc"}, limit
        )

    async def discover_trending_tokens(self, chain: str, limit: int) -> DiscoveryResult:
        self._require("trending", chain)
        return await self._pool_listing(
            chain, "trending", "trending_pools", {"duration": self.trending_duration}, limit
        )

    async def lookup_exact_tokens(self, chain: str, addresses: Sequence[str]) -> DiscoveryResult:
        self._require("lookup", chain)
        network = GECKOTERMINAL_NETWORKS[chain]

        async def batch(addrs: list[str]) -> DiscoveryResult:
            fetched_at = self.now()
            body = await self._get(
                f"/networks/{network}/tokens/multi/{','.join(addrs)}", {"include": "top_pools"}
            )
            pools, malformed = parse_gt_document(body, chain, network)
            listing = Listing(self.name, "lookup", "tokens/multi", fetched_at)
            return build_candidates(
                pools,
                listing,
                self.config,
                malformed_rows=malformed,
                restrict_to={canonical_id(chain, a) for a in addrs},
            )

        return _combine(await self._batched(chain, addresses, batch))

    async def _pool_listing(
        self, chain: str, kind: DiscoveryKind, endpoint: str, params: dict[str, str], limit: int
    ) -> DiscoveryResult:
        network = GECKOTERMINAL_NETWORKS[chain]
        fetched_at = self.now()
        body = await self._get(
            f"/networks/{network}/{endpoint}", {"include": GT_INCLUDE, "page": "1", **params}
        )
        pools, malformed = parse_gt_document(body, chain, network)
        pools = pools[:limit]
        positions: dict[str, int] = {}
        for i, pool in enumerate(pools):
            positions.setdefault(canonical_id(chain, pool.base.address), i)
        name = endpoint + ("?" + "&".join(f"{k}={v}" for k, v in params.items()) if params else "")
        listing = Listing(self.name, kind, name, fetched_at)
        return build_candidates(pools, listing, self.config, positions, malformed)


def parse_gt_document(body: Any, chain: str, network: str) -> tuple[list[DexPool], int]:
    """Pools in a GeckoTerminal JSON:API document (from `data` and `included`), plus how
    many pool rows were malformed. Raises if the document itself is malformed."""
    if body is None:
        return [], 0  # 404: nothing known
    if not isinstance(body, dict) or not isinstance(body.get("data"), list | dict):
        raise MarketDataUnavailableError("GeckoTerminal returned an unexpected response")
    data: list[Any] = body["data"] if isinstance(body["data"], list) else [body["data"]]
    raw_included = body.get("included")
    included: list[Any] = raw_included if isinstance(raw_included, list) else []
    rows = [r for r in [*data, *included] if isinstance(r, dict)]
    malformed = len(data) + len(included) - len(rows)  # non-object entries
    tokens: dict[str, TokenRef] = {}
    for row in rows:
        if row.get("type") == "token" and isinstance(row.get("id"), str):
            attrs = _dict(row.get("attributes"))
            address = attrs.get("address")
            if isinstance(address, str) and address:
                tokens[row["id"]] = TokenRef(
                    address=address,
                    symbol=_text(attrs.get("symbol")),
                    name=_text(attrs.get("name")),
                )
    pools: list[DexPool] = []
    for row in rows:
        if row.get("type") != "pool":
            continue
        pool = _gt_pool(row, tokens, chain, network)
        if pool is None:
            malformed += 1
        else:
            pools.append(pool)
    return pools, malformed


def _gt_pool(
    row: dict[str, Any], tokens: dict[str, TokenRef], chain: str, network: str
) -> DexPool | None:
    attrs, rel = _dict(row.get("attributes")), _dict(row.get("relationships"))
    address = attrs.get("address")
    base = _gt_token(rel.get("base_token"), tokens, network)
    quote = _gt_token(rel.get("quote_token"), tokens, network)
    dex = _dict(_dict(rel.get("dex")).get("data")).get("id")
    if not (isinstance(address, str) and address and isinstance(dex, str) and dex):
        return None
    if base is None or quote is None:
        return None
    txns, volume = _dict(attrs.get("transactions")), _dict(attrs.get("volume_usd"))
    change = _dict(attrs.get("price_change_percentage"))
    windows = []
    for w in SCOUT_WINDOWS:
        counts = _dict(txns.get(w))
        stats = WindowStats(
            window=w,
            buys=_count(counts.get("buys")),
            sells=_count(counts.get("sells")),
            buyers=_count(counts.get("buyers")),
            sellers=_count(counts.get("sellers")),
            volume_usd=_non_negative(_number(volume.get(w))),
            price_change_pct=_number(change.get(w)),
        )
        if any(
            v is not None
            for v in (stats.buys, stats.sells, stats.volume_usd, stats.price_change_pct)
        ):
            windows.append(stats)
    return DexPool(
        chain=chain,
        dex=dex,
        pair_address=address,
        url=f"https://www.geckoterminal.com/{network}/pools/{address}",
        base=base,
        quote=quote,
        price_usd=_positive(_number(attrs.get("base_token_price_usd"))),
        price_native=_positive(_number(attrs.get("base_token_price_quote_token"))),
        liquidity_usd=_non_negative(_number(attrs.get("reserve_in_usd"))),
        market_cap_usd=_positive(_number(attrs.get("market_cap_usd"))),
        fdv_usd=_positive(_number(attrs.get("fdv_usd"))),
        pair_created_at=_iso(attrs.get("pool_created_at")),
        windows=windows,
    )


def _gt_token(value: Any, tokens: dict[str, TokenRef], network: str) -> TokenRef | None:
    token_id = _dict(_dict(value).get("data")).get("id")
    if not isinstance(token_id, str) or not token_id.startswith(network + "_"):
        return None
    if token_id in tokens:
        return tokens[token_id]
    address = token_id[len(network) + 1 :]
    return TokenRef(address=address) if address else None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _iso(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


# --- DEX Screener ---------------------------------------------------------------------------

DS_BASE_URL = "https://api.dexscreener.com"


class DexScreenerDiscoveryProvider(_HttpDiscoveryProvider):
    name = "DEX Screener"
    kinds: frozenset[DiscoveryKind] = frozenset({"new", "lookup"})
    chains = frozenset(DEX_CHAINS)

    def __init__(
        self,
        config: ScoutConfig | None = None,
        gate: RequestGate | None = None,
        base_url: str = DS_BASE_URL,
        transport: httpx2.AsyncBaseTransport | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        config = config or ScoutConfig()
        super().__init__(
            base_url,
            gate or RequestGate(self.name, config.dexscreener),
            config,
            transport,
            now,
            {"accept": "application/json"},
        )

    async def discover_new_tokens(self, chain: str, limit: int) -> DiscoveryResult:
        """Newest token profiles on `chain`, then their pools in batched exact lookups."""
        self._require("new", chain)
        body = await self._get("/token-profiles/latest/v1")  # all chains: one cached request
        rows = _json_list(body, self.name)
        addresses: list[str] = []
        malformed = 0
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("tokenAddress"), str):
                malformed += 1
                continue
            if row.get("chainId") == chain:
                address = normalize_address(chain, row["tokenAddress"]) or row["tokenAddress"]
                if address not in addresses:
                    addresses.append(address)
        addresses = addresses[:limit]
        fetched_at = self.now()
        positions = {canonical_id(chain, a): i for i, a in enumerate(addresses)}
        listing = Listing(
            self.name,
            "new",
            "token-profiles/latest",
            fetched_at,
            note="newly created DEX Screener token profile (profiles are paid; not a quality signal)",
        )
        result = await self._lookup(chain, addresses, listing, positions)
        if malformed:
            result.rejected.extend(
                build_candidates([], listing, self.config, malformed_rows=malformed).rejected
            )
        return result

    async def discover_active_tokens(self, chain: str, limit: int) -> DiscoveryResult:
        self._require("active", chain)
        raise AssertionError("unreachable")  # pragma: no cover

    async def discover_trending_tokens(self, chain: str, limit: int) -> DiscoveryResult:
        self._require("trending", chain)  # boosts are paid promotion: never used
        raise AssertionError("unreachable")  # pragma: no cover

    async def lookup_exact_tokens(self, chain: str, addresses: Sequence[str]) -> DiscoveryResult:
        self._require("lookup", chain)
        listing = Listing(self.name, "lookup", "tokens/v1", self.now())
        return await self._lookup(chain, addresses, listing, None)

    async def _lookup(
        self,
        chain: str,
        addresses: Sequence[str],
        listing: Listing,
        positions: dict[str, int] | None,
    ) -> DiscoveryResult:
        async def batch(addrs: list[str]) -> DiscoveryResult:
            body = await self._get(f"/tokens/v1/{chain}/{','.join(addrs)}")
            rows = _json_list(body, self.name)
            pools = [p for row in rows if (p := parse_pair(row)) is not None]
            return build_candidates(
                pools,
                listing,
                self.config,
                positions,
                malformed_rows=len(rows) - len(pools),
                restrict_to={canonical_id(chain, a) for a in addrs},
            )

        return _combine(await self._batched(chain, addresses, batch))
