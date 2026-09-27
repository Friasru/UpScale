"""Social trend providers: public-post search behind one interface.

Scout never depends on a particular network. A provider searches public posts for a batch
of terms (token addresses and $tickers) and returns normalized `SocialPost`s; attribution,
windows and momentum are provider-independent. Providers only use official APIs with
the caller's own credentials, never bypass access controls or rate limits, and are
"not configured" (never searched) without the access they need.

Implemented:

* **Reddit** (official Data API, OAuth application-only token; requires Reddit's approval
  of the API application under its Responsible Builder Policy; 100 queries / minute).
* **Farcaster** via **Neynar** (cast search, `x-api-key`; credit-based plans).
* **X** (API v2 recent search, bearer token; pay-per-use, so it also needs an explicit
  opt-in and is capped per run).
* **Discourse** forums (the public `search.json` any public Discourse forum serves
  anonymously; each configured forum is a separate source).
* **StaticSocialProvider**: offline, fixture-backed (tests and local development).

Every request goes through a `RequestGate` (cache, deduplication, rate limit, concurrency
cap, timeout). Terms are batched into as few queries as each API's syntax allows.
"""

import asyncio
import base64
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx2

from upscale.services.market_data import MarketDataUnavailableError
from upscale.services.scout.gate import RateLimitReachedError, RequestGate
from upscale.services.scout.social.config import SocialProviderConfig
from upscale.services.scout.social.models import SocialPost, SocialSearchResult
from upscale.services.scout.social.text import urls_in

AuthorKeyer = Callable[[str, str], str]  # (platform, raw author id) -> opaque key


class SocialProviderError(MarketDataUnavailableError):
    pass


class SocialTrendProvider(Protocol):
    name: str
    platform: str
    requirement: str  # the access this provider needs, reported when not configured
    max_terms_per_query: int
    max_query_chars: int

    @property
    def configured(self) -> bool: ...

    def render_query(self, terms: Sequence[str]) -> str:
        """One search query for a batch of terms, in the API's syntax."""
        ...

    async def search(
        self, terms: Sequence[str], since: datetime, keyer: AuthorKeyer
    ) -> SocialSearchResult:
        """Public posts matching any of `terms`, created since `since` (newest first)."""
        ...


def plan_queries(
    provider: SocialTrendProvider, terms_by_token: dict[str, list[str]]
) -> list[tuple[list[str], list[str]]]:
    """Group tokens into batched queries: [(token ids, terms)]. A token's terms always stay
    in one query, so every token is covered by exactly one search."""
    plans: list[tuple[list[str], list[str]]] = []
    ids: list[str] = []
    terms: list[str] = []
    for cid, token_terms in terms_by_token.items():
        candidate = [*terms, *token_terms]
        fits = (
            len(candidate) <= provider.max_terms_per_query
            and len(provider.render_query(candidate)) <= provider.max_query_chars
        )
        if ids and not fits:
            plans.append((ids, terms))
            ids, terms = [], []
        if not ids and (
            len(token_terms) > provider.max_terms_per_query
            or len(provider.render_query(token_terms)) > provider.max_query_chars
        ):
            # one token alone is too long: search its terms one by one
            for term in token_terms:
                plans.append(([cid], [term]))
            continue
        ids.append(cid)
        terms = [*terms, *token_terms]
    if ids:
        plans.append((ids, terms))
    return plans


def _ticker_word(term: str) -> str:
    return term[1:] if term.startswith("$") else term


# --- Offline --------------------------------------------------------------------------------


class StaticSocialProvider:
    """Serves fixed posts (fixtures). `fail` makes every search raise, `configured=False`
    behaves like a provider without credentials."""

    requirement = "none (offline fixtures)"

    def __init__(
        self,
        name: str,
        platform: str,
        posts: Sequence[dict[str, Any]] = (),
        *,
        configured: bool = True,
        fail: Exception | None = None,
        max_terms_per_query: int = 20,
        max_query_chars: int = 1000,
        gate: RequestGate | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        max_results: int | None = None,
    ):
        self.name = name
        self.platform = platform
        self.posts = list(posts)
        self._configured = configured
        self.fail = fail
        self.max_terms_per_query = max_terms_per_query
        self.max_query_chars = max_query_chars
        self.gate = gate
        self.now = now
        self.max_results = max_results
        self.queries: list[str] = []

    @property
    def configured(self) -> bool:
        return self._configured

    def render_query(self, terms: Sequence[str]) -> str:
        return " OR ".join(terms)

    async def search(
        self, terms: Sequence[str], since: datetime, keyer: AuthorKeyer
    ) -> SocialSearchResult:
        query = self.render_query(terms)

        async def run() -> SocialSearchResult:
            self.queries.append(query)
            if self.fail is not None:
                raise self.fail
            checked = self.now()
            wanted = [t.lower() for t in terms] + [_ticker_word(t).lower() for t in terms]
            rows = sorted(
                (
                    p
                    for p in self.posts
                    if p["created_at"] > since
                    and p["created_at"] <= checked
                    and any(w in _post_text(p).lower() for w in wanted)
                ),
                key=lambda p: p["created_at"],
                reverse=True,
            )
            complete = since
            if self.max_results is not None and len(rows) > self.max_results:
                rows = rows[: self.max_results]
                complete = rows[-1]["created_at"]
            return SocialSearchResult(
                provider=self.name,
                platform=self.platform,
                checked_at=checked,
                posts=[_static_post(self, p, keyer) for p in rows],
                complete_since=complete,
            )

        if self.gate is None:
            return await run()
        key = f"{query}|{since.replace(second=0, microsecond=0).isoformat()}"
        return await self.gate.run(key, run)


def _post_text(p: dict[str, Any]) -> str:
    return " ".join([p.get("text", ""), *p.get("urls", [])])


def _static_post(
    provider: StaticSocialProvider, p: dict[str, Any], keyer: AuthorKeyer
) -> SocialPost:
    return SocialPost(
        provider=provider.name,
        platform=provider.platform,
        post_id=str(p["id"]),
        created_at=p["created_at"],
        text=p.get("text", ""),
        author_key=keyer(provider.platform, str(p["author"])),
        author_handle=p.get("handle"),
        urls=list(p.get("urls", [])) + urls_in(p.get("text", "")),
        likes=p.get("likes"),
        replies=p.get("replies"),
        reposts=p.get("reposts"),
        quotes=p.get("quotes"),
        promoted=p.get("promoted"),
    )


# --- HTTP base ------------------------------------------------------------------------------


class _HttpSocialProvider:
    name = "social"
    platform = "social"
    requirement = ""

    def __init__(
        self,
        settings: SocialProviderConfig,
        gate: RequestGate | None,
        transport: httpx2.AsyncBaseTransport | None,
        now: Callable[[], datetime],
    ):
        self.settings = settings
        self.gate = gate or RequestGate(self.name, settings.limits)
        self._transport = transport
        self.now = now
        self.max_terms_per_query = settings.max_terms_per_query
        self.max_query_chars = settings.max_query_chars

    async def _get(
        self, url: str, params: dict[str, str], headers: dict[str, str], cache_key: str
    ) -> Any:
        return await self.gate.run(cache_key, lambda: self._request("GET", url, params, headers))

    async def _request(
        self,
        method: str,
        url: str,
        params: dict[str, str] | None,
        headers: dict[str, str],
        data: dict[str, str] | None = None,
    ) -> Any:
        try:
            async with httpx2.AsyncClient(
                timeout=self.gate.limits.timeout_seconds, transport=self._transport
            ) as client:
                response = await client.request(
                    method, url, params=params, headers=headers, data=data
                )
        except httpx2.TimeoutException as exc:
            raise SocialProviderError(f"{self.name} request timed out") from exc
        except httpx2.HTTPError as exc:
            raise SocialProviderError(f"could not reach {self.name}") from exc
        if response.status_code == 429:
            raise RateLimitReachedError(f"{self.name} rate limit reached")
        if response.status_code in (401, 403):
            raise SocialProviderError(
                f"{self.name} refused the credentials (HTTP {response.status_code})"
            )
        if response.status_code != 200:
            raise SocialProviderError(f"{self.name} returned HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError as exc:
            raise SocialProviderError(f"{self.name} returned invalid JSON") from exc

    def _result(
        self, posts: list[SocialPost], since: datetime, truncated: bool, checked: datetime
    ) -> SocialSearchResult:
        posts = [p for p in posts if p.created_at > since]
        oldest = min((p.created_at for p in posts), default=since)
        return SocialSearchResult(
            provider=self.name,
            platform=self.platform,
            checked_at=checked,
            posts=posts,
            complete_since=oldest if truncated else since,
        )


def _obj(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _time(value: Any) -> datetime | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return datetime.fromtimestamp(value, UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _minute(since: datetime) -> str:
    return since.replace(second=0, microsecond=0).isoformat()


# --- Reddit ---------------------------------------------------------------------------------


class RedditProvider(_HttpSocialProvider):
    """Reddit Data API search (`GET https://oauth.reddit.com/search`), application-only
    OAuth. Only public listings are read; post text is used for attribution and never
    stored. Permalinks are kept (public URLs)."""

    name = "Reddit"
    platform = "reddit"
    requirement = (
        "a Reddit API application approved under Reddit's Responsible Builder Policy "
        "(UPSCALE_REDDIT_CLIENT_ID, UPSCALE_REDDIT_CLIENT_SECRET, UPSCALE_REDDIT_USER_AGENT)"
    )
    token_url = "https://www.reddit.com/api/v1/access_token"
    api_url = "https://oauth.reddit.com"

    def __init__(
        self,
        client_id: str | None,
        client_secret: str | None,
        user_agent: str | None,
        settings: SocialProviderConfig,
        gate: RequestGate | None = None,
        transport: httpx2.AsyncBaseTransport | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        clock: Callable[[], float] = time.monotonic,
    ):
        super().__init__(settings, gate, transport, now)
        self._client_id = client_id
        self._client_secret = client_secret
        self._user_agent = user_agent
        self._clock = clock
        self._token: tuple[float, str] | None = None
        self._token_lock = asyncio.Lock()

    @property
    def configured(self) -> bool:
        return bool(self._client_id and self._client_secret and self._user_agent)

    def render_query(self, terms: Sequence[str]) -> str:
        return " OR ".join(f'"{_ticker_word(t)}"' for t in terms)

    async def _access_token(self) -> str:
        async with self._token_lock:
            if self._token and self._token[0] > self._clock():
                return self._token[1]
            basic = base64.b64encode(f"{self._client_id}:{self._client_secret}".encode()).decode()
            body = await self._request(
                "POST",
                self.token_url,
                None,
                {"Authorization": f"Basic {basic}", "User-Agent": self._user_agent or ""},
                data={"grant_type": "client_credentials"},
            )
            token = body.get("access_token") if isinstance(body, dict) else None
            expires = body.get("expires_in") if isinstance(body, dict) else None
            if not isinstance(token, str):
                raise SocialProviderError("Reddit returned no access token")
            ttl = float(expires) - 60 if isinstance(expires, int | float) else 600.0
            self._token = (self._clock() + max(ttl, 60.0), token)
            return token

    async def search(
        self, terms: Sequence[str], since: datetime, keyer: AuthorKeyer
    ) -> SocialSearchResult:
        if not self.configured:
            raise SocialProviderError(f"{self.name} is not configured")
        token = await self._access_token()
        headers = {"Authorization": f"Bearer {token}", "User-Agent": self._user_agent or ""}
        query = self.render_query(terms)
        span = self.now() - since
        t = "hour" if span <= timedelta(hours=1) else "day" if span <= timedelta(days=1) else "week"
        posts: list[SocialPost] = []
        after: str | None = None
        truncated = False
        for page in range(self.settings.max_pages):
            params = {
                "q": query,
                "sort": "new",
                "t": t,
                "type": "link",
                "limit": str(self.settings.max_results_per_page),
                "raw_json": "1",
            }
            if after:
                params["after"] = after
            body = await self._get(
                f"{self.api_url}/search", params, headers, f"{query}|{t}|{_minute(since)}|{page}"
            )
            data = body.get("data") if isinstance(body, dict) else None
            children = data.get("children") if isinstance(data, dict) else None
            if not isinstance(children, list):
                raise SocialProviderError("Reddit returned an unexpected response")
            posts += [p for c in children if (p := self._post(c, keyer)) is not None]
            after = _obj(data).get("after") if isinstance(_obj(data).get("after"), str) else None
            oldest = min((p.created_at for p in posts), default=None)
            if not after or (oldest is not None and oldest <= since):
                break
        else:
            truncated = after is not None
        return self._result(posts, since, truncated, self.now())

    def _post(self, child: Any, keyer: AuthorKeyer) -> SocialPost | None:
        d = child.get("data") if isinstance(child, dict) else None
        if not isinstance(d, dict) or not isinstance(d.get("name"), str):
            return None
        created = _time(d.get("created_utc"))
        author = d.get("author")
        if created is None or not isinstance(author, str) or author == "[deleted]":
            return None
        text = " ".join(str(d.get(k) or "") for k in ("title", "selftext"))
        url = d.get("url")
        permalink = d.get("permalink")
        promoted = d.get("promoted")
        return SocialPost(
            provider=self.name,
            platform=self.platform,
            post_id=d["name"],
            created_at=created,
            text=text,
            author_key=keyer(self.platform, author),
            author_handle=author.lower(),
            urls=[url] if isinstance(url, str) else [],
            likes=_int(d.get("score")),
            replies=_int(d.get("num_comments")),
            promoted=bool(promoted or d.get("is_created_from_ads_ui"))
            if promoted is not None or d.get("is_created_from_ads_ui") is not None
            else None,
            source_url=f"https://www.reddit.com{permalink}" if isinstance(permalink, str) else None,
        )


# --- Farcaster (Neynar) ---------------------------------------------------------------------


class NeynarFarcasterProvider(_HttpSocialProvider):
    """Farcaster cast search through Neynar (`GET /v2/farcaster/cast/search/`)."""

    name = "Farcaster (Neynar)"
    platform = "farcaster"
    requirement = "a Neynar API key (UPSCALE_NEYNAR_API_KEY); searches consume Neynar credits"
    api_url = "https://api.neynar.com/v2/farcaster/cast/search/"

    def __init__(
        self,
        api_key: str | None,
        settings: SocialProviderConfig,
        gate: RequestGate | None = None,
        transport: httpx2.AsyncBaseTransport | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        super().__init__(settings, gate, transport, now)
        self._api_key = api_key

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def render_query(self, terms: Sequence[str]) -> str:
        return " | ".join(f'"{t}"' for t in terms)

    async def search(
        self, terms: Sequence[str], since: datetime, keyer: AuthorKeyer
    ) -> SocialSearchResult:
        if not self.configured:
            raise SocialProviderError(f"{self.name} is not configured")
        query = self.render_query(terms)
        headers = {"x-api-key": self._api_key or "", "accept": "application/json"}
        posts: list[SocialPost] = []
        cursor: str | None = None
        truncated = False
        for page in range(self.settings.max_pages):
            params = {
                "q": query,
                "sort_type": "desc_chron",
                "limit": str(self.settings.max_results_per_page),
            }
            if cursor:
                params["cursor"] = cursor
            body = await self._get(
                self.api_url, params, headers, f"{query}|{_minute(since)}|{page}"
            )
            result = body.get("result") if isinstance(body, dict) else None
            casts = result.get("casts") if isinstance(result, dict) else None
            if not isinstance(casts, list):
                raise SocialProviderError("Neynar returned an unexpected response")
            posts += [p for c in casts if (p := self._post(c, keyer)) is not None]
            nxt = result.get("next") if isinstance(result, dict) else None
            cursor = nxt.get("cursor") if isinstance(nxt, dict) else None
            oldest = min((p.created_at for p in posts), default=None)
            if not cursor or (oldest is not None and oldest <= since):
                break
        else:
            truncated = cursor is not None
        return self._result(posts, since, truncated, self.now())

    def _post(self, cast: Any, keyer: AuthorKeyer) -> SocialPost | None:
        if not isinstance(cast, dict) or not isinstance(cast.get("hash"), str):
            return None
        author = _obj(cast.get("author"))
        fid = author.get("fid")
        created = _time(cast.get("timestamp"))
        if created is None or not isinstance(fid, int):
            return None
        reactions = _obj(cast.get("reactions"))
        replies = _obj(cast.get("replies"))
        embeds = _list(cast.get("embeds"))
        urls = [e["url"] for e in embeds if isinstance(e, dict) and isinstance(e.get("url"), str)]
        username = author.get("username")
        return SocialPost(
            provider=self.name,
            platform=self.platform,
            post_id=cast["hash"],
            created_at=created,
            text=str(cast.get("text") or ""),
            author_key=keyer(self.platform, str(fid)),
            author_handle=username.lower() if isinstance(username, str) else None,
            urls=urls,
            likes=_int(reactions.get("likes_count")),
            reposts=_int(reactions.get("recasts_count")),
            replies=_int(replies.get("count")),
        )


# --- X --------------------------------------------------------------------------------------


class XRecentSearchProvider(_HttpSocialProvider):
    """X API v2 recent search (`GET /2/tweets/search/recent`). X bills every post read, so
    this provider is only configured with a bearer token **and** an explicit paid opt-in,
    and stops at `max_reads_per_run` posts per run. Only post IDs and counts are kept."""

    name = "X"
    platform = "x"
    requirement = (
        "an X API bearer token (UPSCALE_X_BEARER_TOKEN) with pay-per-use credits, plus an "
        "explicit opt-in (x_allow_paid in UPSCALE_SOCIAL_CONFIG): each post read is billed"
    )
    api_url = "https://api.x.com/2/tweets/search/recent"

    def __init__(
        self,
        bearer_token: str | None,
        allow_paid: bool,
        max_reads_per_run: int,
        settings: SocialProviderConfig,
        gate: RequestGate | None = None,
        transport: httpx2.AsyncBaseTransport | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        super().__init__(settings, gate, transport, now)
        self._token = bearer_token
        self.allow_paid = allow_paid
        self.max_reads_per_run = max_reads_per_run
        self.reads_this_run = 0

    @property
    def configured(self) -> bool:
        return bool(self._token) and self.allow_paid and self.max_reads_per_run > 0

    def start_run(self) -> None:
        self.reads_this_run = 0

    def render_query(self, terms: Sequence[str]) -> str:
        return "(" + " OR ".join(terms) + ") -is:retweet"

    async def search(
        self, terms: Sequence[str], since: datetime, keyer: AuthorKeyer
    ) -> SocialSearchResult:
        if not self.configured:
            raise SocialProviderError(f"{self.name} is not configured")
        query = self.render_query(terms)
        start = max(since, self.now() - timedelta(days=7) + timedelta(minutes=1))
        headers = {"Authorization": f"Bearer {self._token}"}
        posts: list[SocialPost] = []
        token: str | None = None
        truncated = False
        for page in range(self.settings.max_pages):
            budget = self.max_reads_per_run - self.reads_this_run
            if budget < 10:
                truncated = True
                break
            params = {
                "query": query,
                "max_results": str(min(self.settings.max_results_per_page, budget)),
                "start_time": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "tweet.fields": "created_at,public_metrics,author_id,entities",
                "expansions": "author_id",
                "user.fields": "username",
            }
            if token:
                params["next_token"] = token
            body = await self._get(
                self.api_url, params, headers, f"{query}|{_minute(since)}|{page}"
            )
            if not isinstance(body, dict):
                raise SocialProviderError("X returned an unexpected response")
            rows = body.get("data", [])
            if not isinstance(rows, list):
                raise SocialProviderError("X returned an unexpected response")
            self.reads_this_run += len(rows)
            includes = _obj(body.get("includes"))
            users = {
                u["id"]: u.get("username")
                for u in _list(includes.get("users"))
                if isinstance(u, dict) and isinstance(u.get("id"), str)
            }
            posts += [p for r in rows if (p := self._post(r, users, keyer)) is not None]
            meta = _obj(body.get("meta"))
            token = meta.get("next_token") if isinstance(meta.get("next_token"), str) else None
            if not token:
                break
        else:
            truncated = token is not None
        return self._result(posts, start if start > since else since, truncated, self.now())

    def _post(self, row: Any, users: dict[str, Any], keyer: AuthorKeyer) -> SocialPost | None:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            return None
        created = _time(row.get("created_at"))
        author = row.get("author_id")
        if created is None or not isinstance(author, str):
            return None
        metrics = _obj(row.get("public_metrics"))
        entities = _obj(row.get("entities"))
        urls = [
            u.get("expanded_url")
            for u in _list(entities.get("urls"))
            if isinstance(u, dict) and isinstance(u.get("expanded_url"), str)
        ]
        handle = users.get(author)
        return SocialPost(
            provider=self.name,
            platform=self.platform,
            post_id=row["id"],
            created_at=created,
            text=str(row.get("text") or ""),
            author_key=keyer(self.platform, author),
            author_handle=handle.lower() if isinstance(handle, str) else None,
            urls=[u for u in urls if isinstance(u, str)],
            likes=_int(metrics.get("like_count")),
            replies=_int(metrics.get("reply_count")),
            reposts=_int(metrics.get("retweet_count")),
            quotes=_int(metrics.get("quote_count")),
            views=_int(metrics.get("impression_count")),
        )


# --- Discourse ------------------------------------------------------------------------------


class DiscourseForumProvider(_HttpSocialProvider):
    """One public Discourse forum's anonymous `search.json`. Each forum is its own source.
    Only add forums whose terms allow automated reading of public content."""

    requirement = "a public Discourse forum URL (UPSCALE_SOCIAL_FORUMS)"

    def __init__(
        self,
        base_url: str,
        settings: SocialProviderConfig,
        gate: RequestGate | None = None,
        transport: httpx2.AsyncBaseTransport | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        host = urlparse(base_url).hostname or base_url
        self.name = f"Discourse ({host})"
        self.platform = f"forum:{host}"
        super().__init__(settings, gate, transport, now)
        self.base_url = base_url.rstrip("/")

    @property
    def configured(self) -> bool:
        return self.base_url.startswith("https://")

    def render_query(self, terms: Sequence[str]) -> str:
        return " ".join(_ticker_word(t) for t in terms)

    async def search(
        self, terms: Sequence[str], since: datetime, keyer: AuthorKeyer
    ) -> SocialSearchResult:
        query = f"{self.render_query(terms)} order:latest after:{since:%Y-%m-%d}"
        body = await self._get(
            f"{self.base_url}/search.json",
            {"q": query},
            {"accept": "application/json"},
            f"{query}|{_minute(since)}",
        )
        if not isinstance(body, dict):
            raise SocialProviderError(f"{self.name} returned an unexpected response")
        rows = body.get("posts", [])
        topics = {
            t["id"]: t.get("title", "")
            for t in body.get("topics", [])
            if isinstance(t, dict) and isinstance(t.get("id"), int)
        }
        if not isinstance(rows, list):
            raise SocialProviderError(f"{self.name} returned an unexpected response")
        posts = [p for r in rows if (p := self._post(r, topics, keyer)) is not None]
        truncated = len(rows) >= self.settings.max_results_per_page
        return self._result(posts, since, truncated, self.now())

    def _post(self, row: Any, topics: dict[int, Any], keyer: AuthorKeyer) -> SocialPost | None:
        if not isinstance(row, dict) or not isinstance(row.get("id"), int):
            return None
        created = _time(row.get("created_at"))
        user = row.get("username")
        if created is None or not isinstance(user, str):
            return None
        topic = row.get("topic_id")
        number = row.get("post_number")
        title = topics.get(topic, "") if isinstance(topic, int) else ""
        text = f"{title} {row.get('blurb') or ''}"
        return SocialPost(
            provider=self.name,
            platform=self.platform,
            post_id=str(row["id"]),
            created_at=created,
            text=text,
            author_key=keyer(self.platform, user),
            author_handle=user.lower(),
            urls=urls_in(text),
            likes=_int(row.get("like_count")),
            source_url=(
                f"{self.base_url}/t/{topic}/{number}"
                if isinstance(topic, int) and isinstance(number, int)
                else None
            ),
        )
