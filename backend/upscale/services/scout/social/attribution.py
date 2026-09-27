"""Deterministic attribution of a public post to exact tokens.

A post may refer to several tokens; each reference is judged on its own:

* **EXACT**: the post contains the token's exact contract / mint. (An EVM address that
  exists on several tracked chains is EXACT only when the post names the chain; otherwise
  it is AMBIGUOUS.)
* **STRONG**: a link to one of the token's official domains; or a post by the token's
  known official account that names its ticker or name; or ticker + name + the token's
  chain, with no other known token sharing that ticker and name on that chain.
* **PROBABLE**: no contract, but two independent references agree and no other known token
  fits them: ticker + name, name + chain, or ticker + chain (for tickers longer than
  `short_symbol_max_length`; a short ticker such as $AI also needs the name).
* **AMBIGUOUS**: several known tokens could be meant, or a single reference (a bare
  ticker, a bare name) with nothing to corroborate it. Scout's directory of known tokens
  is never treated as complete: "no competitor known" is not evidence. The mention is kept
  with its candidates, and never attached to any of them.
* **REJECTED**: the post points elsewhere: it names the ticker but gives a different
  contract, or names a different chain.

Only EXACT, STRONG and (down-weighted) PROBABLE mentions count toward momentum. There are
no rules for particular coins: only identity, chain, addresses, names, links and accounts.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from upscale.services.chains import normalize_address
from upscale.services.scout.social.config import AttributionConfig
from upscale.services.scout.social.models import Attribution, SocialPost, TokenIdentity
from upscale.services.scout.social.text import (
    address_family,
    chain_phrases,
    contains_phrase,
    domains,
    extract_addresses,
    extract_cashtags,
    is_cashtag_symbol,
    mentioned_chains,
    same_words,
)

COUNTED_LEVELS = frozenset({"EXACT", "STRONG", "PROBABLE"})


@dataclass(frozen=True)
class SearchReferences:
    """What identifies one token in a search, provider-neutral: a provider that bills per
    result can use these to fetch only posts attribution could count (see
    `AttributionIndex.references`)."""

    address: str
    cashtag: str | None  # "$SYM", only when the symbol can be written as a cashtag
    name: str | None  # only when it is independent evidence (not the ticker again)
    chain_phrases: tuple[str, ...]  # words / phrases attribution accepts as the chain
    short_symbol: bool  # a chain alone doesn't corroborate this ticker


class AttributionIndex:
    def __init__(
        self,
        tracked: Sequence[TokenIdentity],
        universe: Sequence[TokenIdentity] = (),
        config: AttributionConfig | None = None,
    ):
        """`tracked`: tokens mentions may be attributed to. `universe`: every other token
        known to exist (e.g. every token Scout has seen), used only to detect competing
        tokens that share a ticker or address."""
        self.config = config or AttributionConfig()
        self.tracked = {t.canonical_id: t for t in tracked}
        known = {t.canonical_id: t for t in [*universe, *tracked]}
        self.known = known
        self.by_address: dict[tuple[str, str], list[str]] = {}
        self.by_symbol: dict[str, set[str]] = {}
        for t in known.values():
            family = address_family(t.chain)
            if family:
                key = (family, normalize_address(t.chain, t.address) or t.address)
                self.by_address.setdefault(key, []).append(t.canonical_id)
            if t.symbol:
                self.by_symbol.setdefault(t.symbol.lower(), set()).add(t.canonical_id)

    def terms_for(self, identity: TokenIdentity) -> list[str]:
        """Search terms for one token: its address, and its $ticker when it has one that
        can be written as a cashtag (other symbols could never be attributed by ticker)."""
        terms = [identity.address]
        if identity.symbol and is_cashtag_symbol(identity.symbol):
            terms.append(f"${identity.symbol}")
        return terms

    def references(self, identity: TokenIdentity) -> SearchReferences:
        [address, *rest] = self.terms_for(identity)
        name = identity.name
        independent = (
            name is not None
            and len(name) >= self.config.min_name_length
            and not (identity.symbol and same_words(name, identity.symbol))
        )
        return SearchReferences(
            address=address,
            cashtag=rest[0] if rest else None,
            name=name if independent else None,
            chain_phrases=tuple(chain_phrases(identity.chain)),
            short_symbol=identity.symbol is not None
            and len(identity.symbol) <= self.config.short_symbol_max_length,
        )

    def attribute(self, post: SocialPost) -> list[Attribution]:
        text = " ".join([post.text, *post.urls])
        addresses = extract_addresses(text)
        chains = mentioned_chains(text)
        tags = extract_cashtags(text)
        linked = domains(post.urls)
        out: dict[str, Attribution] = {}
        ambiguous: list[Attribution] = []

        # EXACT: the contract / mint itself
        for family, address in addresses:
            ids = [i for i in self.by_address.get((family, address), []) if i in self.tracked]
            narrowed = [i for i in ids if self.tracked[i].chain in chains] if chains else ids
            narrowed = narrowed or ids
            if len(narrowed) == 1:
                out[narrowed[0]] = Attribution(
                    level="EXACT",
                    canonical_id=narrowed[0],
                    reason="the post contains the token's exact contract / mint",
                    token_reference=address,
                )
            elif len(narrowed) > 1:
                ambiguous.append(
                    Attribution(
                        level="AMBIGUOUS",
                        canonical_id=None,
                        reason="this address exists on several chains and the post doesn't say which",
                        token_reference=address,
                        candidates=sorted(narrowed),
                    )
                )
        post_addresses = {a for _, a in addresses}

        for cid, token in self.tracked.items():
            if cid in out:
                continue
            ticker = f"${token.symbol}".lower() if token.symbol else None
            has_ticker = token.symbol is not None and token.symbol.lower() in tags
            # A name that is just the ticker again ("Degen" for $DEGEN) is no independent
            # evidence: every ticker mention would also "name" the token.
            has_name = (
                token.name is not None
                and len(token.name) >= self.config.min_name_length
                and not (token.symbol and same_words(token.name, token.symbol))
                and contains_phrase(post.text, token.name)
            )
            reference = ticker if has_ticker and ticker else (token.name or "")
            own_address = normalize_address(token.chain, token.address) or token.address

            # STRONG: official links / accounts
            official_domain = linked & token.official_domains
            if official_domain:
                out[cid] = Attribution(
                    level="STRONG",
                    canonical_id=cid,
                    reason=f"links to the token's official site {sorted(official_domain)[0]}",
                    token_reference=sorted(official_domain)[0],
                )
                continue
            handle = (post.author_handle or "").lower()
            official = handle and handle in token.official_accounts.get(post.platform, set())
            if not (has_ticker or has_name):
                continue
            if official:
                out[cid] = Attribution(
                    level="STRONG",
                    canonical_id=cid,
                    reason="posted by the token's known official account, naming the token",
                    token_reference=reference,
                )
                continue

            # REJECTED: the post points to another token
            if post_addresses and own_address not in post_addresses:
                out[cid] = Attribution(
                    level="REJECTED",
                    canonical_id=cid,
                    reason="the post names this ticker / name but gives a different contract",
                    token_reference=reference,
                )
                continue
            if chains and token.chain not in chains:
                out[cid] = Attribution(
                    level="REJECTED",
                    canonical_id=cid,
                    reason=f"the post refers to another chain ({', '.join(sorted(chains))})",
                    token_reference=reference,
                )
                continue

            competitors = self._competitors(token, has_ticker, chains)
            if competitors:
                ambiguous.append(
                    Attribution(
                        level="AMBIGUOUS",
                        canonical_id=None,
                        reason=f"{len(competitors) + 1} known tokens could be meant",
                        token_reference=reference,
                        candidates=sorted({cid, *competitors}),
                    )
                )
                continue
            # Independent references: the ticker, the token's (real) name, its chain. A
            # single one is never enough, however few tokens Scout knows: its directory
            # only holds what Scout has seen, so "no competitor known" isn't evidence.
            on_chain = token.chain in chains
            short = (
                token.symbol is not None
                and len(token.symbol) <= self.config.short_symbol_max_length
            )
            if has_ticker and has_name and on_chain:
                out[cid] = Attribution(
                    level="STRONG",
                    canonical_id=cid,
                    reason="names the token's ticker, name and chain; no other known token matches",
                    token_reference=reference,
                )
            elif has_name and (has_ticker or on_chain) or has_ticker and on_chain and not short:
                context = "the name" if has_ticker and has_name else "the chain"
                out[cid] = Attribution(
                    level="PROBABLE",
                    canonical_id=cid,
                    reason=f"matches one tracked token by ticker / name, corroborated by {context}",
                    token_reference=reference,
                )
            else:
                ambiguous.append(
                    Attribution(
                        level="AMBIGUOUS",
                        canonical_id=None,
                        reason=(
                            "a short ticker and its chain alone could mean many tokens"
                            if has_ticker and on_chain
                            else "one reference only (ticker or name, no contract or "
                            "corroborating context): other tokens can't be ruled out"
                        ),
                        token_reference=reference,
                        candidates=[cid],
                    )
                )
        return [*out.values(), *_unique(ambiguous)]

    def _competitors(self, token: TokenIdentity, by_ticker: bool, chains: set[str]) -> set[str]:
        """Other known tokens the same words could mean."""
        others: set[str] = set()
        if by_ticker and token.symbol:
            others |= self.by_symbol.get(token.symbol.lower(), set())
        if token.name:
            others |= {
                c for c, t in self.known.items() if t.name and t.name.lower() == token.name.lower()
            }
        others.discard(token.canonical_id)
        if chains:
            others = {c for c in others if self.known[c].chain in chains}
        return others


def _unique(items: list[Attribution]) -> list[Attribution]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    out = []
    for a in items:
        key = (a.token_reference.lower(), tuple(a.candidates))
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out
