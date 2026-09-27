"""Scout social / attention intelligence: public attention evidence for exact tokens.

Social activity is evidence only: it never produces BUY / SELL / WAIT. See `analysis` for
how momentum states are determined and `attribution` for how mentions are tied to tokens.
"""

from upscale.services.scout.social.attribution import AttributionIndex
from upscale.services.scout.social.config import (
    SocialConfig,
    SocialConfigError,
    load_social_config,
)
from upscale.services.scout.social.models import (
    NOT_A_TRADING_SIGNAL,
    SocialEvent,
    SocialMomentum,
    SocialRun,
    SocialSourceSnapshot,
    TokenIdentity,
)
from upscale.services.scout.social.providers import (
    DiscourseForumProvider,
    NeynarFarcasterProvider,
    RedditProvider,
    SocialTrendProvider,
    StaticSocialProvider,
    XRecentSearchProvider,
)
from upscale.services.scout.social.service import (
    SocialScoutService,
    identities_from_candidates,
    known_identities,
)
from upscale.services.scout.social.store import SocialStore

__all__ = [
    "NOT_A_TRADING_SIGNAL",
    "AttributionIndex",
    "DiscourseForumProvider",
    "NeynarFarcasterProvider",
    "RedditProvider",
    "SocialConfig",
    "SocialConfigError",
    "SocialEvent",
    "SocialMomentum",
    "SocialRun",
    "SocialScoutService",
    "SocialSourceSnapshot",
    "SocialStore",
    "SocialTrendProvider",
    "StaticSocialProvider",
    "TokenIdentity",
    "XRecentSearchProvider",
    "identities_from_candidates",
    "known_identities",
    "load_social_config",
]
