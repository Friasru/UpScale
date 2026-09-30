"""Point-in-Time Evidence Archive: an append-only record of what production UpScale
actually knew about an asset, and when (market, DEX, on-chain safety, social, Growth Scout,
Analyze decisions), so Replay Lab can reproduce historical decisions with no lookahead and
no current-data substitution.

Production code only imports `hooks` (dependency-free); the recorder, store, payloads,
enrichment and status modules are loaded by `upscale.services` / the CLI.
"""
