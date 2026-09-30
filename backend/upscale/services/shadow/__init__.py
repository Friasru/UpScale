"""Shadow / Paper Strategy Engine v1: simulates what UpScale would have done, without
placing any real trade.

Versioned, immutable strategy configurations observe the same archived production
evidence UpScale had at each decision time (Growth Scout evaluations, exact-pool prices,
Analyze decisions) and decide ENTER / HOLD / EXIT / NO_ACTION; a paper portfolio per
strategy records immutable simulated entries, exits and equity (``IDEALIZED_NO_FEES``).

It has no private keys, no exchange credentials, no order placement, no transaction
signing and no provider requests. Simulated P/L is not real profit.
"""
