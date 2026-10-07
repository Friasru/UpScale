"""Bounded data retention for the Railway /data volume (Scout and Evidence Archive only).

Shadow databases are never written. See `upscale.services.retention.engine` for what each
table keeps, `config` for the policy and `cli` for the status / dry-run / cleanup commands.
"""
