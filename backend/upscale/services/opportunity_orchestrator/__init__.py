"""Scout -> Safety -> Opportunity orchestration bridge (B1: read-only admission, priority
and dry-run). It coordinates components and owns none of their logic: never Scout scoring,
Safety rules, Opportunity rules, providers or execution. B1 writes nothing and makes no
provider call."""
