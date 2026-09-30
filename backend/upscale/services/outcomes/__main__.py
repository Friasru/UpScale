"""python -m upscale.services.outcomes integrity-audit [...]"""

import sys

from upscale.services.outcomes.audit import main

if len(sys.argv) < 2 or sys.argv[1] != "integrity-audit":
    print("usage: python -m upscale.services.outcomes integrity-audit [--help]", file=sys.stderr)
    sys.exit(2)
sys.exit(main(sys.argv[2:]))
