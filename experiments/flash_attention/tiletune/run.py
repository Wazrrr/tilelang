"""Family-study entry point with the existing fixed-grid API preserved."""

import sys
from . import legacy as _impl

if __name__ == "__main__":
    raise SystemExit(_impl.main())
else:
    sys.modules[__name__] = _impl
