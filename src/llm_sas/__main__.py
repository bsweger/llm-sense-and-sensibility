import sys

import structlog

from llm_sas.app import main

logger = structlog.get_logger()

rc = 1
try:
    main()
    rc = 0
except Exception as e:
    logger.exception(the_error=e)
    print("Error:", e, file=sys.stderr)
sys.exit(rc)
