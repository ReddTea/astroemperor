# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import platform
import multiprocessing

_PLATFORM_SYSTEM = platform.system()  # PROBABLY BEST, should bench
_SYS_PLATFORM = sys.platform
_OS_NAME = os.name
_CORES = multiprocessing.cpu_count()
_TERMINAL_WIDTH = os.get_terminal_size().columns
_OS_ROOT = os.path.dirname(__file__)  # where im now
        