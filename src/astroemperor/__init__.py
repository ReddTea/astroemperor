# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Metadata about the package
__version__ = '0.9.10'
__name__ = 'astroemperor'
__url__ = "https://astroemperor.readthedocs.io"

from .emp import Simulation

__all__ = ['Simulation']