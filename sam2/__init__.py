"""Top-level SAM2 repo package shim.

Expose inner package modules (sam2/ directory) at the top-level so that
targets like 'sam2.sam2_video_predictor' and 'sam2.modeling' are importable
when running from the repo source tree without installing the package.
"""

import importlib
import sys as _sys

# Ensure the inner package is importable and Hydra configs are initialized
_inner_pkg = importlib.import_module('sam2.sam2')

# First, import and alias inner modeling so imports inside other inner modules resolve
_inner_modeling = importlib.import_module('sam2.sam2.modeling')
_sys.modules.setdefault('sam2.modeling', _inner_modeling)

# Also import and alias inner utils, needed by modeling
_inner_utils = importlib.import_module('sam2.sam2.utils')
_sys.modules.setdefault('sam2.utils', _inner_utils)

# Now import the inner video predictor module and alias it
_inner_vidpred = importlib.import_module('sam2.sam2.sam2_video_predictor')
_sys.modules.setdefault('sam2.sam2_video_predictor', _inner_vidpred)

# Also expose as attributes on the top-level package object
modeling = _inner_modeling
sam2_video_predictor = _inner_vidpred
utils = _inner_utils
