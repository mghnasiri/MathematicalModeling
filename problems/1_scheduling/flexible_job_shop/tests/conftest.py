import sys
import os

_fjsp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _fjsp_dir in sys.path:
    sys.path.remove(_fjsp_dir)
sys.path.insert(0, _fjsp_dir)
