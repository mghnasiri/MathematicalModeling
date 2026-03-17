import sys
import os

_rcpsp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _rcpsp_dir in sys.path:
    sys.path.remove(_rcpsp_dir)
sys.path.insert(0, _rcpsp_dir)
