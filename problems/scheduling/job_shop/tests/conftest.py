import sys
import os

# Ensure job_shop directory is first in sys.path for correct instance imports
_job_shop_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _job_shop_dir in sys.path:
    sys.path.remove(_job_shop_dir)
sys.path.insert(0, _job_shop_dir)
