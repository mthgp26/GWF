"""pytest 根 conftest：把 code/ 目录加入 sys.path，保证 ``import gwf`` 可用。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
