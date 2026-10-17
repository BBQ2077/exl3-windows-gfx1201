import os, sys
if os.environ.get("EXL3_PYTEST_LIB"):   # pytest 裝在別的資料夾時（pip install pytest --target <dir>）
    sys.path.append(os.environ["EXL3_PYTEST_LIB"])
here = os.path.dirname(os.path.abspath(__file__))
up = os.path.join(os.path.dirname(here), "upstream")
sys.path.insert(0, up)
os.chdir(up)
import pytest
raise SystemExit(pytest.main(["-p", "no:cacheprovider", "-q"] + sys.argv[1:]))
