import sys, pathlib
from hailo_sdk_client import ClientRunner
har, out, timeout = sys.argv[1], sys.argv[2], sys.argv[3]
runner = ClientRunner(har=har)
runner.load_model_script(f"allocator_param(timeout={timeout})\n")
pathlib.Path(out).write_bytes(runner.compile())
print("== written:", out)
