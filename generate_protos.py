import shutil
from caikit.runtime.dump_services import dump_services
import tox_predict

shutil.rmtree("protos", ignore_errors=True)
dump_services("protos")

# python -m caikit.runtime.dump_services("protos")