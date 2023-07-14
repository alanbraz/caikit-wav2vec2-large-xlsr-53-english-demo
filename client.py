# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Third Party
import grpc

# Local
from caikit.runtime.service_factory import ServicePackageFactory
from tox_predict.data_model import SmilesInput

inference_service = ServicePackageFactory().get_service_package(
    ServicePackageFactory.ServiceType.INFERENCE,
)

port = 8085
channel = grpc.insecure_channel(f"localhost:{port}")

client_stub = inference_service.stub_class(channel)

# print(dir(client_stub))

for text in ["CC(NC)C(O)c1ccccc1",
             "Cc1cc(ccc1N)-c1cc(C)c(N)cc1",
             "Cc1cc2c(cc1C)N=C1C(=NC(=O)NC1=O)N2CC(O)C(O)C(O)CO",
             "CCCCCCCCCCCC(=O)OC=C",
             "O=C=Nc1cc(c(Cl)cc1)C(F)(F)F"]:
    input_text_proto = SmilesInput(text=text).to_proto()
    request = inference_service.messages.ToxPredictionTaskRequest(
        text_input=input_text_proto
    )
    response = client_stub.ToxPredictionTaskPredict(
        request, metadata=[("mm-model-id", "tox_predict")]
    )
    print("Text:", text)
    print("RESPONSE:", response.score)
    print("EPA:", response.epa)
    print("EPA mg/kg:", response.epa_mgkg)
