
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
import gradio as gr
import grpc

# Local
from caikit.runtime.service_factory import ServicePackageFactory
from caikit.runtime.grpc_server import RuntimeGRPCServer
from tox_predict.data_model import SmilesInput

inference_service = ServicePackageFactory().get_service_package(
    ServicePackageFactory.ServiceType.INFERENCE,
)

port = 8085
channel = grpc.insecure_channel(f"localhost:{port}")

client_stub = inference_service.stub_class(channel)
  
def tox(smiles):
    input_text_proto = SmilesInput(text=smiles).to_proto()
    request = inference_service.messages.ToxPredictionTaskRequest(
        text_input=input_text_proto
    )
    response = client_stub.ToxPredictionTaskPredict(
        request, metadata=[("mm-model-id", "tox_predict")]
    )
    return response.score

# We instantiate the Textbox class
textbox = gr.Textbox(label="Smiles:", placeholder="CCC", lines=1)
numberbox = gr.Number(label="Score")

with RuntimeGRPCServer(inference_service=inference_service, training_service=None) as backend:
    gr.Interface(
        fn=tox, 
        inputs=textbox, 
        outputs=numberbox, 
        title="MolFormer", 
        # description = """
        # The bot was trained to answer questions based on Rick and Morty dialogues. Ask Rick anything!
        # <img src="https://huggingface.co/spaces/course-demos/Rick_and_Morty_QA/resolve/main/rick.png" width=200px>
        # """,
        # article = "Check out [the original Rick and Morty Bot](https://huggingface.co/spaces/kingabzpro/Rick_and_Morty_Bot) that this demo is based off of.",
        examples=[ "Cc1ncc([N+](=O)[O-])n1CCO", "CCC" ]).launch(share=False, show_tips=False, server_name="0.0.0.0", server_port=8080)
    backend.server.wait_for_termination()
