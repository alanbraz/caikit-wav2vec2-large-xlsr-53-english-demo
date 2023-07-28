# from transformers import pipeline
# p = pipeline("automatic-speech-recognition")

# from huggingsound import SpeechRecognitionModel
# model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
# audio_paths = ["/path/to/file.mp3", "/path/to/another_file.wav"]
# p2 = pipeline("automatic-speech-recognition", "jonatasgrosman/wav2vec2-large-xlsr-53-english")

import gradio as gr
import grpc
import sys
from os import path, listdir


# Local
import caikit
from caikit.runtime import grpc_server, http_server
# import requests
from caikit.runtime.service_factory import ServicePackageFactory
from speech.data_model.transcription import AudioPath

models_directory = path.abspath(path.join(path.dirname(__file__), "models"))
caikit.config.configure(
    config_dict={
        "merge_strategy": "merge",
        "runtime": {
            "local_models_dir": models_directory,
            "library": "speech",
        },
    }
)

sys.path.append(
    path.abspath(path.join(path.dirname(__file__), "../"))
) 

inference_service = ServicePackageFactory().get_service_package(
    ServicePackageFactory.ServiceType.INFERENCE,
)

model_id = "speech"

port = 8085
channel = grpc.insecure_channel(f"localhost:{port}")
client_stub = inference_service.stub_class(channel)

# default pipeline
from transformers import pipeline
p = pipeline("automatic-speech-recognition")
# facebook/wav2vec2-base-960h

def transcribe(path):
    request = inference_service.messages.HuggingFaceSpeechTaskRequest(
        audio_input=AudioPath(file_path=path).to_dict()
    )
    response = client_stub.HuggingFaceSpeechTaskPredict(
        request, metadata=[("mm-model-id", model_id)], timeout=100
    )
    return [ p([path])[0]["text"], response.text, response.probabilities ]

dir_path = "samples"

with grpc_server.RuntimeGRPCServer(inference_service=inference_service, training_service=None) as backend:
    gr.Interface(
        fn=transcribe, 
        inputs=gr.Microphone(type="filepath", format="mp3"), 
        outputs=[ gr.Textbox(label="Default model text"), gr.Textbox(label="Transcription"), gr.Textbox(label="Probabilities") ],
        title="Speech to text demo", 
        allow_flagging="never",
        description = "This demos sends the same audio to the HuggingFace automatic-speech-recognition pipeline default model [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) and the [jonatasgrosman/wav2vec2-large-xlsr-53-english](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english) model.",
        article = "Check out [the original model](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english).",
        examples= [ path.join(dir_path, file_path) for file_path in sorted(listdir(dir_path)) ]
        ).launch(share=False, show_tips=False, server_port=8080, server_name=None)
    backend.wait_for_termination()