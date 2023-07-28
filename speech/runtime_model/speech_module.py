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

# Standard
import os

# Third Party
# from transformers import pipeline  # pylint: disable=import-error
from huggingsound import SpeechRecognitionModel

# Local
from caikit.core import ModuleBase, ModuleLoader, ModuleSaver, TaskBase, module, task
from speech.data_model.transcription import AudioPath, Transcription


@task(
    required_parameters={"audio_input": AudioPath},
    output_type=Transcription,
)
class HuggingFaceSpeechTask(TaskBase):
    pass


@module(
    "wav2vec2-large-xlsr-53-english",
    "HuggingFaceSpeechModule",
    "0.0.1",
    HuggingFaceSpeechTask,
)
class HuggingFaceSpeechModule(ModuleBase):
    """Class to wrap sentiment analysis pipeline from HuggingFace"""

    def __init__(self, model_path) -> None:
        super().__init__()
        loader = ModuleLoader(model_path)
        config = loader.config
        # print(config)
        # exit(0)
        self.model = SpeechRecognitionModel(config.model)

    def run(  # pylint: disable=arguments-differ
        self, audio_input: AudioPath
    ) -> Transcription:
        """Run HF sentiment analysis
        Args:
            file_path_input: str
        Returns:
            Transcription: predicted classes with their confidence score.
        """
        print("audio_input.file_path", audio_input.file_path)
        raw_results = self.model.transcribe([audio_input.file_path])[0]
        return Transcription(raw_results["transcription"], raw_results["probabilities"])

    @classmethod
    def bootstrap(
        cls, model_path="wav2vec2-large-xlsr-53-english"
    ):  # pylint: disable=arguments-differ
        """Load a HuggingFace based caikit model
        Args:
            model_path: str
                Path to HuggingFace model
        Returns:
            HuggingFaceModel
        """
        return cls(model_path)

    def save(self, model_path, **kwargs):  # pylint: disable=arguments-differ
        module_saver = ModuleSaver(
            self,
            model_path=model_path,
        )

        # Extract object to be saved
        with module_saver:
            # Make the directory to save model artifacts
            rel_path, _ = module_saver.add_dir("hf_model")
            save_path = os.path.join(model_path, rel_path)
            self.sentiment_pipeline.save_pretrained(save_path)
            module_saver.update_config({"hf_artifact_path": rel_path})

    # this is how you load the model, if you have a caikit model
    @classmethod
    def load(cls, model_path):  # pylint: disable=arguments-differ
        """Load a HuggingFace based caikit model
        Args:
            model_path: str
                Path to HuggingFace model
        Returns:
            HuggingFaceModel
        """
        return cls(model_path)