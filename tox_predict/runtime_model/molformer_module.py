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
import torch
import yaml
from yaml import SafeLoader

# Local
from caikit.core import ModuleBase, ModuleLoader, ModuleSaver, module, TaskBase, task
# from fast_transformers.masking import LengthMask as LM

from tokenizer import MolTranBertTokenizer
from tox_predict.data_model.tox_prediction import (
    ScoreOutput,
    SmilesInput,
)

from tox_predict.runtime_model.helper import dotdict, convert_to_epa, convert_to_mgkg
from tox_predict.runtime_model.molformer_predict_tox import LightningModule


@task(
    required_parameters={"text_input": SmilesInput},
    output_type=ScoreOutput,
)
class ToxPredictionTask(TaskBase):
    pass


@module(
    "molFormertox-predict",
    "MolFormerModule",
    "0.0.1",
    ToxPredictionTask,
)
class MolFormerModule(ModuleBase):
    """Class to wrap sentiment analysis pipeline from HuggingFace"""

    def __init__(self, model_path) -> None:
        super().__init__()
        loader = ModuleLoader(model_path)
        config = loader.config
        # model = pipeline(model=config.hf_artifact_path, task="sentiment-analysis")
        # self.sentiment_pipeline = model

        print(os.getcwd())
        with open('data/hparams.yaml') as f:
            data = yaml.load(f, Loader=SafeLoader)
            print(data)

        hparams = dotdict(data)
        tokenizer = MolTranBertTokenizer('data/bert_vocab.txt')
        self.model = LightningModule(hparams, tokenizer).load_from_checkpoint(
            'data/checkpoints/N-Step-Checkpoint_3_30000.ckpt',
            strict=False,
            config=hparams,
            tokenizer=tokenizer,
            vocab=len(tokenizer.vocab),
            map_location=torch.device('cpu'))
        self.model.eval()

    def run(self, text_input: SmilesInput) -> ScoreOutput:
        """Run HF sentiment analysis
        Args:
            text_input: SmilesInput
        Returns:
            ClassificationPrediction: predicted classes with their confidence score.
        """

        # Tokenizer - Creating tokens from SMILES
        tokens = self.model.tokenizer([text_input.text],
                                      padding=True, truncation=True,
                                      add_special_tokens=True, return_tensors="pt")
        idx = torch.tensor(tokens['input_ids'])
        mask = torch.tensor(tokens['attention_mask'])

        # Data transformation to feed the model
        token_embeddings = self.model.tok_emb(idx)  # each index maps to a (learnable) vector
        x = self.model.drop(token_embeddings)
        # x = self.model.blocks(x, length_mask=LM(mask.sum(-1)))
        token_embeddings = x

        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        loss_input = sum_embeddings / sum_mask

        outmap_min, _ = torch.min(loss_input, dim=1, keepdim=True)
        outmap_max, _ = torch.max(loss_input, dim=1, keepdim=True)
        outmap = (loss_input - outmap_min) / (outmap_max - outmap_min)  # Broadcasting rules apply

        print('Predicting...')

        outputs = self.model.net.forward(outmap).squeeze()
        print(outputs)

        # # Converting to Epa Categories
        pred_epa = list(convert_to_epa(outputs.reshape(1), [text_input.text]))
        # # Converting to Mg/Kg  Units
        pred_epa_mgkg = list(convert_to_mgkg(outputs.reshape(1), [text_input.text]))
        print(f"EPA:{pred_epa[0]} EPA-mgkg: {pred_epa_mgkg[0]:5.2f}")

        return ScoreOutput(score=outputs.item(), epa=pred_epa[0], epa_mgkg=pred_epa_mgkg[0])

    @classmethod
    def bootstrap(cls, model_path="distilbert-base-uncased-finetuned-sst-2-english"):
        """Load a HuggingFace based caikit model
        Args:
            model_path: str
                Path to HuggingFace model
        Returns:
            HuggingFaceModel
        """
        return cls(model_path)

    def save(self, model_path, **kwargs):
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
    def load(cls, model_path):
        """Load a HuggingFace based caikit model
        Args:
            model_path: str
                Path to HuggingFace model
        Returns:
            HuggingFaceModel
        """
        return cls(model_path)
