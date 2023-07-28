# wav2vec2-large-xlsr-53-english caikit demo

This demos sends the same audio to the HuggingFace automatic-speech-recognition pipeline default model [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) and the [jonatasgrosman/wav2vec2-large-xlsr-53-english](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english) model.

It uses the [caikit](https://github.com/caikit/caikit) to serve the model.

## Before Starting

The following tools are required:

- [python](https://www.python.org) (v3.8+)
- [pip](https://pypi.org/project/pip/) (v23.0+)

**Note: Before installing dependencies and to avoid conflicts in your environment, it is advisable to use a [virtual environment(venv)](https://docs.python.org/3/library/venv.html).**

Install the dependencies: `pip install -r requirements.txt`

## Running the Caikit runtime

In one terminal, start the runtime server:

```shell
python app.py
```