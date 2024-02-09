# DeepFake Detection API

ML pipeline with FastAPI + Docker to serve the model as an API.

*ReDeepFake model was obtained from [here](https://huggingface.co/dataflow/redeepfake)*



### Note:
1. `libgl1` is required by `opencv-python`, most systems have it preinstalled but it would be required when Dockerizing
2. The ReDeepFake model is provided by [DataFlow](https://dataflow.kz) under the MIT License.
