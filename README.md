# RKLLAMA-SERVER
This library lets you quickly create an OpenAI /completions endpoint compatible server with rkllm and rknn(for vision support) inferencing support. 

## Requirements
1. rk3588/rk3576 processor (Currently only tested on rk3588, should work on rk3576)
2. rknpu driver 0.9.8 installed.
Check current version using: `sudo cat /sys/kernel/debug/rknpu/version`
3. uv for python env management.
Install via: 
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installtion and run
### Setup library
```
git clone https://github.com/arbitropy/rkllama-server.git
cd rkllama-server
```
### Configuration
Download the `<model>.rkllm` and optionally `<model-vision.rknn>` files. (TODO: Provide links for preconverted models)
Edit `config.yaml.example` into `config.yaml` and setup the model path, huggingface path(for tokenizer) and generation parameters inside the config file.
Setting the vision special tokens is mandatory for VLMs.

### Run the server
```
uv run server.py
```
### Test the server (TODO: Auto text/multimodal)
```
uv run quick-client-test.py
```
### GUI quick chatbot
```
uv run streamlit run streamlit-client.py
```

## TODO
- Multimodal multiturn and multiimage chat doesn't seem to work. Regular multi turn works
- /models and /completions api full spec support
- Implement config.yaml
- client using openai python