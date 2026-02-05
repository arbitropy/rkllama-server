"""
Multimodal LLM Server with HuggingFace Chat Template Integration

This server combines:
- RKNN-accelerated vision encoder for image embeddings
- RKNN-accelerated LLM for text generation  
- HuggingFace tokenizer for proper chat formatting

Key Features:
- OpenAI-compatible API endpoint (/v1/chat/completions)
- Automatic chat template application from tokenizer
- Streaming responses with Server-Sent Events
- Multimodal support (text + image)

Configuration:
- Update model paths and vision tokens at top of file
- Tokenizer must have chat_template or server will fail
- Vision tokens must match model training data
"""
import ctypes
import os
import sys
import threading
import queue
import json
import base64

# Set LD_LIBRARY_PATH for local RKNN library before importing RKNNLite
LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
if os.path.exists(LIB_DIR):
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = f"{LIB_DIR}:{current_ld_path}" if current_ld_path else LIB_DIR

import cv2
import numpy as np
import logging
from flask import Flask, request, Response, jsonify
from rknnlite.api import RKNNLite
from transformers import AutoTokenizer

# ==========================================
# Logging Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ==========================================
# Configuration
# ==========================================
# # Qwen 3
# LLM_MODEL_PATH = "models/qwen3-vl-2b-instruct_w8a8_rk3588/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm"
# ENC_MODEL_PATH = "models/qwen3-vl-2b-instruct_w8a8_rk3588/qwen3-vl-2b-instruct_w8a8_rk3588.rknn"
# TOKENIZER_PATH = "Qwen/Qwen3-VL-2b-Instruct"
# # Model Specific Settings
# IMG_SIZE = 448
# IMG_TOKENS = 196
# IMG_EMBED_DIM = 2048
# # Vision Token Configuration (model-specific)
# VISION_START_TOKEN = "<|vision_start|>"
# VISION_END_TOKEN = "<|vision_end|>"
# VISION_PAD_TOKEN = "<|image_pad|>"

# InternVL 3
LLM_MODEL_PATH = "models/internvl3_5_2b/internvl3_5-2b-instruct_w8a8_rk3588.rkllm"
ENC_MODEL_PATH = "models/internvl3_5_2b/internvl3_5-2b_vision_rk3588.rknn"
TOKENIZER_PATH = "OpenGVLab/InternVL3_5-2b"
# Model Specific Settings
IMG_SIZE = 448
IMG_TOKENS = 256
# Vision Token Configuration (model-specific)
VISION_START_TOKEN ="<img>"
VISION_END_TOKEN = "</img>"
VISION_PAD_TOKEN = "<IMG_CONTEXT>"

# Library Configuration
LIB_DIR = "./lib"
RKLLM_LIB_PATH = os.path.join(LIB_DIR, "librkllmrt.so")
RKNN_LIB_PATH = os.path.join(LIB_DIR, "librknnrt.so")

TARGET_PLATFORM = "rk3588"  # 'rk3588' or 'rk3576'
HOST_IP = "0.0.0.0"
PORT = 8080

IMAGE_PLACEHOLDER = "<image>"  # Used in user prompts

# ==========================================
# C-Types Definitions
# ==========================================
# Load RKLLM library from local lib folder
try:
    rkllm_lib = ctypes.CDLL(RKLLM_LIB_PATH)
    logger.info(f"Loaded RKLLM library from {RKLLM_LIB_PATH}")
except OSError as e:
    logger.critical(f"Failed to load librkllmrt.so: {e}")
    logger.critical(f"Ensure {RKLLM_LIB_PATH} exists")
    sys.exit(1)

# Verify RKNN library is available for RKNNLite
if os.path.exists(RKNN_LIB_PATH):
    logger.info(f"RKNN library found at {RKNN_LIB_PATH}")
else:
    logger.warning(f"RKNN library not found at {RKNN_LIB_PATH}")
    logger.warning("RKNNLite will attempt to use system-installed library")

RKLLM_Handle_t = ctypes.c_void_p

# Enums
class LLMCallState:
    RKLLM_RUN_NORMAL = 0
    RKLLM_RUN_WAITING = 1
    RKLLM_RUN_FINISH = 2
    RKLLM_RUN_ERROR = 3

class RKLLMInputType:
    RKLLM_INPUT_PROMPT = 0
    RKLLM_INPUT_TOKEN = 1 
    RKLLM_INPUT_EMBED = 2
    RKLLM_INPUT_MULTIMODAL = 3

class RKLLMInferMode:
    RKLLM_INFER_GENERATE = 0
    RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
    RKLLM_INFER_GET_LOGITS = 2

# Structures
class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104)
    ]

class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]

class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [("embed", ctypes.POINTER(ctypes.c_float)), ("n_tokens", ctypes.c_size_t)]

class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [("input_ids", ctypes.POINTER(ctypes.c_int32)), ("n_tokens", ctypes.c_size_t)]

class RKLLMMultiModalInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t)
    ]

class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModalInput)
    ]

class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", ctypes.c_int), # RKLLMInputType
        ("input_data", RKLLMInputUnion)
    ]

class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [("lora_adapter_name", ctypes.c_char_p)]

class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [("save_prompt_cache", ctypes.c_int), ("prompt_cache_path", ctypes.c_char_p)]

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int), # RKLLMInferMode
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("keep_history", ctypes.c_int)
    ]

class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [("hidden_states", ctypes.POINTER(ctypes.c_float)), ("embd_size", ctypes.c_int), ("num_tokens", ctypes.c_int)]

class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [("logits", ctypes.POINTER(ctypes.c_float)), ("vocab_size", ctypes.c_int), ("num_tokens", ctypes.c_int)]

class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [("prefill_time_ms", ctypes.c_float), ("prefill_tokens", ctypes.c_int), ("generate_time_ms", ctypes.c_float), ("generate_tokens", ctypes.c_int), ("memory_usage_mb", ctypes.c_float)]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat)
    ]

CALLBACK_TYPE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)

# ==========================================
# Vision Encoder
# ==========================================
class VisionEncoder:
    def __init__(self, model_path):
        logger.info(f"Initializing Vision Encoder: {model_path}")
        if not os.path.exists(model_path):
            logger.error(f"Encoder model not found at {model_path}")
            sys.exit(1)
            
        self.rknn = RKNNLite()
        
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            logger.error(f"rknn.load_rknn failed with code {ret}")
            sys.exit(1)
            
        ret = self.rknn.init_runtime()
        if ret != 0:
            logger.error(f"rknn.init_runtime failed with code {ret}")
            sys.exit(1)
        logger.info("Vision Encoder initialized successfully")

    def preprocess(self, img_bytes):
        try:
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None: raise ValueError("Failed to decode image bytes")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
            
            if width != height:
                size = max(width, height)
                square = np.full((size, size, 3), 127, dtype=np.uint8)
                x_off, y_off = (size - width) // 2, (size - height) // 2
                square[y_off:y_off+height, x_off:x_off+width] = img
                img = square
                
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = np.expand_dims(img, axis=0)
            return img
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            raise

    def encode(self, img_bytes):
        input_data = self.preprocess(img_bytes)
        outputs = self.rknn.inference(inputs=[input_data])
        embeddings = outputs[0].flatten().astype(np.float32)
        return embeddings

# ==========================================
# LLM Engine (Updated to match flask_server.py)
# ==========================================
class LLMEngine:
    def __init__(self, model_path, platform="rk3588"):
        logger.info(f"Initializing LLM Engine: {model_path}")
        if not os.path.exists(model_path):
            logger.error(f"LLM model not found at {model_path}")
            sys.exit(1)

        self.handle = RKLLM_Handle_t()
        self.q = queue.Queue()
        self.lock = threading.Lock()
        
        # --- Initialization Logic from flask_server.py ---
        rkllm_param = RKLLMParam()
        rkllm_param.model_path = bytes(model_path, 'utf-8')

        rkllm_param.max_context_len = 4096
        rkllm_param.max_new_tokens = 512
        rkllm_param.skip_special_token = True
        rkllm_param.n_keep = -1
        rkllm_param.top_k = 1
        rkllm_param.top_p = 0.9
        rkllm_param.temperature = 0.1
        rkllm_param.repeat_penalty = 1.1
        rkllm_param.frequency_penalty = 0.0
        rkllm_param.presence_penalty = 0.0

        rkllm_param.mirostat = 0
        rkllm_param.mirostat_tau = 5.0
        rkllm_param.mirostat_eta = 0.1

        rkllm_param.is_async = False

        rkllm_param.img_start = VISION_START_TOKEN.encode('utf-8')
        rkllm_param.img_end = VISION_END_TOKEN.encode('utf-8')
        rkllm_param.img_content = VISION_PAD_TOKEN.encode('utf-8')

        rkllm_param.extend_param.base_domain_id = 0
        rkllm_param.extend_param.embed_flash = 1
        rkllm_param.extend_param.n_batch = 1
        rkllm_param.extend_param.use_cross_attn = 0
        rkllm_param.extend_param.enabled_cpus_num = 4
        
        if platform.lower() in ["rk3576", "rk3588"]:
            rkllm_param.extend_param.enabled_cpus_mask = (1 << 4)|(1 << 5)|(1 << 6)|(1 << 7)
        else:
            rkllm_param.extend_param.enabled_cpus_mask = (1 << 0)|(1 << 1)|(1 << 2)|(1 << 3)

        self.cb_func = CALLBACK_TYPE(self._callback)

        # Explicitly define argtypes and restype (Crucial for stability)
        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), CALLBACK_TYPE]
        self.rkllm_init.restype = ctypes.c_int
        
        logger.info("Calling rkllm_init...")
        ret = self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), self.cb_func)
        if ret != 0:
            logger.critical(f"rkllm_init failed with return code: {ret}")
            sys.exit(1)
        
        # Setup Run function
        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.rkllm_run.restype = ctypes.c_int
        
        # Setup Infer Params
        self.rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        self.rkllm_infer_params.keep_history = 0
        
        logger.info("LLM Engine initialized successfully")

    def _callback(self, result, userdata, state):
        try:
            if state == LLMCallState.RKLLM_RUN_NORMAL:
                text = result.contents.text.decode('utf-8', errors='ignore')
                self.q.put(text)
            elif state == LLMCallState.RKLLM_RUN_FINISH:
                self.q.put(None)
            elif state == LLMCallState.RKLLM_RUN_ERROR:
                logger.error("LLM reported runtime error")
                self.q.put(None)
        except Exception as e:
            logger.error(f"Callback exception: {e}")
        return 0

    def infer(self, prompt, image_embeds=None):
        with self.lock:
            logger.info(f"Starting inference. Prompt len: {len(prompt)}. Has Image: {image_embeds is not None}")
            
            inp = RKLLMInput()
            inp.role = b"user"
            inp.enable_thinking = False
            
            if image_embeds is not None:
                inp.input_type = RKLLMInputType.RKLLM_INPUT_MULTIMODAL
                mm = inp.input_data.multimodal_input
                mm.prompt = prompt.encode('utf-8')
                mm.image_embed = image_embeds.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                mm.n_image_tokens = IMG_TOKENS
                mm.n_image = 1
                mm.image_width = IMG_SIZE
                mm.image_height = IMG_SIZE
            else:
                inp.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
                inp.input_data.prompt_input = prompt.encode('utf-8')

            ret = self.rkllm_run(self.handle, ctypes.byref(inp), ctypes.byref(self.rkllm_infer_params), None)
            if ret != 0:
                logger.error(f"rkllm_run failed with code {ret}")
                yield f"[ERROR: rkllm_run code {ret}]"
                return

            while True:
                token = self.q.get()
                if token is None: 
                    break
                yield token

# ==========================================
# Flask Server
# ==========================================
app = Flask(__name__)

vision_model = None
llm_model = None

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.json
        logger.info(f"Received request from {request.remote_addr}")
        
        messages = data.get('messages', [])
        img_bytes = None
        
        # Convert OpenAI messages to HuggingFace format and extract images
        hf_messages = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if isinstance(content, list):
                # Multimodal message - extract text and image
                text_parts = []
                for item in content:
                    if item['type'] == 'text':
                        text_parts.append(item['text'])
                    elif item['type'] == 'image_url':
                        logger.info("Found image in request")
                        try:
                            url = item['image_url']['url']
                            if "base64," in url:
                                b64 = url.split("base64,")[-1]
                            else:
                                b64 = url
                            img_bytes = base64.b64decode(b64)
                        except Exception as e:
                            logger.error(f"Base64 decode failed: {e}")
                            return jsonify({"error": "Invalid image data"}), 400
                
                # Insert image placeholder before text for VLM
                text_content = " ".join(text_parts)
                if img_bytes:
                    text_content = f"{IMAGE_PLACEHOLDER}{text_content}"
                hf_messages.append({"role": role, "content": text_content})
            else:
                # Text-only message
                hf_messages.append({"role": role, "content": content})

        # Apply chat template to format messages properly
        try:
            prompt_text = tokenizer.apply_chat_template(
                hf_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            logger.info(f"Formatted prompt length: {len(prompt_text)}")
        except Exception as e:
            logger.error(f"Chat template application failed: {e}")
            return jsonify({"error": "Chat template error"}), 500
        
        # Encode image if present
        embeddings = None
        if img_bytes:
            try:
                embeddings = vision_model.encode(img_bytes)
            except Exception as e:
                logger.error(f"Vision encoding failed: {e}")
                return jsonify({"error": "Vision model failure"}), 500

        def generate():
            generated_text = ""
            for token in llm_model.infer(prompt_text, embeddings):
                generated_text += token
                chunk = {
                    "id": "chatcmpl-rk",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "rk-vlm",
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
            logger.info(f"Response complete. Total length: {len(generated_text)}")
            end_chunk = {"id": "chatcmpl-rk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(end_chunk)}\n\n[DONE]"

        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        logger.exception("Unhandled server error")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    try:
        # Initialize Tokenizer FIRST
        logger.info("--------------------------------")
        logger.info(f"Loading tokenizer from: {TOKENIZER_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH,
            trust_remote_code=True,
            use_fast=False
        )
        if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
            logger.critical("Tokenizer has no chat_template. Cannot proceed.")
            sys.exit(1)
        logger.info("Tokenizer loaded successfully")
        
        # Initialize LLM SECOND (Resource Heavy)
        logger.info("--------------------------------")
        logger.info(f"Loading LLM from: {LLM_MODEL_PATH}")
        llm_model = LLMEngine(LLM_MODEL_PATH, TARGET_PLATFORM)
        
        # Initialize Encoder THIRD
        logger.info("--------------------------------")
        logger.info(f"Loading Encoder from: {ENC_MODEL_PATH}")
        vision_model = VisionEncoder(ENC_MODEL_PATH)
        logger.info("--------------------------------")

    except Exception as e:
        logger.critical(f"Startup failed: {e}")
        sys.exit(1)

    logger.info(f"Starting Flask Server on {HOST_IP}:{PORT}")
    app.run(host=HOST_IP, port=PORT, threaded=True)