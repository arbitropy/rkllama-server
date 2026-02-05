import ctypes
import os
import sys
import threading
import queue
import json
import base64
import time
import yaml
from contextlib import asynccontextmanager

# Set LD_LIBRARY_PATH for local RKNN library before importing RKNNLite
LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
if os.path.exists(LIB_DIR):
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = f"{LIB_DIR}:{current_ld_path}" if current_ld_path else LIB_DIR

import cv2
import numpy as np
import logging
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
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
# Configuration Loading
# ==========================================
def load_config(config_path="config.yaml"):
    """Load YAML configuration file with validation"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.critical(f"Config file not found: {config_path}")
        logger.critical("Please create config.yaml from config.yaml.example")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.critical(f"Invalid YAML in config file: {e}")
        sys.exit(1)

# Load configuration
config = load_config()

# ==========================================
# Configuration
# ==========================================
# Extract configuration values
LLM_MODEL_PATH = config['model']['llm_path']
ENC_MODEL_PATH = config['model']['encoder_path']
TOKENIZER_PATH = config['model']['tokenizer_path']

# Model metadata
MODEL_ID = config['model']['id']

# Vision configuration
IMG_SIZE = config['vision']['image_size']
IMG_TOKENS = config['vision']['image_tokens']
VISION_START_TOKEN = config['vision']['start_token']
VISION_END_TOKEN = config['vision']['end_token']
VISION_PAD_TOKEN = config['vision']['pad_token']
IMAGE_PLACEHOLDER = config['vision']['placeholder']

# Library configuration
RKLLM_LIB_PATH = config['libraries']['rkllm_path']
RKNN_LIB_PATH = config['libraries']['rknn_path']

# Server configuration
TARGET_PLATFORM = config['platform']['type']
HOST_IP = config['server']['host']
PORT = config['server']['port']

# Generation defaults
GEN_CONFIG = config['generation']

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
        ("input_type", ctypes.c_int),
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
# LLM Engine
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
        
        # --- Initialization Logic ---
        # Initialize params from config defaults
        rkllm_param = RKLLMParam()
        rkllm_param.model_path = bytes(model_path, 'utf-8')

        rkllm_param.max_context_len = GEN_CONFIG['max_context_len']
        rkllm_param.max_new_tokens = GEN_CONFIG['max_new_tokens']
        rkllm_param.skip_special_token = GEN_CONFIG['skip_special_token']
        rkllm_param.n_keep = GEN_CONFIG['n_keep']
        rkllm_param.top_k = GEN_CONFIG['top_k']
        rkllm_param.top_p = GEN_CONFIG['top_p']
        rkllm_param.temperature = GEN_CONFIG['temperature']
        rkllm_param.repeat_penalty = GEN_CONFIG['repeat_penalty']
        rkllm_param.frequency_penalty = GEN_CONFIG['frequency_penalty']
        rkllm_param.presence_penalty = GEN_CONFIG['presence_penalty']

        rkllm_param.mirostat = GEN_CONFIG['mirostat']
        rkllm_param.mirostat_tau = GEN_CONFIG['mirostat_tau']
        rkllm_param.mirostat_eta = GEN_CONFIG['mirostat_eta']

        rkllm_param.is_async = config['advanced']['is_async']

        rkllm_param.img_start = VISION_START_TOKEN.encode('utf-8')
        rkllm_param.img_end = VISION_END_TOKEN.encode('utf-8')
        rkllm_param.img_content = VISION_PAD_TOKEN.encode('utf-8')

        # Extended parameters from config
        rkllm_param.extend_param.base_domain_id = config['advanced']['base_domain_id']
        rkllm_param.extend_param.embed_flash = config['advanced']['embed_flash']
        rkllm_param.extend_param.n_batch = config['advanced']['n_batch']
        rkllm_param.extend_param.use_cross_attn = config['advanced']['use_cross_attn']
        rkllm_param.extend_param.enabled_cpus_num = 4
        
        # CPU core configuration from config
        if config['platform']['cpu_mode'] == 'auto':
            # Auto-select based on platform
            if platform.lower() in ["rk3576", "rk3588"]:
                rkllm_param.extend_param.enabled_cpus_mask = (1 << 4)|(1 << 5)|(1 << 6)|(1 << 7)
            else:
                rkllm_param.extend_param.enabled_cpus_mask = (1 << 0)|(1 << 1)|(1 << 2)|(1 << 3)
        else:
            # Use manual mask from config
            rkllm_param.extend_param.enabled_cpus_mask = config['platform']['cpu_mask_manual']

        self.cb_func = CALLBACK_TYPE(self._callback)

        # Explicitly define argtypes and restype
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
        
        # Setup Infer Params from config
        self.rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        self.rkllm_infer_params.keep_history = 0
        
        # Store default generation params (can be overridden per request)
        self.default_params = GEN_CONFIG.copy()
        
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

    def infer(self, prompt, image_embeds=None, n_images=0):
        with self.lock:
            logger.info(f"Starting inference. Prompt len: {len(prompt)}. Images: {n_images}")
            
            inp = RKLLMInput()
            inp.role = b"user"
            inp.enable_thinking = False
            
            if image_embeds is not None and n_images > 0:
                inp.input_type = RKLLMInputType.RKLLM_INPUT_MULTIMODAL
                mm = inp.input_data.multimodal_input
                mm.prompt = prompt.encode('utf-8')
                mm.image_embed = image_embeds.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                mm.n_image_tokens = IMG_TOKENS
                mm.n_image = n_images  # Dynamic image count
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
# FastAPI Server
# ==========================================

# Global model placeholders
vision_model = None
llm_model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle model loading on startup
    and cleanup on shutdown.
    """
    global vision_model, llm_model, tokenizer
    try:
        # Initialize Tokenizer
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
        
        # Initialize LLM
        logger.info("--------------------------------")
        logger.info(f"Loading LLM from: {LLM_MODEL_PATH}")
        llm_model = LLMEngine(LLM_MODEL_PATH, TARGET_PLATFORM)
        
        # Initialize Encoder
        logger.info("--------------------------------")
        logger.info(f"Loading Encoder from: {ENC_MODEL_PATH}")
        vision_model = VisionEncoder(ENC_MODEL_PATH)
        logger.info("--------------------------------")
        
        yield # Server runs here
        
    except Exception as e:
        logger.critical(f"Startup failed: {e}")
        sys.exit(1)
    finally:
        logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

@app.get('/v1/models')
async def list_models():
    """OpenAI-compatible models list endpoint. Returns configured model, accepts any model name in requests."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time())
            }
        ]
    }

@app.get('/v1/models/{model}')
async def retrieve_model(model: str):
    """OpenAI-compatible model retrieval endpoint. Accepts any model name."""
    # Accept any model name, return configured model
    return {
        "id": MODEL_ID,
        "object": "model",
        "created": int(time.time())
    }

@app.post('/v1/chat/completions')
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint with streaming support"""
    try:
        data = await request.json()
        logger.info(f"Received request from {request.client.host}")
        
        # Parse OpenAI request parameters 
        messages = data.get('messages', [])
        # NOTE: Generation params and model name params aren't used from the request, they are hardcoded in config.
        
        image_list = []  # Collect all images in order
        
        # Convert OpenAI messages to HuggingFace format and extract images
        hf_messages = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if isinstance(content, list):
                # Multimodal message - preserve native content structure for tokenizer
                processed_content = []
                for item in content:
                    if item['type'] == 'text':
                        processed_content.append({"type": "text", "text": item['text']})
                    elif item['type'] == 'image_url':
                        try:
                            url = item['image_url']['url']
                            b64 = url.split("base64,")[-1] if "base64," in url else url
                            img_bytes = base64.b64decode(b64)
                            image_list.append(img_bytes)
                            # Add image marker for tokenizer
                            processed_content.append({"type": "image"})
                            logger.info(f"Extracted image {len(image_list)}")
                        except Exception as e:
                            logger.error(f"Base64 decode failed: {e}")
                            raise HTTPException(status_code=400, detail="Invalid image data")
                
                hf_messages.append({"role": role, "content": processed_content})
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
            logger.info(f"Formatted prompt length: {len(prompt_text)}, images: {len(image_list)}")
        except Exception as e:
            logger.error(f"Chat template application failed: {e}")
            raise HTTPException(status_code=500, detail="Chat template error")
        
        # Encode and concatenate all image embeddings
        embeddings = None
        n_images = len(image_list)
        if n_images > 0:
            try:
                embed_list = [vision_model.encode(img) for img in image_list]
                # Concatenate: [img1_tokens..., img2_tokens..., imgN_tokens...]
                embeddings = np.concatenate(embed_list, axis=0)
                logger.info(f"Encoded {n_images} images, total embed shape: {embeddings.shape}")
            except Exception as e:
                logger.error(f"Vision encoding failed: {e}")
                raise HTTPException(status_code=500, detail="Vision model failure")

        # Get stream parameter from request (default to False)
        stream = data.get('stream', False)
        
        def generate_stream():
            """Generate OpenAI-compatible streaming response"""
            generated_text = ""
            chunk_id = f"chatcmpl-{int(time.time())}"
            created_time = int(time.time())
            
            # Stream tokens from LLM
            for token in llm_model.infer(prompt_text, embeddings, n_images):
                generated_text += token
                chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Final chunk with finish reason and usage stats
            logger.info(f"Generation complete. Output tokens (approx): {len(generated_text.split())}")
            
            # Approximate token counts (simple word-based estimation)
            prompt_tokens = len(prompt_text.split()) + (n_images * IMG_TOKENS)
            completion_tokens = len(generated_text.split())
            
            end_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
            yield f"data: {json.dumps(end_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        if stream:
            return StreamingResponse(generate_stream(), media_type='text/event-stream')
        else:
            # Non-streaming mode: collect all tokens and return as single response
            generated_text = ""
            chunk_id = f"chatcmpl-{int(time.time())}"
            created_time = int(time.time())
            
            for token in llm_model.infer(prompt_text, embeddings, n_images):
                generated_text += token
            
            logger.info(f"Generation complete. Output tokens (approx): {len(generated_text.split())}")
            
            # Approximate token counts
            prompt_tokens = len(prompt_text.split()) + (n_images * IMG_TOKENS)
            completion_tokens = len(generated_text.split())
            
            return {
                "id": chunk_id,
                "object": "chat.completion",
                "created": created_time,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled server error")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    logger.info(f"Starting FastAPI Server on {HOST_IP}:{PORT}")
    uvicorn.run(app, host=HOST_IP, port=PORT, log_config=None)