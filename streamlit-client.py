import streamlit as st
import requests
import json
import base64
import logging

# ==========================================
# Configuration & Setup
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8080/v1/chat/completions"

st.set_page_config(page_title="RKNN Multimodal Chat", layout="centered")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# Sidebar: Settings & Controls
# ==========================================
with st.sidebar:
    st.header("Configuration")
    
    # System Prompt Input
    system_prompt = st.text_area(
        "System Prompt", 
        value="You are a helpful multimodal assistant.",
        help="Instructions for how the model should behave."
    )
    
    # Clear Chat Button
    if st.button("Clear Chat", type="primary"):
        st.session_state.messages = []
        st.rerun()

st.title("RKNN Multimodal Chatbot")

# ==========================================
# Helper Functions
# ==========================================
def encode_image(uploaded_file):
    """Encodes a file to base64 for API transmission."""
    try:
        bytes_data = uploaded_file.getvalue()
        b64_str = base64.b64encode(bytes_data).decode('utf-8')
        return f"data:{uploaded_file.type};base64,{b64_str}"
    except Exception as e:
        logger.error(f"Image encoding error: {e}")
        return None

# ==========================================
# Main Chat Interface
# ==========================================

# 1. Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], str):
            st.markdown(msg["content"])
        elif isinstance(msg["content"], list):
            for item in msg["content"]:
                if item["type"] == "text":
                    st.markdown(item["text"])
                elif item["type"] == "image_url":
                    st.image(item["image_url"]["url"], width=300)

# 2. Handle Input (Text + Multiple Images)
# accept_file=True enables the attachment button. 
# The widget returns a list of files in prompt.files if multiple are selected.
if prompt := st.chat_input("Type a message...", accept_file="multiple", file_type=["png", "jpg", "jpeg"]):
    
    # Construct User Payload
    user_content = []
    
    # Add Text
    if prompt.text:
        user_content.append({"type": "text", "text": prompt.text})
    
    # Add Multiple Images
    if prompt.files:
        for file in prompt.files:
            b64_img = encode_image(file)
            if b64_img:
                user_content.append({
                    "type": "image_url", 
                    "image_url": {"url": b64_img}
                })
    
    if not user_content:
        st.warning("Please provide text or an image.")
        st.stop()

    # Append to local history
    st.session_state.messages.append({"role": "user", "content": user_content})
    
    # Render immediately
    with st.chat_message("user"):
        for item in user_content:
            if item["type"] == "text":
                st.markdown(item["text"])
            elif item["type"] == "image_url":
                st.image(item["image_url"]["url"], width=300)

    # 3. Stream Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # Prepare Messages: Prepend System Prompt + History
            api_messages = []
            if system_prompt.strip():
                api_messages.append({"role": "system", "content": system_prompt})
            api_messages.extend(st.session_state.messages)

            payload = {
                "messages": api_messages,
                "stream": True,
                "model": "rk-vlm"
            }
            
            with requests.post(API_URL, json=payload, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if decoded.startswith("data: "):
                            data_str = decoded[6:]
                            if data_str.strip() == "[DONE]": break
                            try:
                                chunk = json.loads(data_str)["choices"][0].get("delta", {}).get("content", "")
                                full_response += chunk
                                response_placeholder.markdown(full_response + "▌")
                            except: continue
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Server Error: {e}")
            logger.error(f"Inference failed: {e}")