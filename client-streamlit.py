import streamlit as st
import requests
import base64
import json
import logging

# ==========================================
# Logging Configuration
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# Configuration & UI Setup
# ==========================================
st.set_page_config(page_title="Rockchip VLM Chat", page_icon="🤖")
st.title("🤖 Rockchip VLM Chatbot")

# Sidebar for Server Configuration
with st.sidebar:
    st.header("Server Settings")
    server_ip = st.text_input("Server IP", value="localhost")
    server_port = st.text_input("Server Port", value="8080")
    api_url = f"http://{server_ip}:{server_port}/v1/chat/completions"
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# Helper Functions
# ==========================================
def encode_image(image_file):
    """Convert uploaded file to base64 string."""
    try:
        return base64.b64encode(image_file.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Image encoding failed: {e}")
        return None

# ==========================================
# Chat Display
# ==========================================
# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Handle multimodal content (list) vs text content (str)
        if isinstance(message["content"], list):
            for item in message["content"]:
                if item["type"] == "text":
                    st.markdown(item["text"])
                elif item["type"] == "image_url":
                    # Display a small preview of the image in history
                    st.image(item["image_url"]["url"], caption="Uploaded Image", width=300)
        else:
            st.markdown(message["content"])

# ==========================================
# User Input & API Interaction
# ==========================================
# Image Uploader
uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])

# Chat Input
if prompt := st.chat_input("Ask me something about the image..."):
    
    # 1. Prepare Content Block
    user_content = []
    
    # Add Text
    user_content.append({"type": "text", "text": prompt})
    
    # Add Image if uploaded
    if uploaded_file:
        base64_image = encode_image(uploaded_file)
        if base64_image:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
    
    # 2. Append User Message to State & Display
    st.session_state.messages.append({"role": "user", "content": user_content})
    with st.chat_message("user"):
        st.markdown(prompt)
        if uploaded_file:
            st.image(uploaded_file, width=300)

    # 3. Call VLM Server (Streaming)
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            payload = {
                "model": "rockchip-vlm",
                "messages": st.session_state.messages,
                "stream": True
            }
            
            logger.info(f"Sending request to {api_url}")
            response = requests.post(api_url, json=payload, stream=True, timeout=60)
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        # Decode "data: {...}" chunks
                        line_text = line.decode("utf-8")
                        if line_text.startswith("data: "):
                            data_str = line_text[6:]
                            if data_str == "[DONE]":
                                break
                            
                            try:
                                chunk = json.loads(data_str)
                                content = chunk["choices"][0].get("delta", {}).get("content", "")
                                full_response += content
                                response_placeholder.markdown(full_response + "▌")
                            except json.JSONDecodeError:
                                continue
                
                response_placeholder.markdown(full_response)
                # 4. Store Assistant Response
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                st.error(f"Error: Server returned status {response.status_code}")
                logger.error(f"API Error: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the VLM server. Check IP/Port.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.exception("Inference loop failed")