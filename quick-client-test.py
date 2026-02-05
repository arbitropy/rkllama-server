"""
Quick test client using OpenAI Python library for RKNN-LLM server
"""
import base64
from openai import OpenAI

# --- Configuration ---
IMAGE_PATH = "demo.jpeg"
PROMPT = """Describe the image."""

def test_vlm():
    # Initialize OpenAI client with custom base URL
    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key="dummy"  # API key not required but library needs it
    )
    
    # Encode the image to Base64
    try:
        with open(IMAGE_PATH, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: {IMAGE_PATH} not found.")
        return

    print(f"\nPrompt: {PROMPT}")
    print("Response: ", end="", flush=True)

    try:
        # Create chat completion using OpenAI library
        response = client.chat.completions.create(
            model="internvl3_5-2b",  # Can be any name, server accepts all
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                        },
                        {"type": "text", "text": PROMPT}
                    ]
                }
            ],
            stream=True
        )

        # Process streaming response
        for chunk in response:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end='', flush=True)

    except Exception as e:
        print(f"\nError: {e}")

    print("\n\n[Finished]")

if __name__ == "__main__":
    test_vlm()