import base64
import requests
import json
import sys

# --- Configuration ---
SERVER_URL = "http://localhost:8080/v1/chat/completions"
IMAGE_PATH = "1-crop.jpeg"  #
PROMPT = """
You are an expert food quality analyst. Your task is to assess the amount of food in a cropped image of a food tray.

### INSTRUCTIONS:
1. **Analyze Content:** Identify the type of food present. Is it a solid (e.g., rice, chicken), a liquid (e.g., soup, sauce), or a mix?
2. **Estimate Volume:** Compare the current amount of food to the total capacity of the tray compartment.
3. **Determine State:** Select the most accurate state from the definitions below:
   - **empty:** 0% food; the tray surface is completely visible.
   - **low:** Less than 25% full; only a small portion or scrap remains.
   - **medium:** 25% to 75% full; clearly used but contains a significant portion.
   - **full:** More than 75% full; the tray appears nearly or completely filled.
   - **closed:** A lid or cover is on the tray, making it impossible to see the food.

### OUTPUT FORMAT:
<reasoning>
- Food Type: [Solid/Liquid/Mixed]
- Observations: [Briefly describe the food and how much of the tray bottom is visible]
- Logic: [Explain why the food level matches the selected category]
</reasoning>

<tray_state>
[empty/low/medium/full/closed]
</tray_state>
"""
def test_vlm():
    # 1. Encode the image to Base64
    try:
        with open(IMAGE_PATH, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: {IMAGE_PATH} not found.")
        return

    # 2. Prepare the OpenAI-compatible payload
    payload = {
        "model": "rk-vlm",
        "messages": [
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
        "stream": True
    }

    # 3. Send the request and process the stream
    print(f"\nPrompt: {PROMPT}")
    print("Response: ", end="", flush=True)

    try:
        response = requests.post(SERVER_URL, json=payload, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                # Decode the line from bytes to string
                line_str = line.decode('utf-8')

                # Handle the [DONE] signal
                if line_str == "data: [DONE]":
                    break

                # Remove the "data: " prefix
                if line_str.startswith("data: "):
                    content_json = line_str[6:]
                    try:
                        data = json.loads(content_json)
                        # Extract the text content from the delta
                        content = data['choices'][0]['delta'].get('content', '')
                        
                        # Print the content word by word
                        print(content, end='', flush=True)
                    except json.JSONDecodeError:
                        continue

    except requests.exceptions.RequestException as e:
        print(f"\nConnection Error: {e}")

    print("\n\n[Finished]")

if __name__ == "__main__":
    test_vlm()