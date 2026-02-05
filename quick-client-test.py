"""
Quick test client using OpenAI Python library for RKNN-LLM server
"""
import base64
from openai import OpenAI

# --- Configuration ---
IMAGE_PATH = "1-crop.jpeg"
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