import os
import time
import requests
from PIL import Image as PILImage
from io import BytesIO
import base64
import json

# Set API key
api_key = os.environ.get("DEEPINFRA_TOKEN")

# Function to save the image from a URL to the local file system
def save_image(image_url, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    response = requests.get(image_url)
    img = PILImage.open(BytesIO(response.content))
    img.save(os.path.join(folder, filename))

# Function to save the image from a base64-encoded string
def save_image_from_base64(image_data, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    img = PILImage.open(BytesIO(image_data))
    img.save(os.path.join(folder, filename))

# Function to generate images for each prompt using DeepInfra API
def generate_images():
    models = ['sdxl-turbo', 'flux-1.1-pro']  # Using black-forest and sdxl-turbo models

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    for prompt_name, prompts in prompts_data.items():
        high_prompt, low_prompt = prompts  # Each prompt has a high and low variant

        for model in models:
            for level, prompt in zip(['high', 'low'], [high_prompt, low_prompt]):
                folder = os.path.join(parent_dir, "Images", model, f"{prompt_name.replace(' ', '_')}", level)
                
                for i in range(20):  # Generate 20 images for each prompt
                    try:
                        # Set endpoint based on model
                        if model == 'sdxl-turbo':
                            endpoint = f"https://api.deepinfra.com/v1/inference/stabilityai/{model}"
                        else:
                            endpoint = "https://api.deepinfra.com/v1/inference/black-forest-labs/FLUX-1.1-pro"

                        response = requests.post(
                            endpoint,
                            json={"prompt": prompt},
                            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                        )

                        print(f"API Response Status Code: {response.status_code}")
                        response_json = response.json()
                        print(f"Full API Response: {response_json}")

                        # Check response and save the image
                        if response.status_code == 200 and "images" in response_json:
                            for image_data_base64 in response_json["images"]:
                                image_data = base64.b64decode(image_data_base64.split(",")[1])
                                filename = f"image_{i+1}.png"
                                save_image_from_base64(image_data, folder, filename)
                                print(f"Saved image to {os.path.join(folder, filename)}")
                        elif response.status_code == 422:
                            print(f"Unprocessable Entity - Please check the request data. Response: {response_json}")
                        else:
                            print(f"Failed to generate image for prompt: {prompt}, Model: {model}, Status Code: {response.status_code}")

                    except Exception as e:
                        print(f"Error generating image for prompt: {prompt} - {e}")

                    # Add a short delay to avoid making requests too frequently
                    time.sleep(1)

# Load prompts from prompts_t2i.json
with open("prompts_t2i.json", "r") as file:
    prompts_data = json.load(file)

# Generate images
generate_images()
