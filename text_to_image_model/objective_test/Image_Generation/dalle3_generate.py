import os
import time
import requests
from openai import OpenAI
from PIL import Image as PILImage
from io import BytesIO
import json

# Set API key
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Function to save the image to the local file system
def save_image(image_url, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    response = requests.get(image_url)
    img = PILImage.open(BytesIO(response.content))
    img.save(os.path.join(folder, filename))

# Load definitions from definition.json
with open("definition.json", "r") as def_file:
    definitions = json.load(def_file)

# Load prompts from prompts_obj.json
with open("prompts_obj.json", "r") as prompts_file:
    prompts_data = json.load(prompts_file)

# Main function to generate images for each prompt
def generate_images():
    model_name = 'dalle3'  # Set the model name here

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    for prompt_name, prompt_variants in prompts_data.items():
        # Get the definition for the current category
        definition = definitions.get(prompt_name, "")

        # Modify each prompt by adding the definition
        high_prompt = f"{prompt_name} is {definition} {prompt_variants[0]}"
        low_prompt = f"{prompt_name} is {definition} {prompt_variants[1]}"

        # Loop over 'high' and 'low' levels
        for level, prompt in zip(['high', 'low'], [high_prompt, low_prompt]):
            folder = os.path.join(parent_dir, "Images", model_name, f"{prompt_name.replace(' ', '_')}", level)
            
            for i in range(20):  # Generate 20 images for each prompt
                try:
                    # Print the prompt being used
                    print(f"Using prompt: {prompt}")
                    
                    response = client.images.generate(
                        model="dall-e-3",
                        prompt=prompt,
                        size="1024x1024",
                        quality="standard",
                        n=1,  # Generate 1 image per request
                    )

                    if response.data:
                        image_url = response.data[0].url
                        filename = f"image_{i + 1}.png"
                        save_image(image_url, folder, filename)
                        print(f"Saved image to {os.path.join(folder, filename)}")
                    else:
                        print(f"Failed to generate image for prompt: {prompt}")

                except Exception as e:
                    print(f"Error generating image for prompt: {prompt} - {e}")

                # Add a short delay to avoid making requests too frequently
                time.sleep(1)

# Generate images for all prompts
generate_images()
