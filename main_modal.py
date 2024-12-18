import modal

# Modal setup
app = modal.App("image-generation-app")  # Updated to use App instead of Stub
gpu_image = modal.Image.debian_slim().pip_install(
    "torch", "diffusers", "transformers", "fastapi", "uvicorn"
)

@app.function(image=gpu_image, gpu="A10G")  # Attach the function to the app
def generate_image(prompt: str):
    from diffusers import StableDiffusionPipeline
    import torch
    from io import BytesIO

    device = "cuda"
    model_id = "CompVis/stable-diffusion-v1-4"

    # Load model
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to(device)

    # Generate image
    with torch.autocast("cuda"):
        image = pipe(prompt).images[0]

    # Return image as bytes
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

@app.local_entrypoint()
def main():
    print("Enter a prompt:")
    prompt = input()
    image_data = generate_image.remote(prompt)
    with open("output.png", "wb") as f:
        f.write(image_data)
    print("Image saved to output.png")

