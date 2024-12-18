import os
from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
import uvicorn
from io import BytesIO
from fastapi.responses import StreamingResponse

# Load Stable Diffusion Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "CompVis/stable-diffusion-v1-4"

print("Loading Stable Diffusion Model...")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
pipe.to(device)

# FastAPI app
app = FastAPI()

# Request body
class PromptRequest(BaseModel):
    prompt: str

# Generate image endpoint
@app.post("/generate/")
async def generate_image(request: PromptRequest):
    prompt = request.prompt
    print(f"Generating image for: {prompt}")

    # Generate the image
    with torch.autocast("cuda" if device == "cuda" else "cpu"):
        image = pipe(prompt).images[0]

    # Save to a buffer
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")


# Run the app locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
