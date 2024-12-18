from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from fastapi.responses import StreamingResponse
from io import BytesIO

app = FastAPI()

# Replace this with your Modal API endpoint
MODAL_API_URL = "https://your-modal-endpoint/modal/generate"

class PromptRequest(BaseModel):
    prompt: str

@app.post("/api/generate-image/")
async def generate_image(request: PromptRequest):
    # Forward the request to your Modal API
    try:
        response = requests.post(MODAL_API_URL, json={"prompt": request.prompt})
        if response.status_code == 200:
            # Return the image directly as a stream
            image_data = BytesIO(response.content)
            return StreamingResponse(image_data, media_type="image/png")
        else:
            raise HTTPException(status_code=500, detail="Image generation failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
