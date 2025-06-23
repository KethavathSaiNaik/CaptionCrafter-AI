# backend/api_main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
from captioner import generate_caption

app = FastAPI(title="Image Caption Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend access
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(BytesIO(img_bytes))
    caption = generate_caption(image)
    return {"caption": caption}
