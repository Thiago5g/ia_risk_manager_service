from fastapi import FastAPI, UploadFile, File, Form
from inference import process_and_store_image, compare_image
import uvicorn
import os

app = FastAPI()

@app.post("/teach")
async def teach(file: UploadFile = File(...), explanation: str = Form(...)):
    contents = await file.read()
    result = process_and_store_image(contents, explanation)
    return {"message": "Ensinamento armazenado com sucesso", "embedding": result}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    result = compare_image(contents)
    return result

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))  # Render injeta essa variável dinamicamente
    uvicorn.run("main:app", host="0.0.0.0", port=port)
