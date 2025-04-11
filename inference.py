from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import torch
import json
import os
import numpy as np

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
DB_PATH = "./memory.json"

def extract_embedding(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return outputs.squeeze().numpy()

def process_and_store_image(image_bytes, explanation):
    embedding = extract_embedding(image_bytes).tolist()
    memory = []
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'r') as f:
            memory = json.load(f)
    memory.append({"embedding": embedding, "explanation": explanation})
    with open(DB_PATH, 'w') as f:
        json.dump(memory, f)
    return embedding

def compare_image(image_bytes):
    embedding = extract_embedding(image_bytes)
    if not os.path.exists(DB_PATH):
        return {"message": "Nenhum ensinamento armazenado ainda."}

    with open(DB_PATH, 'r') as f:
        memory = json.load(f)

    similarities = []
    for item in memory:
        stored_vec = np.array(item["embedding"])
        sim = cosine_similarity(embedding, stored_vec)
        similarities.append((sim, item["explanation"]))

    similarities.sort(reverse=True, key=lambda x: x[0])
    return {"similarity": similarities[0][0], "matched_explanation": similarities[0][1]}

def cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))