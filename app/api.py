import datetime
import io
import os
import uuid
from contextlib import asynccontextmanager
from torchvision import transforms

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from torch import nn
from torchvision.models import resnet18
from ultralytics import YOLOE

from app.config import (
    IMAGES_COLLECTION,
    SEGMENTS_COLLECTION,
    UPLOAD_DIR,
    VECTOR_DIM_IMAGE,
)
from app.embed import encode_image, image_to_embeddings
from app.model import PrototypicalNetworks


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    app.state.segmenter = YOLOE("yoloe-11l-seg-pf.pt")
    # app.state.encoder, app.state.preprocess = clip.load("ViT-B/32", device=device)

    convolutional_network = resnet18(pretrained=True)
    convolutional_network.fc = nn.Flatten()
    encoder = PrototypicalNetworks(convolutional_network).to(device)
    encoder.load_state_dict(
        torch.load(
            "/app/weights/model_few_shot_fine_tuned_resnet18.pth", map_location=device
        )
    )
    app.state.encoder = encoder
    app.state.preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    yield


app = FastAPI(lifespan=lifespan)
instrumentator = Instrumentator().instrument(app).expose(app)

os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

client = QdrantClient(host="qdrant", port=6333)


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    file_extension = os.path.splitext(file.filename)[1]
    image_id = str(uuid.uuid4())
    unique_filename = f"{image_id}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    metadata = {
        "filename": unique_filename,
        "upload_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    zero_vector = [0.0] * VECTOR_DIM_IMAGE
    point = PointStruct(id=image_id, vector=zero_vector, payload=metadata)
    client.upsert(collection_name=IMAGES_COLLECTION, points=[point])

    return JSONResponse({"message": "Upload successful", "id": image_id})


@app.post("/segment/{image_id}")
async def segment_image(image_id: str):
    image_path = os.path.join(UPLOAD_DIR, f"{image_id}.jpg")
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    embeddings = image_to_embeddings(
        image_path, app.state.segmenter, [app.state.encoder, app.state.preprocess]
    )

    points = []
    for embedding in embeddings:
        segment_id = str(uuid.uuid4())
        payload = {"parent_image_id": image_id}
        point = PointStruct(id=segment_id, vector=embedding, payload=payload)
        points.append(point)

    client.upsert(collection_name=SEGMENTS_COLLECTION, points=points)

    return {
        "message": "Segmentation completed",
        "num_segments": len(points),
        "segment_ids": [p.id for p in points],
    }


@app.post("/search")
async def search_similar_segments(file: UploadFile = File(...), top_k: int = 1):
    content = await file.read()
    image = io.BytesIO(content)
    embedding = encode_image(image, [app.state.encoder, app.state.preprocess])

    search_results = client.search(
        collection_name="segments", query_vector=embedding, limit=top_k
    )

    results = [
        {"id": hit.id, "score": hit.score, "payload": hit.payload}
        for hit in search_results
    ]

    return {"results": results}
