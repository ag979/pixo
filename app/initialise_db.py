from app.config import (
    IMAGES_COLLECTION,
    IMAGES_VECTOR_DIM,
    SEGMENTS_COLLECTION,
    SEGMENTS_VECTOR_DIM,
)
from qdrant_client import QdrantClient, models

client = QdrantClient(host="localhost", port=6333)

# params modified as per https://qdrant.tech/documentation/guides/optimize/#3-high-precision-with-high-speed-search
client.recreate_collection(
    collection_name=IMAGES_COLLECTION,
    vectors_config=models.VectorParams(size=IMAGES_VECTOR_DIM, distance=models.Distance.COSINE),
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            always_ram=True,
        ),
    ),
)
print(f"Collection '{IMAGES_COLLECTION}' created successfully")


client.recreate_collection(
    collection_name=SEGMENTS_COLLECTION,
    vectors_config=models.VectorParams(size=SEGMENTS_VECTOR_DIM, distance=models.Distance.COSINE),
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            always_ram=True,
        ),
    ),
)
print(f"Collection '{SEGMENTS_COLLECTION}' created successfully")