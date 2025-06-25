import logging
import os
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # or DEBUG for more details
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

def log_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        logger.info(f"⏱️  {func.__name__} completed in {duration:.3f}s ")
        return result
    return wrapper


@log_time
def upload_image(client, base_url, image_path):
    assert os.path.exists(image_path), "Image file missing"
    with open(image_path, "rb") as f:
        response = client.post(f"{base_url}/upload", files={"file": f})
    assert response.status_code == 200
    image_id = response.json().get("id")
    logger.info(f"✅ Image uploaded successfully http://localhost:8000/uploads/{image_id}.jpg")
    return image_id

@log_time
def segment_image(client, base_url, image_id):
    segment_url = f"{base_url}/segment/{image_id}"
    response = client.post(segment_url, json={"mode": "quick"})
    assert response.status_code == 200
    segment_ids = response.json().get("segment_ids", [])
    logger.info(f"✅ {len(segment_ids)} segments were generated ")

@log_time
def retrieve_image(client, base_url, image_path):
    logger.info(f"ℹ️  Query image: {image_path}")
    with open(image_path, "rb") as f:
        files = {"file": ("demo_guepard.jpg", f, "image/jpeg")}
        response = client.post(f"{base_url}/search", files=files)
    assert response.status_code == 200
    results = response.json().get("results", [])
    best_match = results[0]["payload"]["parent_image_id"]
    logger.info(f"✅ Retrieved image: http://localhost:8000/uploads/{best_match}.jpg ")
    assert len(results) > 0, "No similar images returned"
