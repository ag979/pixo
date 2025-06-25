# Image Search Engine Use-Case
This repository contains an image search engine fine-tuned on the [my-first-project-klagq](https://universe.roboflow.com/azza-te8hj/my-first-project-klagq) dataset for real-time retrieval.

The search engine can perform the following operations:

 - Upload images to local storage and record state in database
 - Generate segmentation masks of main image components (using [YOLOE](https://docs.ultralytics.com/models/yoloe/))
 - Generate embeddings from segmentation masks and store them in the [qdrant](https://qdrant.tech/) vector database for indexing and similarity search 
 - Perform vector search using qdrant

## Installation
The engine may be started locally by running:
```
docker compose up -d --build
```

## API Usage Guide
The API supports 3 functionalities: upload image; segment and compute segment embeddings of uploaded image, retrieve similar image.
### Upload image
```
curl -X POST "http://127.0.0.1:8000/upload" -F "file=@assets/demo_human.jpg"
```
### Segment and compute segment embeddings
```
IMAGE_ID="734b86aa-3a3a-4805-add1-cee210a67a10" # use retrieved image ID
curl -X POST http://127.0.0.1:8000/segment/${IMAGE_ID} \
     -H "Content-Type: application/json" \
     -d '{"mode": "quick"}'
```
### Retrieve similar image
```
curl -X POST "http://127.0.0.1:8000/search" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@assets/debug_guepard.jpg"
```


## End-to-end Testing
To run all operations supported by the engine run the following:
```
pytest --log-cli-level=INFO tests/integration/test.py
```



## Architecture and Approach

### System Overview
Image Upload → Segmentation (YOLOE) → Embedding Generation (Few-shot fine-tuned ResNet18) → Vector Search (Qdrant)

### Segmentation Model Selection
For image segmentation [YOLOE](https://docs.ultralytics.com/models/yoloe/) was used. This model provides open-vocab detection and segmentation capabilities in real-time, which is a good choice for extracting most meaningful image components


### Embeddings Model Selection
Embedding model chosen was a [resnet18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) backbone trained on [ImageNet](https://www.image-net.org/)

### Few-shot fine-tuning strategy

The embeddings model has been fine-tuned using a Prototypical Network methodology via the **train_few_shot.ipynb** notebook implemented using the [easy-few-shot-learning](https://github.com/sicara/easy-few-shot-learning) framework. Fine-tuning has increased accuracy on a refence eval from 65.40% to 88.20%.

**Pre-trained**

![](assets/model_pre_trained.png)


**Few-shot fine-tuned**

![](assets/model_fine_tuned.png)

## API Documentation & Testing
The FastAPI application provides comprehensive interactive API documentation.

- [Swagger UI](http://127.0.0.1:8000/docs): this allows testing all endpoints directly in the browser with file uploads, JSON responses, and results
- [ReDoc](http://127.0.0.1:8000/redoc): API documentation
- [OpenAPI Schema](http://127.0.0.1:8000/openapi.json): machine-readable API specification


## Monitoring & Logging

- [Prometheus Metrics](http://localhost:9090): scrapes metrics from FastAPI application **/metrics** endpoint
- [Grafana Dashboard](http://localhost:3000): visualization platform connected to Prometheus can be used to view relevant metrics ([default username and password](https://signoz.io/guides/what-is-the-default-username-and-password-for-grafana-login-page/#grafanas-default-username-and-password))