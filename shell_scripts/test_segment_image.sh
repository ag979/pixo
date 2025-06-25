#!/bin/bash


IMAGE_ID="734b86aa-3a3a-4805-add1-cee210a67a10"
curl -X POST http://127.0.0.1:8000/segment/${IMAGE_ID} \
     -H "Content-Type: application/json" \
     -d '{"mode": "quick"}'

