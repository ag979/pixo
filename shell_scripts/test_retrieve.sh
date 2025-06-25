#!/bin/bash


curl -X POST "http://127.0.0.1:8000/search" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@assets/debug_guepard.jpg"