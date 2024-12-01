#!/bin/bash
docker rm -f garagedoor 2>/dev/null || true
docker run -d \
  --name garagedoor \
  --restart always \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  --memory=20g \
  garagedoor
