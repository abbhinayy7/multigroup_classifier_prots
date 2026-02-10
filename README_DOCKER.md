# Docker Setup for Multigroup Classifier

This document describes how to build and run the multigroup classifier in a Docker container for reproducible results across different environments.

## Prerequisites

- Docker Engine 20.10+ or Docker Desktop
- At least 4 GB of available disk space
- 2-4 CPU cores and 4-8 GB RAM recommended

## Building the Image

Build the Docker image using the provided Dockerfile:

```bash
docker build -f Dockerfile.multigroup -t multigroup_classifier:latest .
```

**Note:** The build may take 5-10 minutes on first run due to pip dependency installation. Subsequent builds will use cached layers.

### Image Size

The final image is approximately **2-3 GB** due to Python 3.11, scientific libraries (scikit-learn, XGBoost, pandas, numpy), and build tools.

## Running the Container

### 1. Run Default Test (Recommended)

Executes the binary vs multigroup comparison test automatically:

```bash
docker run --rm -v "$(pwd)/output:/app/output" \
  --memory=8g --cpus=4 \
  multigroup_classifier:latest
```

**Output:** Results written to `./output` directory on your host machine.

### 2. Interactive Shell

Start a bash shell for manual exploration:

```bash
docker run --rm -it -v "$(pwd):/app" \
  --memory=8g --cpus=4 \
  multigroup_classifier:latest /bin/bash
```

### 3. Run Custom Python Script

Execute any Python script in the project:

```bash
docker run --rm -v "$(pwd):/app" \
  --memory=8g --cpus=4 \
  multigroup_classifier:latest python py_scripts/train.py --help
```

## Volume Mounting

| Mount Type | Host Path | Container Path | Purpose |
|-----------|-----------|-----------------|---------|
| `-v $(pwd):/app` | Current directory | `/app` | Full project access |
| `-v $(pwd)/output:/app/output` | `./output` | `/app/output` | Store results on host |
| `-v $(pwd)/data:/app/data` | `./data` | `/app/data` | Input data location |

## Resource Limits

Adjust for your system:

```bash
# Conservative (laptop with 8 GB RAM, 2 cores)
--memory=4g --cpus=2

# Moderate (workstation with 16 GB RAM, 4 cores)
--memory=8g --cpus=4

# High-performance (server with 32+ GB RAM, 8+ cores)
--memory=16g --cpus=8
```

## Entrypoint Behavior

The container's entrypoint script (`entrypoint.sh`) works as follows:

- **No arguments:** Runs `test_binary_vs_multigroup.py` by default.
- **With arguments:** Executes the provided command.

Examples:

```bash
# Default: run test
docker run multigroup_classifier:latest

# Python script
docker run multigroup_classifier:latest python py_scripts/evaluate.py

# Shell command
docker run -it multigroup_classifier:latest /bin/bash
```

## Reproducibility Features

1. **Pinned Python version:** Python 3.11 (specified in Dockerfile)
2. **Locked dependencies:** `requirements.txt` with exact versions
3. **Non-root user:** Runs as `appuser` for security
4. **Deterministic layers:** Multi-stage caching for faster rebuilds
5. **.dockerignore:** Excludes cache/output files from image

## Common Issues

### "Out of Memory" During Build or Run

Increase `--memory` limit:
```bash
docker run --rm --memory=16g multigroup_classifier:latest
```

### "Permission Denied" on Output Files

Ensure host output directory is writable:
```bash
chmod 777 ./output
```

### Image Not Found

Rebuild the image:
```bash
docker build -f Dockerfile.multigroup -t multigroup_classifier:latest .
```

### Slow pip Install During Build

Docker will use cached layers if the `requirements.txt` hasn't changed. To force a fresh install:
```bash
docker build --no-cache -f Dockerfile.multigroup -t multigroup_classifier:latest .
```

## Performance Tips

1. **Use named volumes** for repeated runs (avoids reformatting output):
   ```bash
   docker volume create multigroup_output
   docker run -v multigroup_output:/app/output multigroup_classifier:latest
   ```

2. **Build variants** for different use cases:
   ```bash
   # GPU variant (if CUDA available)
   docker build -f Dockerfile.multigroup.gpu -t multigroup_classifier:gpu .
   ```

3. **Monitor resource usage** during execution:
   ```bash
   docker stats --no-stream
   ```

## Data Input

Place your input data in a `data/` directory on the host:

```bash
mkdir -p ./data
cp your_data.tsv ./data/
docker run -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  multigroup_classifier:latest python py_scripts/train.py --data /app/data/your_data.tsv
```

## Cleaning Up

Remove unused images and containers:

```bash
# Remove image
docker rmi multigroup_classifier:latest

# Remove all stopped containers
docker container prune -f

# Remove all dangling images
docker image prune -f
```

## Advanced: Multi-Stage Build

To reduce final image size, we could use a multi-stage build. Contact maintainers if needed.

## Support

For issues or questions:
1. Check `docker logs <container-id>`
2. Run with `-v` flag for verbose output
3. Test locally without Docker first to isolate issues

---

**Last Updated:** February 10, 2026  
**Python Version:** 3.11  
**Recommended Base Image:** `python:3.11-slim`
