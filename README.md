# CoRT XTTS Embedder

This repository contains scripts for experimenting with DeepSeek's API and a FastAPI server. The project relies on `cort_xtts_embedder.py`, a FastAPI application used to interact with the embedding system.

## Prerequisites

1. **Anaconda** (or Miniconda) is recommended for managing the Python environment.
2. Create a new environment and install the required packages. Example commands:
   ```bash
   conda create -n xtts-env python=3.10
   conda activate xtts-env
   pip install fastapi uvicorn requests pydantic numpy scipy pandas torch librosa ffmpeg-python TTS pydub
   ```
   The list above covers the dependencies used across the scripts in this repository. You may install additional packages if the application reports missing modules.
3. Ensure `ffmpeg` is available on your system (either through your package manager or downloadable binaries).
4. Obtain a valid **DEEPSEEK_API_KEY** and set it as an environment variable:
   ```bash
   export DEEPSEEK_API_KEY="<your-api-key>"
   ```
   The API key is required for calls to DeepSeek's API.

## Running the FastAPI App

After installing the prerequisites and setting the API key, start the server with:
```bash
python cort_xtts_embedder.py
```
The application will run on `http://localhost:8000` by default. You can modify the host or port inside `cort_xtts_embedder.py` if needed.

