### SolarVision Backend

Production-grade FastAPI backend for rooftop PV detection, validation, and batch processing.


## Hackathon Context & Quickstart

Our three-person team built this backend during a fast-paced hackathon sprint. Because shared GPUs were unavailable, the custom detector only trained for roughly 15–25 epochs. That 15-epoch local run hits roughly 20% accuracy on our hackathon validation subset, reflecting the constraints. We hosted training on Roboflow and exported a workflow that currently reports **mAP@50 = 82.4%**, **precision = 74.3%**, and **recall = 79.0%**. We pair detections with an Ollama-hosted LLaMA Phi-3 validator for qualitative checks. ***For reviewer convenience the env configuration lives in [\.env](.env); rotate those keys after judging.*** That hosted Roboflow workflow is the high-accuracy reference, while our limited-runtime export ships as [src/weights/best.pt](src/weights/best.pt) for offline use.

### Running the Model End-to-End

1. Install Virtualenv tooling: run `pip install virtualenv` if it is not already available.
2. Create and activate a virtual environment:
   - `python -m venv .venv`
   - Windows PowerShell: `.venv\Scripts\Activate.ps1`
   - macOS/Linux: `source .venv/bin/activate`
3. Install dependencies listed in requirements.txt (example: `pip install -r requirements.txt`).
4. Launch the API with `uvicorn src.main:app --reload`.
5. Open `http://127.0.0.1:8000/docs` for interactive testing.
6. Use the `POST /v1/verify/upload` endpoint, click **Try it out**, and upload a rooftop image.
7. Enter these reference query values (adjust as needed):
  - `gsd_m_per_px`: 0.25
  - `center_x_px`: 200
  - `center_y_px`: 140
8. Execute the request to view detections and overlays. The same settings work for `/v1/validate/upload` when validator output is needed.

### Direct Roboflow Workflow Test

Run the hosted workflow outside the backend if you want to inspect raw Roboflow output quickly:

```python
pip install inference-sdk
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
  api_url="https://serverless.roboflow.com",
  api_key="ZcI9dDFEjlVQfBliwb2u"
)

result = client.run_workflow(
  workspace_name="solar-k78aq",
  workflow_id="detect-count-and-visualize-2",
  images={
    "image": "img.jpg"
  },
  use_cache=True
)

print(result)
```

# For running the backend

```uvicorn src.main:app --reload```

## Prerequisites

- Python 3.11 (recommended). Install via:
  - Windows: https://www.python.org/downloads/
  - macOS: `brew install python@3.11`
  - Linux: Use your distro’s package manager or pyenv.

- uv (fast Python package manager/runner)
  - Install: `pip install uv` or `pipx install uv`
  - Docs: https://github.com/astral-sh/uv

- Git
- Optional: Ollama (for validator)
  - Install: https://ollama.com
  - Run: `ollama serve` and pull a vision model (e.g., `ollama pull llava-phi3`)

## Project Layout

```
backend/
  src/
    api.py
    core/
      config.py
      schemas.py
    pipeline/
      pipeline.py
      validation/response.py
      sources/static_maps.py
      intelligence/ollama.py
      model/model.py
  pyproject.toml
  env/.env           <-- keep local; DO NOT commit
```

## Environment Variables

Create `backend/env/.env` with your keys and settings. Share this privately (e.g., WhatsApp). Do not push it to GitHub.

Example:
```
# Detectors
ROBOFLOW_API_KEY=your_roboflow_key
ROBOFLOW_API_URL=https://serverless.roboflow.com
ROBOFLOW_WORKSPACE=your_workspace
ROBOFLOW_WORKFLOW_ID=detect-count-and-visualize-2

# Imagery sources
GOOGLE_STATIC_MAPS_KEY=your_google_key
BING_MAPS_KEY=your_bing_key

# Validator
OLLAMA_HOST=http://localhost:11434

# Toggles & tuning
SOURCE_PROVIDER=upload        # upload | google | bing
MODEL_PROVIDER=roboflow       # roboflow | local
VALIDATE=on                   # on | off
PREPROCESS=on                 # on | off
DETECTION_CONF_THR=0.55
DETECTION_IOU_THR=0.50
INFERENCE_IMG_SIZE=1280
FILL_RATIO=0.65
USE_CACHE=true

# Local model (optional)
YOLO_WEIGHTS=weights/yolov8.pt
```

## Setup with uv

1. Clone the repo:
```
git clone https://github.com/your-org/your-repo.git
cd your-repo/backend
```

2. Create a virtual environment:
```
uv venv .venv
```

3. Activate the environment:
- macOS/Linux:
  ```
  source .venv/bin/activate
  ```
- Windows (PowerShell):
  ```
  .venv\Scripts\Activate.ps1
  ```

4. Install dependencies from `pyproject.toml`:
```
uv pip install -e .
```
If you don’t have a PEP 517/518 installable package, install with:
```
uv pip install -r requirements.txt
```
or directly from `pyproject.toml`:
```
uv pip install -r <(python -c "import tomllib,sys;print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))")
```

5. Load environment variables:
- Export `.env` or use a loader. Easiest is to run via `uvicorn` after exporting:
  - macOS/Linux:
    ```
    set -a
    source env/.env
    set +a
    ```
  - Windows (PowerShell):
    ```
    Get-Content env/.env | ForEach-Object {
      if ($_ -match '^\s*#') { return }
      $parts = $_.Split('=',2)
      [System.Environment]::SetEnvironmentVariable($parts[0], $parts[1])
    }
    ```

## Running the Server

- Development:
```
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

- Production (example):
```
uvicorn src.main:app --host 0.0.0.0 --port 8000
```
Use a process manager (systemd, pm2, etc.) or a container in production.

## Health Check

- GET http://127.0.0.1:8000/v1/health
  - Returns `status: ok` and whether Ollama is reachable.

## Key Endpoints

- Verify (upload image)
  - POST `/v1/verify/upload`
  - Query params: `include_overlay, preprocess, gsd_m_per_px, center_x_px, center_y_px`
  - Multipart: `file=@image.jpg`
  - Returns normalized JSON with overlays:
    - `overlay_png_base64` (annotated)
    - `original_png_base64`
    - `overlay_preprocessed_png_base64`

- Validate (upload image → LMM rationale)
  - POST `/v1/validate/upload`
  - Query params: `model_name, include_overlay, preprocess, gsd_m_per_px, center_x_px, center_y_px`
  - Multipart: `file=@image.jpg`

- Verify by Lat/Lon (requires API keys)
  - POST `/v1/verify/latlon`
  - Query: `lat, lon, provider=google|bing, zoom, width, height, scale, include_overlay, preprocess, gsd_m_per_px`
  - Returns normalized JSON; if keys missing, returns `missing_*_key`.

- Validate by Lat/Lon
  - POST `/v1/validate/latlon` (same params as above + `model_name`)

- Batch CSV (Lat/Lon list; requires keys)
  - POST `/v1/verify/batch`
  - Multipart: `file=@sites.csv` with columns: `sample_id, lat, lon`
  - Returns:
    - `results`: list of per-site JSON responses
    - `overlays_zip_base64`: ZIP with `{sample_id}.png`

- Model smoke test
  - POST `/v1/model/test`
  - Multipart: `file=@image.jpg`
  - Query: `include_raw`

- Config
  - GET `/v1/config`
  - POST `/v1/config` to override runtime fields (non-persistent)

## Example cURL

- Upload verify:
```
curl -X POST "http://127.0.0.1:8000/v1/verify/upload?include_overlay=true&preprocess=true&gsd_m_per_px=0.35" \
  -F "file=@test.jpg"
```

- Upload validate:
```
curl -X POST "http://127.0.0.1:8000/v1/validate/upload?model_name=llava-phi3&include_overlay=false&preprocess=true&gsd_m_per_px=0.35" \
  -F "file=@test.jpg"
```

- Lat/Lon verify (Google):
```
curl -X POST "http://127.0.0.1:8000/v1/verify/latlon?lat=37.4219999&lon=-122.0840575&provider=google&zoom=19&width=640&height=640&include_overlay=true&preprocess=true&gsd_m_per_px=0.35"
```

- Batch:
```
curl -X POST "http://127.0.0.1:8000/v1/verify/batch?provider=google&include_overlay=true&preprocess=true&gsd_m_per_px=0.35" \
  -F "file=@sites.csv"
```

## Notes

- If you don’t have Google/Bing API keys, lat/lon endpoints will return `missing_*_key`. Upload endpoints work without keys.
- Keep `env/.env` private and off Git. Share with teammates via WhatsApp or other secure channels.
- If using the validator, ensure Ollama is running and the model is available. Update `OLLAMA_HOST` if running remotely.

## Troubleshooting

- Import errors: confirm `PYTHONPATH` includes `backend/src` or run from `backend` directory.
- Dependencies: if `uv pip install -e .` fails, check `pyproject.toml` `[project] dependencies`.
- Roboflow errors: verify `ROBOFLOW_*` values and workflow ID.
- Overlay fields missing: ensure `include_overlay=true` in the request.

## License

Internal use for hackathon/demo. Update with your organization’s license as needed.


