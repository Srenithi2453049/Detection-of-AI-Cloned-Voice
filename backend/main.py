from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from . import audio_utils, config, model_def
app = FastAPI(
    title="Deepfake Audio Detection PWA",
    description="Single-entry Progressive Web App with Hybrid CNN-LSTM (MFCC) backend.",
    version="1.0.0",
)
# Allow browser to call the API (even though we now serve UI and API from same origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Serve the existing frontend files (index.html, script.js, styles.css) from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
app.mount(
    "/static",
    StaticFiles(directory=str(PROJECT_ROOT / "static"), html=False),
    name="static",
)
MODEL = None
@app.on_event("startup")
def load_model_on_startup() -> None:
    """Load the trained CNN-LSTM model into memory once at startup."""
    global MODEL
    try:
        MODEL = model_def.load_model()
    except FileNotFoundError:
        MODEL = None
@app.get("/health")
def health() -> JSONResponse:
    """Simple health check endpoint for uptime monitoring."""
    status = "ok"
    details = {}
    model_loaded = MODEL is not None
    details["model_loaded"] = model_loaded
    if not model_loaded:
        status = "degraded"
    return JSONResponse({"status": status, "details": details})
@app.get("/", include_in_schema=False)
def ui_index() -> FileResponse:
    """Serve the main PWA UI."""
    index_path = PROJECT_ROOT / "index.html"
    return FileResponse(index_path)
@app.get("/script.js", include_in_schema=False)
def ui_script() -> FileResponse:
    """Serve the frontend JavaScript."""
    return FileResponse(PROJECT_ROOT / "script.js")
@app.get("/styles.css", include_in_schema=False)
def ui_styles() -> FileResponse:
    """Serve the frontend CSS."""
    return FileResponse(PROJECT_ROOT / "styles.css")
@app.get("/manifest.json", include_in_schema=False)
def ui_manifest() -> FileResponse:
    """Serve the PWA manifest so browsers can detect the app."""
    return FileResponse(PROJECT_ROOT / "manifest.json")
@app.get("/sw.js", include_in_schema=False)
def ui_service_worker() -> FileResponse:
    """Serve the service worker script for offline/PWA support."""
    return FileResponse(PROJECT_ROOT / "sw.js")
@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    Predict whether an uploaded audio file is real or deepfake.

    Request (multipart/form-data):
        file: audio file (.wav recommended)

    Response (JSON):
        {
          "fake_probability": 0.87,
          "label": "fake" | "real"
        }
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please train the model first.",
        )

    if not file.filename.lower().endswith((".wav", ".flac", ".mp3", ".ogg", ".m4a", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    tmp_path = Path("tmp_uploaded_audio")

    try:
        content = await file.read()
        with tmp_path.open("wb") as f:
            f.write(content)

        mfcc_batch = audio_utils.extract_mfcc_from_file(tmp_path)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {exc}")
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    proba = float(model_def.predict_proba(MODEL, mfcc_batch)[0])
    label = "fake" if proba >= 0.5 else "real"

    return JSONResponse(
        {
            "fake_probability": proba,
            "label": label,
        }
    )
def get_app() -> FastAPI:
    """Helper for ASGI servers."""
    return app

