from dotenv import load_dotenv
import os

# Load env/.env on startup
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "env", ".env"))

from fastapi import FastAPI
from .api import router as api_router

app = FastAPI(title="PV Rooftop Backend", version="0.4.0")
app.include_router(api_router, prefix="/v1")