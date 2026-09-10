from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from detection.router import router as detection_router

app = FastAPI(title="ANPR Detection Module")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection_router)