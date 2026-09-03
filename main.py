from fastapi import FastAPI
from detection.router import router as detection_router

app = FastAPI(title="ANPR Detection Module")

app.include_router(detection_router)

@app.get("/")
def health_check():
    return {"status": "detection module running"}