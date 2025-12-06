# webapp/backend/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI(title="MRI GenAI Demo")

@app.get("/")
def read_root():
    return {"message": "MRI GenAI backend running."}

@app.post("/denoise")
async def denoise_endpoint(
    t1_file: UploadFile = File(...),
    dmri_file: UploadFile = File(...),
    model_name: str = "swin_unetr",
):
    # TODO:
    # 1. Read NIfTI / npy from files
    # 2. Preprocess
    # 3. Run model inference
    # 4. Return some metadata / URLs to images
    return JSONResponse(
        {"status": "not_implemented", "model_name": model_name}
    )
