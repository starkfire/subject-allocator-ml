from fastapi import FastAPI, HTTPException, File, UploadFile

app = FastAPI()

@app.get("/ping")
async def ping():
    return { "message": "Pong!" }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    VALID_MIMETYPES = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]

    if file.content_type not in VALID_MIMETYPES:
        raise HTTPException(status_code=400, detail="Invalid File Type")

    file_contents = await file.read()

    return {
        "filename": file.filename,
        "mimetype": file.content_type,
        "file_size": len(file_contents)
    }
