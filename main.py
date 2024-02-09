from fastapi import FastAPI, File, UploadFile
import helper

app = FastAPI()

@app.get("/")
async def hello():
    return {"ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await file.read()
    image_name = file.filename
    image_pred = helper.processing(image_name)
    return {
        'file_name' : image_name,
        'result' : image_pred
    }   