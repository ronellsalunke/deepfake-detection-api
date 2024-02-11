from fastapi import FastAPI, File, UploadFile
import helper

app = FastAPI()

@app.get("/")
async def hello():
    return {"ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    processed_img = helper.preproc(contents)
    image_pred,percentage = helper.processing(processed_img)
    return {
        'result' : image_pred,
        'likliness': percentage
    }   