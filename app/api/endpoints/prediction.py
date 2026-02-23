from fastapi import APIRouter, File, UploadFile
from app.services import ai_model

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    processed_img = ai_model.preproc(contents)
    image_pred, percentage = ai_model.processing(processed_img)
    return {
        'result' : image_pred,
        'likeliness': percentage
    }
