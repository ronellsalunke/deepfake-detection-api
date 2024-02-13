from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import helper

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Image Upload</title>
        </head>
        <body>
            <h1>Upload an Image</h1>
            <form action="/predict/" method="post" enctype="multipart/form-data">
                <input type="file" name="file">
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    processed_img = helper.preproc(contents)
    image_pred,percentage = helper.processing(processed_img)
    return {
        'result' : image_pred,
        'likliness': percentage
    }   