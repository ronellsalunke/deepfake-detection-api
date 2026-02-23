from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.api.endpoints import health, prediction

app = FastAPI(title="Deepfake Detection API")

app.include_router(health.router, tags=["health"])
app.include_router(prediction.router, tags=["prediction"])

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home():
    return """
    <html>
        <head>
            <title>Image Upload</title>
        </head>
        <body>
            <h1>Upload an Image</h1>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file">
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """
