
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as T
from PIL import Image
import io
import numpy as np

app = FastAPI()

models = {
    "Resnet-18 model": "../ML/checkpoints/transfer_exported.pt",
     "VGG-16": "../ML/checkpoints/transfer_exported_vgg.pt",
     "MobileNet-v3-small": "../ML/checkpoints/transfer_exported_mobile_v3_small.pt"
}

def load_model(model_path):
    return torch.jit.load(model_path)

@app.post("/classify/")
async def classify_image(file: UploadFile = File(...), model_choice: str = Form(...)):
    try:
        model_path = models[model_choice]
        model = load_model(model_path)

        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        img = T.ToTensor()(img).unsqueeze_(0)

        softmax = model(img).data.cpu().numpy().squeeze()
        idxs = np.argsort(softmax)[::-1]

        results = []
        for i in range(5):
            landmark_name = model.class_names[idxs[i]]
            probability = softmax[idxs[i]]
            results.append({"landmark": landmark_name, "probability": f"{probability:.2f}"})

        return JSONResponse(status_code=200, content={"results": results})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})






# Run the server: uvicorn main:app --reload
