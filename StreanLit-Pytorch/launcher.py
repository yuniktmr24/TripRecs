# import streamlit as st
# import torch
# import torchvision.transforms as T
# from PIL import Image
# import numpy as np

# # Load trained PT model
# learn_inf = torch.jit.load("transfer_exported.pt")

# def classify_image(img):
#     timg = T.ToTensor()(img).unsqueeze_(0)
#     softmax = learn_inf(timg).data.cpu().numpy().squeeze()
#     idxs = np.argsort(softmax)[::-1]
#     results = []
#     for i in range(5):
#         landmark_name = learn_inf.class_names[idxs[i]]
#         probability = softmax[idxs[i]]
#         results.append(f"{landmark_name} (prob: {probability:.2f})")
#     return results

# st.title("Landmark Classifier")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
#     st.write("")
#     st.write("Classifying...")
#     labels = classify_image(image)
#     for label in labels:
#         st.write(label)

# # To run the app:
# # streamlit run your_script.py


import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Paths to models
models = {
    "Resnet-18 model": "../ML/checkpoints/transfer_exported.pt",
     "VGG-16": "../ML/checkpoints/transfer_exported_vgg.pt",
     "MobileNet-v3-small": "../ML/checkpoints/transfer_exported_mobile_v3_small.pt"
}

def load_model(model_path):
    model = torch.jit.load(model_path)
    model.eval()  # Set model to evaluation mode
    return model

def classify_image(img, model):
    timg = T.ToTensor()(img).unsqueeze_(0)
    softmax = model(timg).data.cpu().numpy().squeeze()
    idxs = np.argsort(softmax)[::-1]
    results = []
    for i in range(5):
        landmark_name = model.class_names[idxs[i]]
        probability = softmax[idxs[i]]
        results.append(f"{landmark_name} (prob: {probability:.2f})")
    return results

st.title("TripRecs - Landmark Classifier")

# Dropdown for model selection
model_choice = st.selectbox("Choose a backend for inference:", list(models.keys()))
loaded_model = load_model(models[model_choice])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    labels = classify_image(image, loaded_model)
    for label in labels:
        st.write(label)
#streamlit run your_script.py
