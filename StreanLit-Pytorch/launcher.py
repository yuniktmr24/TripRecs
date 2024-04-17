import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
import random
import re

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
        # Gather multiple sample images for each prediction
        image_paths = get_multiple_sample_images(landmark_name, num_images=5)
        results.append({
            "landmark": landmark_name,
            "probability": f"{probability:.2f}",
            "sample_image_paths": image_paths
        })
    return results

def clean_landmark_name(name):
    # Remove leading numbers and period using regular expression
    cleaned_name = re.sub(r'^\d+\.', '', name).replace('_', ' ')
    return cleaned_name.strip()

def get_sample_image_path(landmark_name):
    folder_path = f"../ML/landmark_images/train/{landmark_name}"
    if os.path.exists(folder_path):
        sample_image = random.choice(os.listdir(folder_path))
        return os.path.join(folder_path, sample_image)
    return None

def get_multiple_sample_images(landmark_name, num_images=5):
    folder_path = f"../ML/landmark_images/train/{landmark_name}"
    image_paths = []
    if os.path.exists(folder_path) and len(os.listdir(folder_path)) >= num_images:
        images = random.sample(os.listdir(folder_path), num_images)
        for image in images:
            image_paths.append(os.path.join(folder_path, image))
    return image_paths

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
        st.write(f"{label['landmark']} (prob: {label['probability']})")
    
    st.write("## You may also like:")
    # Create a row of columns for each prediction
    for label in labels:
        #st.write(f"### Samples for {clean_landmark_name(label['landmark'])}")
        cols = st.columns(5)
        if label['sample_image_paths']:
            for col, img_path in zip(cols, label['sample_image_paths']):
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    col.image(img, use_column_width=True)

