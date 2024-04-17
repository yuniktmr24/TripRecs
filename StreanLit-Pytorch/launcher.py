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
        image_paths = get_multiple_sample_images(landmark_name, num_images=5)
        results.append({
            "landmark": landmark_name,
            "probability": f"{probability:.2f}",
            "sample_image_paths": image_paths
        })
    return results

def clean_landmark_name(name):
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

def search_landmarks(query):
    folder_base = "../ML/landmark_images/train/"
    landmarks = [d for d in os.listdir(folder_base) if os.path.isdir(os.path.join(folder_base, d))]
    matched_landmarks = [landmark for landmark in landmarks if query.lower() in landmark.lower()]
    results = {}
    for landmark in matched_landmarks:
        images = get_multiple_sample_images(landmark, num_images=5)
        results[landmark] = images
    return results

st.title("TripRecs - Landmark Classifier")

# Dropdown for model selection
model_choice = st.selectbox("Choose a backend for inference:", list(models.keys()))
loaded_model = load_model(models[model_choice])

# Image upload and classification
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    labels = classify_image(image, loaded_model)
    for label in labels:
        st.write(f"{label['landmark']} (prob: {label['probability']})")

    st.write("## You may also like:")
    for label in labels:
        cols = st.columns(5)
        for col, img_path in zip(cols, label['sample_image_paths']):
            if os.path.exists(img_path):
                img = Image.open(img_path)
                col.image(img, use_column_width=True)

# Search functionality
st.write("## Search for a Location")
search_query = st.text_input("Enter a location to search for:")
if search_query:
    search_results = search_landmarks(search_query)
    if search_results:
        for landmark, images in search_results.items():
            st.write(f"### Images for {clean_landmark_name(landmark)}")
            cols = st.columns(5)
            for col, img_path in zip(cols, images):
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    col.image(img, use_column_width=True)
    else:
        st.write("No matching locations found.")
