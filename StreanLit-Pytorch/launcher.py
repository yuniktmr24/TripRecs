import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
import random
import re

# Define your model paths and other constants
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

def get_multiple_sample_images(landmark_name, num_images=5):
    folder_path = f"../ML/landmark_images/train/{landmark_name}"
    image_paths = []
    if os.path.exists(folder_path) and len(os.listdir(folder_path)) >= num_images:
        images = random.sample(os.listdir(folder_path), num_images)
        for image in images:
            image_paths.append(os.path.join(folder_path, image))
    return image_paths

def clean_landmark_name(name):
    return re.sub(r'^\d+\.', '', name).replace('_', ' ').strip()

def main_page():
    """ This page handles image classification and displaying similar images. """
    st.title("TripRecs - Landmark Classifier")
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
        for label in labels:
            cols = st.columns(5)
            for col, img_path in zip(cols, label['sample_image_paths']):
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    col.image(img, use_column_width=True)

def search_by_images_page():
    """ This page provides a search interface for users to find landmarks. """
    st.title("Search Landmarks")
    search_query = st.text_input("Enter a location to search for:")
    if search_query:
        search_results = search_landmarks(search_query)
        if search_results:
            for landmark, images in search_results.items():
                st.write(f"### Images for {clean_landmark_name(landmark)}")
                cols = st.columns(5)  # Fewer columns for larger images
                for col, img_path in zip(cols, images):
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        col.image(img, use_column_width=True)
        else:
            st.write("No matching locations found.")

def search_landmarks(query):
    folder_base = "../ML/landmark_images/train/"
    landmarks = [d for d in os.listdir(folder_base) if os.path.isdir(os.path.join(folder_base, d))]
    query_lower = query.lower()
    matched_landmarks = [landmark for landmark in landmarks if query_lower in landmark.lower().replace('_', ' ')]
    results = {}
    for landmark in matched_landmarks:
        images = get_multiple_sample_images(landmark, num_images=5)
        results[landmark] = images
    return results

def travel_fare_page():
    """ This page allows users to search for travel fares to landmarks. """
    st.title("Travel Fare Lookup")
    location_query = st.text_input("Enter a location to lookup travel fares for:", key="fareSearch")

    # Placeholder for actual travel fare data lookup
    if location_query:
        # Example: Pretend we look up fares here and found some data
        # In a real application, you would query a database or an API based on `location_query`
        travel_fares = {"Example Destination": "$200 - $400", "Another Destination": "$300 - $500"}
        
        st.write(f"### Travel fare results for: {location_query}")
        for destination, fare in travel_fares.items():
            st.write(f"**{destination}:** {fare}")
        
        # If no fares found, display a message
        # st.write("No travel fares found for this location.")


def set_custom_css():
    """Injects custom CSS to style the app."""
    custom_css = """
    <style>
        /* Modify the sidebar background color */
        .css-1lcbmhc {
            background-color: #f0f0f5 !important;
        }
        
        /* Modify sidebar text color */
        .css-1lcbmhc a {
            color: #6C6C6F !important;
        }

        /* Sidebar link hover effect */
        .css-1lcbmhc a:hover {
            color: #ffffff !important;
            background-color: #4a4a4a !important;
        }

        /* Active page styling in sidebar */
        .css-1lcbmhc a[data-baseweb="tab"] {
            font-weight: bold !important;
            color: #4A90E2 !important; /* Change this color for non-hover active links */
        }

        /* Sidebar width adjustment if needed */
        .css-1lcbmhc {
            width: 300px !important;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

set_custom_css()

# Main navigation
st.sidebar.title("TripRecs - Navigation")
page = st.sidebar.radio(
    "Choose a page:",
    ["Inference", "Search by Images", "Travel Fare"]  
)
# Page selection
if page == "Inference":
    main_page()
elif page == "Search by Images":
    search_by_images_page()
elif page == "Travel Fare": 
    travel_fare_page()


