# app.py
import os
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'
app.config['UPLOAD_FOLDER'] = os.path.join(app.config['STATIC_FOLDER'], 'uploads')
app.config['DATASET_FOLDER'] = os.path.join(app.config['STATIC_FOLDER'], 'dataset')

# Load pre-trained VGG16 model
vgg16_model = VGG16(weights='imagenet', include_top=False)

# Function to extract features from an image using VGG16 model
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = vgg16_model.predict(img)
    features = features.flatten()
    return features

# Function to calculate cosine similarity between two feature vectors
def calculate_similarity(query_features, dataset_features):
    similarity_scores = cosine_similarity(query_features.reshape(1, -1), dataset_features)
    return similarity_scores.flatten()

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for image similarity search
@app.route('/search', methods=['POST'])
def search():
    query_image = request.files['query_image']
    query_image_path = os.path.join(app.config['UPLOAD_FOLDER'], query_image.filename)
    query_image.save(query_image_path)

    similar_images = []

    query_features = extract_features(query_image_path)

    dataset_features = []
    """dataset_images = []"""

    # Iterate through the dataset folder to extract features from each image
    for filename in os.listdir(app.config['DATASET_FOLDER']):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(app.config['DATASET_FOLDER'], filename)
            similar_images.append(img_path)
            features = extract_features(img_path)
            dataset_features.append(features)

    dataset_features = np.array(dataset_features)

    similarity_scores = calculate_similarity(query_features, dataset_features)
    sorted_indices = np.argsort(similarity_scores)[::-1]  # Sort in descending order

    top_results = []
    for i in range(5):  # Retrieve top 5 similar images
        img_path = similar_images[sorted_indices[i]]
        top_results.append(img_path)

    return render_template('results.html', query_image=query_image_path, similar_images=top_results)

if __name__ == '__main__':
    app.run(debug=True)
