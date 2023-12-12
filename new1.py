import os
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from annoy import AnnoyIndex
import pickle

app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'
app.config['UPLOAD_FOLDER'] = os.path.join(app.config['STATIC_FOLDER'], 'uploads')
app.config['DATASET_FOLDER'] = os.path.join(app.config['STATIC_FOLDER'], 'dataset')
app.config['FEATURES_FILE'] = os.path.join(app.config['STATIC_FOLDER'], 'features.pkl')

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

# Function to create an Annoy index
def create_index(dataset_features, num_trees=10):
    index = AnnoyIndex(dataset_features.shape[1], metric='manhattan')
    for i, features in enumerate(dataset_features):
        index.add_item(i, features)
    index.build(num_trees)
    return index

# Function to search for similar images using Annoy index
def search_similar_images(query_features, index, k=5):
    similar_indices = index.get_nns_by_vector(query_features, k, search_k=-1, include_distances=False)
    return similar_indices

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

    if os.path.exists(app.config['FEATURES_FILE']):
        # Load precomputed features from file
        with open(app.config['FEATURES_FILE'], 'rb') as f:
            dataset_features, dataset_images = pickle.load(f)
    else:
        dataset_features = []
        dataset_images = []

        # Iterate through the dataset folder to extract features from each image
        for filename in os.listdir(app.config['DATASET_FOLDER']):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(app.config['DATASET_FOLDER'], filename)
                similar_images.append(img_path)
                features = extract_features(img_path)
                dataset_features.append(features)
                dataset_images.append(img_path)

        dataset_features = np.array(dataset_features)
        dataset_images = np.array(dataset_images)

        # Save computed features to file
        with open(app.config['FEATURES_FILE'], 'wb') as f:
            pickle.dump((dataset_features, dataset_images), f)

    index = create_index(dataset_features)

    similar_indices = search_similar_images(query_features, index, k=5)
    top_results = dataset_images[similar_indices]

    return render_template('results.html', query_image=query_image_path, similar_images=top_results)

if __name__ == '__main__':
    app.run(debug=True)
