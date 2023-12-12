import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from annoy import AnnoyIndex
import pickle

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
    index = AnnoyIndex(dataset_features.shape[1], metric='euclidean')
    for i, features in enumerate(dataset_features):
        index.add_item(i, features)
    index.build(num_trees)
    return index

# Function to search for similar images using Annoy index
def search_similar_images(query_features, index, k=5):
    similar_indices = index.get_nns_by_vector(query_features, k, search_k=-1, include_distances=False)
    return similar_indices

# Load dataset features and images
def load_dataset_features():
    dataset_folder = 'static/dataset'
    features_file = 'static/features.pkl'

    if os.path.exists(features_file):
        with open(features_file, 'rb') as f:
            dataset_features, dataset_images = pickle.load(f)
    else:
        dataset_features = []
        dataset_images = []

        for filename in os.listdir(dataset_folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(dataset_folder, filename)
                features = extract_features(img_path)
                dataset_features.append(features)
                dataset_images.append(img_path)

        dataset_features = np.array(dataset_features)
        dataset_images = np.array(dataset_images)

        with open(features_file, 'wb') as f:
            pickle.dump((dataset_features, dataset_images), f)

    return dataset_features, dataset_images

# Evaluate image similarity search app
def evaluate():
    dataset_features, dataset_images = load_dataset_features()
    index = create_index(dataset_features)

    query_folder = 'static/uploads'
    evaluation_folder = 'static/evaluation'
    query_files = os.listdir(query_folder)

    for query_file in query_files:
        query_path = os.path.join(query_folder, query_file)
        query_image = Image.open(query_path)
        query_image.show()

        query_features = extract_features(query_path)
        similar_indices = search_similar_images(query_features, index)
        top_results = dataset_images[similar_indices]

        query_name = os.path.splitext(query_file)[0]
        result_folder = os.path.join(evaluation_folder, query_name)
        os.makedirs(result_folder, exist_ok=True)

        for i, result_image in enumerate(top_results):
            result_name = f"result_{i+1}.jpg"
            result_path = os.path.join(result_folder, result_name)
            Image.open(result_image).save(result_path)

            print(f"Saved result {i+1}: {result_path}")

# Run the evaluation
if __name__ == '__main__':
    evaluate()
