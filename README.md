# Grocery Shop Image Similarity Search App

# Overview
This repository contains code and resources for an Image Similarity Search application designed for a grocery shop. The app utilizes Flask as the backend framework, TensorFlow for image processing with the VGG16 model, and Annoy for efficient similarity search. The frontend interface is built using HTML and CSS.

# Introduction
The objective of this project is to create an application that allows users to search for visually similar items within the grocery shop's inventory. Leveraging image processing techniques and a neural network like VGG16, the app provides an efficient way to find items visually similar to a given query image.

# Features
Image Similarity Search: Uses VGG16 model in TensorFlow to extract image features for similarity comparison.

Annoy for Fast Search: Implements Annoy for indexing and fast approximate nearest neighbor search.

Flask Backend: Provides a Flask-based backend for handling image queries and serving results.

HTML & CSS Interface: Offers a user-friendly web interface for uploading query images and viewing similar items.

# Usage

Start the Application: Run the Flask server to start the image similarity search application.

Upload Images: Use the provided interface to upload an image as a query.

Find Similar Items: The app should display visually similar items from the grocery shop's inventory based on the query image.
