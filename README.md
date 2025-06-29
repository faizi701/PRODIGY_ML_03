Hand Gesture Recognition Project
Overview
This repository contains a hand gesture recognition system developed using a Convolutional Neural Network (CNN) trained on the LeapGestRecog dataset. The project includes a Jupyter notebook for model training, a TensorFlow Lite model for inference, and a Streamlit web application for real-time gesture recognition via webcam. The model classifies 10 hand gestures (e.g., "thumb", "palm", "down") with a test accuracy of approximately 99.90%.
Features

Trains a CNN on 128x128 RGB images from the LeapGestRecog dataset.
Converts the model to TensorFlow Lite for efficient deployment.
Real-time gesture recognition using a webcam via a Streamlit app.
Supports 10 gesture classes with balanced data (2000 samples per class).

Prerequisites

Python 3.11 or higher
Required libraries:
opencv-python
numpy
matplotlib
scikit-learn
tensorflow
streamlit
streamlit-webrtc



Usage
Training the Model

Open hand-recognition (7).ipynb in Jupyter Notebook or a compatible environment.
Ensure the dataset is correctly loaded from the dataset folder.
Run all cells to train the model, save it as leap_gesture_model.h5, and convert it to leap_gesture_model.tflite.
The label map is saved as label_map.npy.

Running the Streamlit App

Ensure the trained leap_gesture_model.tflite and label_map.npy are in the root directory or specified path.

Start the Streamlit app:
streamlit run app.py


Access the app at http://localhost:8501 in your browser.

Use your webcam to perform gestures or upload images for prediction.


Files

hand-recognition (7).ipynb: Jupyter notebook for model training and evaluation.
app.py: Streamlit application for real-time gesture recognition.
leap_gesture_model.tflite: Trained TensorFlow Lite model.
label_map.npy: Mapping of integer labels to gesture names.
dataset/: Directory for the LeapGestRecog dataset (to be added locally).

Notes

The model was trained on 128x128 RGB images to match webcam input, addressing issues with grayscale-only training.
Increasing image size to 224x224 caused RAM crashes on Kaggle; 128x128 is stable.
For real-time use, ensure webcam permissions are granted in your browser.
Test accuracy may vary slightly based on local hardware and dataset preprocessing.

Contributing
Feel free to fork this repository, submit issues, or propose enhancements via pull requests.
Acknowledgments

Dataset: LeapGestRecog from Kaggle 
https://www.kaggle.com/datasets/gti-upm/leapgestrecog
Inspiration: Kaggle community and Streamlit documentation.
