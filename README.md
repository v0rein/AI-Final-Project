# Face Similarity Comparison App

This application allows users to compare two face images and determine their similarity using multiple deep learning models.

## Features

- Compares two face images using five different face recognition models.
- Each model has optimized thresholds for accuracy.
- Provides a consensus-based verification system.
- Displays detailed confidence indicators, average similarity, and average distance.

## Models Used

The application utilizes the following face recognition models:

- VGG-Face (Cosine distance metric)
- Facenet (Euclidean L2 distance metric)
- ArcFace (Cosine distance metric)
- OpenFace (Cosine distance metric)
- DeepFace (Cosine distance metric)

## How to Use

1.  Ensure you have Python installed.
2.  Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
3.  Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```
4.  The application will open in your web browser.
5.  Upload two face images in the designated areas.
6.  Click the "Compare Faces" button.
7.  View the detailed similarity results, including overall similarity, distance, and model consensus.

## Tips for Best Results

- Use clear, frontal face images.
- Ensure good lighting conditions in the images.
- Avoid obstructions such as glasses or heavy makeup if possible.
- Encourage neutral facial expressions.
- Use high-resolution images for better accuracy.
