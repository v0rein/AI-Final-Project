import streamlit as st
import os
import cv2
from deepface import DeepFace
import numpy as np
from PIL import Image
import tempfile

# Configure the app
st.set_page_config(
    page_title="Face Similarity Comparison",
    page_icon="👥",
    layout="wide"
)

MODELS = {
    "VGG-Face": {"threshold": 0.30, "distance_metric": "cosine"},
    "Facenet": {"threshold": 0.25, "distance_metric": "euclidean_l2"},
    "ArcFace": {"threshold": 0.20, "distance_metric": "cosine"},
    "OpenFace": {"threshold": 0.10, "distance_metric": "cosine"},
    "DeepFace": {"threshold": 0.35, "distance_metric": "cosine"}
}

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .image-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
    }
    .image-box {
        width: 48%;
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .result-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .progress-container {
        height: 20px;
        background-color: #e9ecef;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        background-color: #4CAF50;
        width: 0%;
        transition: width 0.5s ease;
    }
    .face-highlight {
        position: absolute;
        border: 3px solid #FF5722;
        border-radius: 5px;
        display: none;
    }
    .metric-box {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("👥 Face Similarity Comparison")
st.markdown("""
Compare two face images using multiple face recognition models.
Upload clear frontal face images for best results.
""")

# Initialize session state for images
if 'image1' not in st.session_state:
    st.session_state.image1 = None
if 'image2' not in st.session_state:
    st.session_state.image2 = None

# Image upload columns
col1, col2 = st.columns(2)


def load_image(image_file):
    """Load and validate an image file."""
    if image_file is not None:
        try:
            img = Image.open(image_file)
            return img
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return None
    return None


def detect_faces(image_path):
    """Detect faces in an image and return the first face coordinates."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return None

        return faces[0]  # Return first face (x, y, w, h)
    except Exception:
        return None


# Image upload and display
with col1:
    st.subheader("First Image")
    image_file1 = st.file_uploader("Upload first face image", type=[
                                   "jpg", "jpeg", "png"], key="img1")
    if image_file1 is not None:
        st.session_state.image1 = load_image(image_file1)
        st.image(st.session_state.image1,
                 caption="Image 1", use_column_width=True)

with col2:
    st.subheader("Second Image")
    image_file2 = st.file_uploader("Upload second face image", type=[
                                   "jpg", "jpeg", "png"], key="img2")
    if image_file2 is not None:
        st.session_state.image2 = load_image(image_file2)
        st.image(st.session_state.image2,
                 caption="Image 2", use_column_width=True)


def compare_faces_multimodel(img1_path, img2_path):
    results = []
    for model_name, config in MODELS.items():
        try:
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=model_name,
                distance_metric=config["distance_metric"],
                enforce_detection=True,
                detector_backend="opencv"
            )

            distance = float(result['distance'])
            verified = distance < config["threshold"]
            similarity = max(0, 100 - (distance * 100))

            results.append({
                'verified': verified,
                'similarity': similarity,
                'distance': distance
            })
        except Exception:
            continue

    if not results:
        return None

    # Calculate average results
    avg_similarity = sum(r['similarity'] for r in results) / len(results)
    avg_distance = sum(r['distance'] for r in results) / len(results)
    consensus = sum(1 for r in results if r['verified']) / len(results)

    return {
        'similarity': avg_similarity,
        'distance': avg_distance,
        'consensus': consensus,
        'model_count': len(results)
    }


if st.button("Compare Faces", type="primary", use_container_width=True):
    if st.session_state.image1 and st.session_state.image2:
        with st.spinner("Analyzing faces..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp1:
                if st.session_state.image1.mode == 'RGBA':
                    st.session_state.image1 = st.session_state.image1.convert(
                        'RGB')
                st.session_state.image1.save(tmp1.name, format='JPEG')
                img1_path = tmp1.name

            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp2:
                if st.session_state.image2.mode == 'RGBA':
                    st.session_state.image2 = st.session_state.image2.convert(
                        'RGB')
                st.session_state.image2.save(tmp2.name, format='JPEG')
                img2_path = tmp2.name

            try:
                result = compare_faces_multimodel(img1_path, img2_path)

                if result:
                    st.subheader("Overall Comparison Result")

                    # Visual progress bar
                    st.markdown(f"""
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {result['similarity']}%"></div>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average Similarity",
                                  f"{result['similarity']:.1f}%")
                        st.metric("Average Distance",
                                  f"{result['distance']:.4f}")

                    with col2:
                        st.metric("Model Consensus",
                                  f"{result['consensus']*100:.0f}% ({int(result['consensus']*result['model_count'])}/{result['model_count']} models agree)")

                        if result['consensus'] >= 0.75:
                            st.success("✅ Strong match consensus")
                        elif result['consensus'] >= 0.5:
                            st.warning("⚠️ Partial match consensus")
                        else:
                            st.error("❌ No match consensus")

                    # Final recommendation
                    if result['similarity'] < 70:
                        st.warning("Low confidence result - possible error")

                    with st.expander("How to interpret these results"):
                        st.markdown("""
                        - **Similarity Score**: Percentage indicating how similar the faces are (higher is better)
                        - **Distance**: The distance between face embeddings (lower is better)
                        - **Threshold**: The decision threshold for each model
                        - **Confidence Levels**:
                            - High: >85% similarity
                            - Medium: 70-85% similarity
                            - Low: <70% similarity
                        """)

                else:
                    st.error("No models produced valid results")

            except Exception as e:
                st.error(f"Comparison failed: {str(e)}")

            finally:
                try:
                    os.unlink(img1_path)
                    os.unlink(img2_path)
                except:
                    pass
    else:
        st.warning("Please upload both images first.")

# Information sections
with st.expander("About this app"):
    st.markdown("""
    **Multi-Model Face Similarity Comparison Tool**
    
    This application uses multiple deep learning models to compare two face images and determine their similarity.
    
    **Features:**
    - Uses five different face recognition models
    - Each model with optimized thresholds
    - Consensus-based verification system
    - Detailed confidence indicators
    
    **Models Used:**
    - VGG-Face (Cosine)
    - Facenet (Euclidean L2)
    - ArcFace (Cosine)
    - OpenFace (Cosine)
    - DeepFace (Cosine)
    """)

with st.sidebar:
    st.markdown("""
    ### Instructions
    1. Upload two face images
    2. Click "Compare Faces"
    3. View detailed similarity results
    
    **Tips for best results:**
    - Use clear frontal face images
    - Ensure good lighting conditions
    - Avoid obstructions (glasses, heavy makeup)
    - Maintain neutral facial expressions
    - Use high-resolution images
    """)
