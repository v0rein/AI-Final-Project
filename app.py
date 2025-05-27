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
    "VGG-Face": {"threshold": 0.40, "distance_metric": "cosine"},
    "Facenet": {"threshold": 0.40, "distance_metric": "euclidean"}, # DeepFace default for Facenet is 0.40
    "Facenet512": {"threshold": 1.0, "distance_metric": "euclidean_l2"}, # Updated from Facenet to Facenet512 to match newer DeepFace model names
    "ArcFace": {"threshold": 0.68, "distance_metric": "cosine"}, # DeepFace default for ArcFace is 0.68
    "Dlib": {"threshold": 0.07, "distance_metric": "euclidean_l2"}, # Dlib often uses euclidean_l2, threshold needs tuning, DeepFace suggests 0.07 for its Dlib ResNet model
    "OpenFace": {"threshold": 0.80, "distance_metric": "cosine"}, # This seems low, DeepFace's default for OpenFace is often higher (e.g. 0.40 with euclidean_l2) or needs checking
    "DeepFace": {"threshold": 0.59, "distance_metric": "cosine"} # DeepFace model (wrapper) default is 0.23
}
# Note: Thresholds for DeepFace models can vary slightly based on their training and the specific version.
# It's good practice to check DeepFace documentation or experiment for optimal thresholds.
# For "Dlib", DeepFace often refers to a ResNet-based model trained with Dlib.
# If you intend to use Dlib's HOG + landmark based older recognition, that's a different setup.
# Assuming DeepFace's "Dlib" model name refers to its built-in Dlib-based recognizer.

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
Compare two face images using selected face recognition models.
Upload clear frontal face images for best results.
""")

# Initialize session state for images
if 'image1' not in st.session_state:
    st.session_state.image1 = None
if 'image2' not in st.session_state:
    st.session_state.image2 = None

# Image upload columns
col1_img, col2_img = st.columns(2)


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

# Image upload and display
with col1_img:
    st.subheader("First Image")
    image_file1 = st.file_uploader("Upload first face image", type=[
                                   "jpg", "jpeg", "png"], key="img1")
    if image_file1 is not None:
        st.session_state.image1 = load_image(image_file1)
        if st.session_state.image1:
            st.image(st.session_state.image1,
                     caption="Image 1", use_column_width=True)

with col2_img:
    st.subheader("Second Image")
    image_file2 = st.file_uploader("Upload second face image", type=[
                                   "jpg", "jpeg", "png"], key="img2")
    if image_file2 is not None:
        st.session_state.image2 = load_image(image_file2)
        if st.session_state.image2:
            st.image(st.session_state.image2,
                     caption="Image 2", use_column_width=True)

st.markdown("---") # Separator

# Model Selection
st.subheader("⚙️ Model Selection")
selected_model_names = st.multiselect(
    "Select models to use for comparison (at least one):",
    options=list(MODELS.keys()),
    default=list(MODELS.keys()) # Select all by default
)

st.markdown("---") # Separator

def compare_faces_multimodel(img1_path, img2_path, chosen_models):
    results_summary = [] # To store individual model results for detailed view later if needed
    aggregated_metrics = {'similarity': [], 'distance': [], 'verified_count': 0}

    if not chosen_models:
        st.error("No models selected for comparison.")
        return None

    progress_bar_st = st.progress(0)
    status_text_st = st.empty()
    num_chosen_models = len(chosen_models)

    for i, model_name in enumerate(chosen_models):
        status_text_st.text(f"Processing with {model_name} ({i+1}/{num_chosen_models})...")
        config = MODELS[model_name]
        try:
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=model_name,
                distance_metric=config["distance_metric"],
                enforce_detection=True, # Keep True for robustness
                detector_backend="opencv" # You can change this or make it configurable
            )

            distance = float(result['distance'])
            verified = distance < config["threshold"] # DeepFace 'verified' key is already boolean

            # Calculate similarity. This can be tricky as different metrics have different ranges.
            # For cosine: similarity = (1 - distance) * 100
            # For euclidean/euclidean_l2: harder to map directly to 0-100% similarity.
            # A common approach is 100 / (1 + distance), or based on max expected distance.
            # For simplicity, let's use DeepFace's direct "verified" and the distance itself.
            # We can derive a generic similarity score for visualization if needed.
            # Here, a simple inverse relation to distance might be:
            # similarity_score = max(0, 100 - (distance * X)) where X depends on typical distance range.
            # For consistency with your previous code:
            if config["distance_metric"] == "cosine":
                similarity_score = max(0, (1 - distance) * 100)
            else: # euclidean, euclidean_l2
                # This heuristic might need adjustment based on typical euclidean distance ranges for each model
                # A smaller distance means more similar. A distance of 0 is 100% similar.
                # If typical max distance where faces are still somewhat similar is ~1.0 for Facenet,
                # then similarity = max(0, 100 - distance * 100).
                # If typical max distance for Dlib is ~0.6 for "same", then for Dlib, distance * (100/0.6)
                similarity_score = max(0, 100 - (distance * (100 / (config["threshold"]*2 if config["threshold"] > 0 else 1) ) ) ) # Very rough heuristic
                # Let's stick to the simpler one you had, and average that.
                # similarity_score = max(0, 100 - (distance * 100)) # This makes less sense for euclidean distances > 1

            # Using DeepFace 'verified' directly is more reliable for "same/not same"
            # For overall similarity percentage, we average the distances and try to convert
            # Or better, focus on "verified" consensus and average distance.

            aggregated_metrics['distance'].append(distance)
            if result.get('verified', False): # Use .get for safety
                 aggregated_metrics['verified_count'] += 1

            results_summary.append({
                'model': model_name,
                'verified': result.get('verified', False),
                'distance': distance,
                'threshold': config["threshold"]
                # 'similarity_score': similarity_score # If you want to calculate and show individual model similarity
            })

        except Exception as e:
            st.warning(f"Could not process with model {model_name}: {str(e)}")
            results_summary.append({
                'model': model_name,
                'verified': 'Error',
                'distance': 'N/A',
                'threshold': config["threshold"]
            })
        progress_bar_st.progress((i + 1) / num_chosen_models)
    
    status_text_st.text("Analysis complete!")

    if not aggregated_metrics['distance']: # No models ran successfully
        st.error("None of the selected models could produce a result.")
        return None

    # Calculate average results from successful runs
    successful_runs = len(aggregated_metrics['distance'])
    avg_distance = sum(aggregated_metrics['distance']) / successful_runs
    
    # For overall similarity, it's better to use consensus or an average of normalized distances.
    # Let's use consensus for "verified" and report average distance.
    # The "similarity" from your original code was `max(0, 100 - (distance * 100))`.
    # We can average this if desired, but it's a bit arbitrary for non-cosine distances.
    # Let's calculate an average similarity based on a normalized scale.
    # A simple way: average the 'verified' status (as 0 or 1) and scale to 100.
    # Or, if we want a "similarity score" similar to your original:
    avg_similarity_score = 0
    temp_similarities = []
    for r_sum in results_summary:
        if isinstance(r_sum['distance'], float):
            # This is a heuristic and might not be perfectly comparable across distance metrics
            # For cosine distance (0 to 2, 0 is identical): (1 - dist)
            # For Euclidean L2 (0 to large, 0 is identical): 1 / (1 + dist) or similar
            # Sticking to the provided formula for simplicity in aggregation:
            # Let's re-evaluate this. If a model says "verified", it's "similar".
            # The overall "similarity" could be the % of models that verified, or an average of individual similarities.

            # Let's try to derive a similarity from distance for each model, then average THAT.
            # For cosine: sim = 1 - dist.
            # For Euclidean: More complex. Let's use the consensus for a "similarity score".
            pass # We will use consensus for main similarity indicator

    consensus = aggregated_metrics['verified_count'] / successful_runs if successful_runs > 0 else 0
    
    # Overall "similarity percentage" can be the consensus percentage itself.
    overall_similarity_percent = consensus * 100


    return {
        'overall_similarity_percent': overall_similarity_percent, # Based on consensus
        'avg_distance': avg_distance,
        'consensus_ratio': consensus,
        'verified_count': aggregated_metrics['verified_count'],
        'models_attempted': num_chosen_models,
        'models_succeeded': successful_runs,
        'individual_results': results_summary # For detailed display
    }


if st.button("Compare Faces", type="primary", use_container_width=True):
    if not selected_model_names:
        st.warning("⚠️ Please select at least one model for comparison.")
    elif st.session_state.image1 and st.session_state.image2:
        with st.spinner("Preparing images and analyzing faces... This may take a moment."):
            # Create temporary files for DeepFace
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp1, \
                 tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp2:

                # Ensure images are in RGB format
                img1_processed = st.session_state.image1
                if img1_processed.mode == 'RGBA':
                    img1_processed = img1_processed.convert('RGB')
                img1_processed.save(tmp1.name, format='JPEG')
                img1_path = tmp1.name

                img2_processed = st.session_state.image2
                if img2_processed.mode == 'RGBA':
                    img2_processed = img2_processed.convert('RGB')
                img2_processed.save(tmp2.name, format='JPEG')
                img2_path = tmp2.name

            try:
                comparison_result = compare_faces_multimodel(img1_path, img2_path, selected_model_names)

                if comparison_result:
                    st.subheader("📊 Overall Comparison Result")

                    # Visual progress bar for overall similarity (consensus-based)
                    st.markdown(f"""
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {comparison_result['overall_similarity_percent']}%"></div>
                    </div>
                    """, unsafe_allow_html=True)

                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.metric("Overall Similarity (Consensus)",
                                  f"{comparison_result['overall_similarity_percent']:.1f}%")
                        st.metric("Average Distance (successful models)",
                                  f"{comparison_result['avg_distance']:.4f}")

                    with res_col2:
                        st.metric("Model Consensus",
                                  f"{comparison_result['consensus_ratio']*100:.0f}% ({comparison_result['verified_count']}/{comparison_result['models_succeeded']} models agree)")
                        
                        st.caption(f"Based on {comparison_result['models_succeeded']} out of {comparison_result['models_attempted']} selected models successfully processed.")

                        if comparison_result['models_succeeded'] == 0:
                             st.error("❌ No models could successfully compare the images.")
                        elif comparison_result['consensus_ratio'] >= 0.75: # e.g., 3 out of 4, or all if < 4
                            st.success("✅ Strong match consensus")
                        elif comparison_result['consensus_ratio'] >= 0.5:
                            st.warning("⚠️ Partial match consensus")
                        else:
                            st.error("❌ No match consensus")
                    
                    st.markdown("---")
                    st.subheader("🔬 Detailed Model Results")
                    
                    # Display individual model results in a more structured way
                    for res in comparison_result['individual_results']:
                        model_name = res['model']
                        status = "✅ Verified" if res['verified'] is True else ("❌ Not Verified" if res['verified'] is False else "⚠️ Error")
                        distance = f"{res['distance']:.4f}" if isinstance(res['distance'], float) else res['distance']
                        threshold = f"{res['threshold']:.2f}"

                        col_m, col_s, col_d, col_t = st.columns([2,2,1,1])
                        with col_m:
                            st.markdown(f"**{model_name}**")
                        with col_s:
                            if res['verified'] is True:
                                st.markdown(f"<span style='color:green'>{status}</span>", unsafe_allow_html=True)
                            elif res['verified'] is False:
                                st.markdown(f"<span style='color:red'>{status}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<span style='color:orange'>{status}</span>", unsafe_allow_html=True)
                        with col_d:
                            st.markdown(f"Dist: {distance}")
                        with col_t:
                            st.markdown(f"Thresh: {threshold}")
                        st.divider()


                    with st.expander("How to interpret these results"):
                        st.markdown("""
                        - **Overall Similarity (Consensus)**: Percentage of selected models that agree the faces are the same.
                        - **Average Distance**: The average distance score from models that ran successfully (lower is generally better).
                        - **Model Consensus**: Ratio and count of models that verified the faces as a match.
                        - **Detailed Model Results**:
                            - **Status**: Whether each model verified the faces as a match (✅), not a match (❌), or encountered an error (⚠️).
                            - **Dist**: The distance calculated by the model. Lower means more similar.
                            - **Thresh**: The threshold used by that model. If Distance < Threshold, it's a match.
                        - **Confidence Levels (based on Consensus)**:
                            - Strong Match: >= 75% of successful models agree.
                            - Partial Match: >= 50% and < 75% of successful models agree.
                            - No Match: < 50% of successful models agree.
                        """)

                else:
                    # Error messages are now handled within compare_faces_multimodel or if no models selected
                    if selected_model_names: # if models were selected but function returned None
                        st.error("Comparison could not be completed. Ensure faces are detectable in both images.")

            except Exception as e:
                st.error(f"An unexpected error occurred during comparison: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

            finally:
                # Clean up temporary files
                try:
                    os.unlink(img1_path)
                    os.unlink(img2_path)
                except Exception as e:
                    st.warning(f"Could not delete temporary files: {e}") # Non-critical
    else:
        st.warning("⚠️ Please upload both images first.")

# Information sections
with st.expander("About this app"):
    st.markdown(f"""
    **Multi-Model Face Similarity Comparison Tool**
    
    This application uses multiple deep learning models to compare two face images and determine their similarity. 
    You can select which models to use for the comparison.
    
    **Features:**
    - User-selectable face recognition models
    - Each model with pre-set thresholds (may require tuning for optimal performance)
    - Consensus-based verification system
    - Detailed per-model and overall results
    
    **Models Available:**
    { "".join([f"- {model_name}<br>" for model_name in MODELS.keys()]) }
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    ### Instructions
    1. Upload two face images.
    2. Select the face recognition models you want to use.
    3. Click "Compare Faces".
    4. View detailed similarity results.
    
    **Tips for best results:**
    - Use clear frontal face images.
    - Ensure good lighting conditions.
    - Avoid obstructions (e.g., sunglasses, heavy makeup if possible).
    - Maintain relatively neutral facial expressions.
    - Use reasonably high-resolution images.
    """)