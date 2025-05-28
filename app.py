import streamlit as st
import os
import cv2
from deepface import DeepFace
import numpy as np
from PIL import Image
import tempfile

# Konfigurasi halaman aplikasi Streamlit
st.set_page_config(
    page_title="Analisis & Perbandingan Wajah Tingkat Lanjut",
    page_icon="🔬",
    layout="wide"
)

# Daftar MODEL yang akan digunakan beserta konfigurasi threshold dan metrik jaraknya
MODELS = {
    "VGG-Face": {"threshold": 0.68, "distance_metric": "cosine"},
    "Facenet": {"threshold": 0.40, "distance_metric": "euclidean"},
    "Facenet512": {"threshold": 1.00, "distance_metric": "euclidean_l2"},
    "ArcFace": {"threshold": 0.68, "distance_metric": "cosine"},
    "Dlib": {"threshold": 0.07, "distance_metric": "euclidean_l2"},
    "SFace": {"threshold": 0.594, "distance_metric": "cosine"},
    "OpenFace": {"threshold": 0.80, "distance_metric": "euclidean_l2"},
}

# Daftar DETECTOR_BACKEND yang bisa dipilih pengguna (Dibatasi)
SUPPORTED_DETECTORS = ["opencv", "dlib", "mediapipe"]

# CSS Kustom
st.markdown("""
<style>
    .stApp { max-width: 1300px; margin: 0 auto; }
    .progress-container { height: 20px; background-color: #e9ecef; border-radius: 10px; margin: 1rem 0; }
    .progress-bar { height: 100%; border-radius: 10px; background-color: #4CAF50; width: 0%; transition: width 0.5s ease; }
</style>
""", unsafe_allow_html=True)

# Judul utama aplikasi
st.title("🔬 Analisis & Perbandingan Wajah Tingkat Lanjut")
st.markdown("""
Unggah dua gambar wajah untuk membandingkan kemiripannya menggunakan model dan detektor wajah pilihan.
Anda juga dapat menganalisis atribut wajah individual dan menyesuaikan ambang batas (threshold) model.
""")

# --- Inisialisasi Session State ---
default_session_states = {
    'image1_original': None, 'image1_display': None, 'image1_analysis': None, 'image1_face_count': 0, 'img1_current_filename': None,
    'image2_original': None, 'image2_display': None, 'image2_analysis': None, 'image2_face_count': 0, 'img2_current_filename': None,
    'custom_thresholds': {},
    'selected_detector_backend': SUPPORTED_DETECTORS[0]
}
for key, value in default_session_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Fungsi-Fungsi Pembantu ---
def load_image(image_file):
    """Memuat file gambar menjadi objek PIL.Image."""
    if image_file is not None:
        try: img = Image.open(image_file); return img
        except Exception as e: st.error(f"Gagal memuat gambar: {str(e)}"); return None
    return None

def get_image_with_bounding_boxes(pil_image, detector_backend):
    """Mendeteksi wajah dalam gambar PIL dan mengembalikan gambar dengan kotak pembatas (bounding box) 
       berwarna hijau serta jumlah wajah yang terdeteksi."""
    if pil_image is None: return None, 0
    img_cv = np.array(pil_image.convert('RGB')); img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    faces_detected_count = 0; temp_file_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            pil_image.convert('RGB').save(tmp_file.name, format='JPEG'); temp_file_path = tmp_file.name
        extracted_faces_info = DeepFace.extract_faces(
            img_path=temp_file_path, detector_backend=detector_backend, enforce_detection=False, align=False, silent=True
        )
        for face_info in extracted_faces_info:
            if isinstance(face_info, dict) and 'facial_area' in face_info and face_info.get('confidence', 0) > 0.1:
                fx, fy, fw, fh = face_info['facial_area']['x'], face_info['facial_area']['y'], \
                                 face_info['facial_area']['w'], face_info['facial_area']['h']
                # Gambar kotak HIJAU (0, 255, 0) di BGR
                cv2.rectangle(img_cv, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2) 
                faces_detected_count += 1
        img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_cv_rgb), faces_detected_count
    except Exception as e:
        st.warning(f"Deteksi wajah dengan '{detector_backend}' gagal: {e}. Menampilkan gambar asli.")
        print(f"Error di get_image_with_bounding_boxes (backend: {detector_backend}): {e}")
        img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_cv_rgb), 0
    finally:
        if temp_file_path and os.path.exists(temp_file_path): os.unlink(temp_file_path)

def analyze_face_attributes_func(pil_image, detector_backend):
    """Menganalisis atribut wajah (usia, gender, emosi, ras) dari gambar PIL."""
    if pil_image is None: return None
    temp_file_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            pil_image.convert('RGB').save(tmp_file.name, format='JPEG'); temp_file_path = tmp_file.name
        analysis_result = DeepFace.analyze(
            img_path=temp_file_path, actions=['age', 'gender', 'emotion', 'race'],
            enforce_detection=True, detector_backend=detector_backend, silent=True
        )
        if isinstance(analysis_result, list) and len(analysis_result) > 0: return analysis_result[0]
        elif isinstance(analysis_result, dict): return analysis_result # Kompatibilitas dengan DeepFace lama
        return {"error": "Tidak ada wajah terdeteksi atau analisis gagal."}
    except Exception as e:
        st.warning(f"Analisis atribut dengan '{detector_backend}' gagal: {e}")
        print(f"Error di analyze_face_attributes_func (backend: {detector_backend}): {e}")
        return {"error": str(e)}
    finally:
        if temp_file_path and os.path.exists(temp_file_path): os.unlink(temp_file_path)

# --- UI Pilihan Global: Detector Backend ---
st.sidebar.subheader("⚙️ Pengaturan Detektor Wajah")
current_detector = st.session_state.get('selected_detector_backend', SUPPORTED_DETECTORS[0])
if current_detector not in SUPPORTED_DETECTORS:
    current_detector = SUPPORTED_DETECTORS[0]
    st.session_state.selected_detector_backend = current_detector

selected_detector = st.sidebar.selectbox(
    "Pilih Backend Detektor Wajah:",
    options=SUPPORTED_DETECTORS,
    index=SUPPORTED_DETECTORS.index(current_detector),
    key="detector_selector",
    help="Backend yang berbeda memiliki kecepatan dan akurasi yang berbeda. Pastikan dependensi terinstal."
)
if selected_detector != st.session_state.selected_detector_backend:
    st.session_state.selected_detector_backend = selected_detector
    st.session_state.image1_display = None
    st.session_state.image2_display = None
    st.rerun()


# --- UI untuk Unggah Gambar dan Analisis Atribut ---
col1_img, col2_img = st.columns(2)

# --- Penanganan Gambar 1 ---
with col1_img:
    st.subheader("Gambar Pertama")
    uploaded_file_obj1 = st.file_uploader("Unggah gambar wajah pertama", type=["jpg", "jpeg", "png"], key="img1_uploader")
    if uploaded_file_obj1 is None:
        if st.session_state.img1_current_filename is not None:
            st.session_state.image1_original = None; st.session_state.image1_display = None
            st.session_state.image1_analysis = None; st.session_state.image1_face_count = 0
            st.session_state.img1_current_filename = None
    else:
        if st.session_state.img1_current_filename != uploaded_file_obj1.name or st.session_state.image1_original is None:
            st.session_state.image1_original = load_image(uploaded_file_obj1)
            st.session_state.img1_current_filename = uploaded_file_obj1.name
            st.session_state.image1_display = None; st.session_state.image1_analysis = None
            st.session_state.image1_face_count = 0
        if st.session_state.image1_original:
            if st.session_state.image1_display is None:
                with st.spinner(f"Mendeteksi wajah di Gambar 1 (detektor: {st.session_state.selected_detector_backend})..."):
                    st.session_state.image1_display, st.session_state.image1_face_count = \
                        get_image_with_bounding_boxes(st.session_state.image1_original, st.session_state.selected_detector_backend)
            if st.session_state.image1_display:
                st.image(st.session_state.image1_display, caption=f"Gambar 1 ({st.session_state.image1_face_count} wajah terdeteksi)", use_container_width=True)
            if st.button("Analisis Atribut - Gambar 1", key="analyze1_btn", use_container_width=True):
                if st.session_state.image1_original:
                    with st.spinner(f"Menganalisis atribut Gambar 1 (detektor: {st.session_state.selected_detector_backend})..."):
                        st.session_state.image1_analysis = analyze_face_attributes_func(st.session_state.image1_original, st.session_state.selected_detector_backend)
                else: st.warning("Harap unggah Gambar 1 untuk menganalisis atribut.")
    if st.session_state.image1_analysis:
        with st.container(border=True):
            if "error" in st.session_state.image1_analysis: st.error(f"Error Analisis Gbr 1: {st.session_state.image1_analysis['error']}")
            else:
                st.markdown("**Atribut Gambar 1:**"); res = st.session_state.image1_analysis
                attr_colA, attr_colB = st.columns(2)
                with attr_colA:
                    st.metric("Perk. Usia", f"{res.get('age', 'N/A')}")
                    dom_gender = max(res.get('gender', {}), key=res.get('gender', {}).get, default='N/A')
                    st.metric("Gender", dom_gender)
                with attr_colB:
                    st.metric("Emosi", f"{res.get('dominant_emotion', 'N/A').capitalize()}")
                    st.metric("Ras", f"{res.get('dominant_race', 'N/A').capitalize()}")

# --- Penanganan Gambar 2 ---
with col2_img:
    st.subheader("Gambar Kedua")
    uploaded_file_obj2 = st.file_uploader("Unggah gambar wajah kedua", type=["jpg", "jpeg", "png"], key="img2_uploader")
    if uploaded_file_obj2 is None:
        if st.session_state.img2_current_filename is not None:
            st.session_state.image2_original = None; st.session_state.image2_display = None
            st.session_state.image2_analysis = None; st.session_state.image2_face_count = 0
            st.session_state.img2_current_filename = None
    else:
        if st.session_state.img2_current_filename != uploaded_file_obj2.name or st.session_state.image2_original is None:
            st.session_state.image2_original = load_image(uploaded_file_obj2)
            st.session_state.img2_current_filename = uploaded_file_obj2.name
            st.session_state.image2_display = None; st.session_state.image2_analysis = None
            st.session_state.image2_face_count = 0
        if st.session_state.image2_original:
            if st.session_state.image2_display is None:
                with st.spinner(f"Mendeteksi wajah di Gambar 2 (detektor: {st.session_state.selected_detector_backend})..."):
                    st.session_state.image2_display, st.session_state.image2_face_count = \
                        get_image_with_bounding_boxes(st.session_state.image2_original, st.session_state.selected_detector_backend)
            if st.session_state.image2_display:
                st.image(st.session_state.image2_display, caption=f"Gambar 2 ({st.session_state.image2_face_count} wajah terdeteksi)", use_container_width=True)
            if st.button("Analisis Atribut - Gambar 2", key="analyze2_btn", use_container_width=True):
                if st.session_state.image2_original:
                    with st.spinner(f"Menganalisis atribut Gambar 2 (detektor: {st.session_state.selected_detector_backend})..."):
                        st.session_state.image2_analysis = analyze_face_attributes_func(st.session_state.image2_original, st.session_state.selected_detector_backend)
                else: st.warning("Harap unggah Gambar 2 untuk menganalisis atribut.")
    if st.session_state.image2_analysis:
        with st.container(border=True):
            if "error" in st.session_state.image2_analysis: st.error(f"Error Analisis Gbr 2: {st.session_state.image2_analysis['error']}")
            else:
                st.markdown("**Atribut Gambar 2:**"); res = st.session_state.image2_analysis
                attr_colA, attr_colB = st.columns(2)
                with attr_colA:
                    st.metric("Perk. Usia", f"{res.get('age', 'N/A')}")
                    dom_gender = max(res.get('gender', {}), key=res.get('gender', {}).get, default='N/A')
                    st.metric("Gender", dom_gender)
                with attr_colB:
                    st.metric("Emosi", f"{res.get('dominant_emotion', 'N/A').capitalize()}")
                    st.metric("Ras", f"{res.get('dominant_race', 'N/A').capitalize()}")
st.markdown("---")

# --- UI untuk Pemilihan SATU Model dan Penyesuaian Threshold ---
st.subheader("⚙️ Konfigurasi Model Perbandingan")
selected_model_name = st.selectbox(
    "Pilih model untuk perbandingan:",
    options=list(MODELS.keys()),
    index=0,
    key="model_selector"
)

if selected_model_name:
    if selected_model_name not in st.session_state.custom_thresholds:
        st.session_state.custom_thresholds[selected_model_name] = MODELS[selected_model_name]["threshold"]
    
    with st.expander(f"Sesuaikan Threshold untuk {selected_model_name} (Opsional)", expanded=True):
        default_model_threshold = MODELS[selected_model_name]["threshold"]
        current_user_threshold = st.session_state.custom_thresholds.get(selected_model_name, default_model_threshold)
        step_val, min_val = 0.01, 0.0
        if MODELS[selected_model_name]["distance_metric"] == "cosine": max_val = 1.0
        elif MODELS[selected_model_name]["distance_metric"] == "euclidean": max_val = 50.0 # Contoh rentang
        elif MODELS[selected_model_name]["distance_metric"] == "euclidean_l2": max_val = 2.0 # Contoh rentang
        else: max_val = default_model_threshold * 3 if default_model_threshold > 0.1 else 1.0
        safe_slider_value = max(min_val, min(current_user_threshold, max_val))
        new_threshold = st.slider(
            f"Threshold untuk {selected_model_name} (Metrik: {MODELS[selected_model_name]['distance_metric']})",
            min_value=min_val, max_value=max_val, value=safe_slider_value, step=step_val,
            key=f"thresh_single_{selected_model_name}", # Kunci unik
            help=f"Lebih rendah berarti lebih ketat. Default: {default_model_threshold:.3f}. Jika jarak < threshold, maka cocok."
        )
        st.session_state.custom_thresholds[selected_model_name] = new_threshold
st.markdown("---")

# --- Logika Perbandingan (HANYA SATU MODEL) dan Tampilan Hasil ---
def compare_face_single_model(img1_path, img2_path, model_name_to_use, config_for_model, detector_backend):
    """Membandingkan dua wajah menggunakan SATU model dengan konfigurasi threshold yang diberikan."""
    try:
        result = DeepFace.verify(
            img1_path=img1_path, img2_path=img2_path, model_name=model_name_to_use,
            distance_metric=config_for_model["distance_metric"],
            enforce_detection=True, detector_backend=detector_backend, silent=True
        )
        distance = float(result['distance'])
        verified = distance < config_for_model["threshold"]
        return {'model': model_name_to_use, 'verified': verified, 'distance': distance,
                'threshold': config_for_model["threshold"], 'error': None}
    except Exception as e:
        st.warning(f"Tidak dapat memproses dengan model {model_name_to_use} (detektor: {detector_backend}): {str(e)}")
        print(f"--- ERROR TRACEBACK: {model_name_to_use}, {detector_backend} ---")
        import traceback; traceback.print_exc(); print(f"--- AKHIR TRACEBACK ---")
        return {'model': model_name_to_use, 'verified': 'Error', 'distance': 'N/A',
                'threshold': config_for_model["threshold"], 'error': str(e)}

if st.button("Bandingkan Wajah dengan Model Terpilih", type="primary", use_container_width=True, key="compare_single_btn"):
    if not selected_model_name: st.warning("⚠️ Harap pilih model untuk perbandingan.")
    elif st.session_state.image1_original and st.session_state.image2_original:
        active_model_config = {
            "threshold": st.session_state.custom_thresholds.get(selected_model_name, MODELS[selected_model_name]["threshold"]),
            "distance_metric": MODELS[selected_model_name]["distance_metric"]
        }
        current_detector_for_comparison = st.session_state.selected_detector_backend
        with st.spinner(f"Menyiapkan gambar dan menganalisis dengan {selected_model_name} (detektor: {current_detector_for_comparison})..."):
            img1_path, img2_path = "", ""
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp1, \
                     tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp2:
                    st.session_state.image1_original.convert('RGB').save(tmp1.name, format='JPEG'); img1_path = tmp1.name
                    st.session_state.image2_original.convert('RGB').save(tmp2.name, format='JPEG'); img2_path = tmp2.name
                
                comparison_result_single = compare_face_single_model(
                    img1_path, img2_path, selected_model_name, active_model_config, current_detector_for_comparison
                )
                if comparison_result_single:
                    st.subheader(f"📊 Hasil Perbandingan dengan Model: {selected_model_name}")
                    res = comparison_result_single
                    if res['error']: st.error(f"Gagal memproses: {res['error']}")
                    else:
                        similarity_percent_visual = 0
                        if MODELS[selected_model_name]["distance_metric"] == "cosine" and isinstance(res['distance'], float):
                            similarity_percent_visual = max(0, (1 - res['distance']) * 100)
                        elif isinstance(res['distance'], float) and res['threshold'] > 0:
                            scale_factor = res['threshold'] * 1.5 
                            similarity_percent_visual = max(0, 100 * (1 - (min(res['distance'], scale_factor) / scale_factor)))
                        else: similarity_percent_visual = 100 if res['verified'] else 0
                        st.markdown(f"""<div class="progress-container"><div class="progress-bar" style="width: {similarity_percent_visual:.0f}%"></div></div>""", unsafe_allow_html=True)
                        status_emoji = "✅" if res['verified'] else "❌"
                        status_text = "Terverifikasi (Wajah Sama)" if res['verified'] else "Tidak Terverifikasi (Wajah Berbeda)"
                        col_res1, col_res2, col_res3 = st.columns(3)
                        with col_res1: st.metric("Status", f"{status_emoji} {status_text}")
                        with col_res2: st.metric("Jarak", f"{res['distance']:.4f}" if isinstance(res['distance'], float) else res['distance'])
                        with col_res3: st.metric("Threshold Digunakan", f"{res['threshold']:.3f}")
                        if res['verified']: st.success(f"Menurut model {selected_model_name}, kedua wajah **cocok**.")
                        else: st.error(f"Menurut model {selected_model_name}, kedua wajah **tidak cocok**.")
                else: st.error("Perbandingan tidak dapat diselesaikan.")
            except Exception as e:
                st.error(f"Terjadi kesalahan tak terduga: {str(e)}"); import traceback; st.error(traceback.format_exc())
            finally:
                if img1_path and os.path.exists(img1_path): os.unlink(img1_path)
                if img2_path and os.path.exists(img2_path): os.unlink(img2_path)
    else:
        st.warning("⚠️ Harap unggah kedua gambar terlebih dahulu.")

# --- Footer dan Informasi Tambahan ---
st.markdown("---")
with st.expander("Tentang Aplikasi Ini & Model yang Digunakan", expanded=False):
    st.markdown("#### Analisis & Perbandingan Wajah Tingkat Lanjut")
    st.markdown("Aplikasi ini memungkinkan Anda membandingkan dua wajah menggunakan model AI terpilih, menganalisis atribut wajah, dan menyesuaikan parameter.")
    st.markdown("**Model yang Tersedia untuk Perbandingan:**")
    for model_name_loop, config_loop in MODELS.items():
        st.markdown(f"- **{model_name_loop}**: Metrik: `{config_loop['distance_metric']}`, Default Threshold: `{config_loop['threshold']:.3f}`")
    st.markdown(f"\n**Backend Detektor Wajah yang Digunakan:** `{st.session_state.selected_detector_backend}`")

with st.sidebar:
    # st.image("URL_IKON_ANDA_DISINI", width=80) # Ganti dengan URL ikon yang sesuai jika ada
    st.markdown("### Instruksi Penggunaan"); st.markdown("""1. **Pilih Detektor Wajah** (di sidebar kiri).\n2. **Unggah Gambar** 1 & 2.\n3. **(Opsional) Analisis Atribut**.\n4. **Pilih Model Perbandingan**.\n5. **(Opsional) Sesuaikan Threshold**.\n6. **Bandingkan Wajah**.\n7. **Lihat Hasil**.""")
    st.markdown("### Tips"); st.markdown("- Gunakan gambar wajah jelas & frontal.\n- Pencahayaan baik membantu.\n- Resolusi lebih tinggi lebih baik.")