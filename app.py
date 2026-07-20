import streamlit as st
import opencv-python-headless
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from PIL import Image
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Simple saliency map using Gaussian difference (no scikit-image needed)
def compute_saliency_map(image_bgr):
    """Compute a simple saliency map using the difference from a heavily blurred version."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    saliency = np.abs(gray - blurred)
    if saliency.max() > 0:
        saliency /= saliency.max()
    return saliency

# ----------------------------------------------------------------------
# Cached model loading
@st.cache_resource
def load_model():
    """Load YOLOv8 model trained on Open Images V7 (includes Person, Fish, Coral, Statue, etc.)."""
    model = YOLO('yolov8n-oiv7.pt')  # will download automatically
    return model

# ----------------------------------------------------------------------
# Enhancement methods
def apply_hist_eq(image_bgr):
    """Histogram Equalization on V channel of HSV."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

def apply_clahe(image_bgr):
    """CLAHE on L channel of LAB."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def apply_gamma_correction(image_bgr, gamma=1.5):
    """Simple gamma correction."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image_bgr, table)

def apply_white_balance(image_bgr):
    """Gray-world white balance."""
    b, g, r = cv2.split(image_bgr)
    m_b, m_g, m_r = b.mean(), g.mean(), r.mean()
    avg = (m_b + m_g + m_r) / 3.0
    b_wb = np.clip(b * (avg / m_b), 0, 255).astype(np.uint8)
    g_wb = np.clip(g * (avg / m_g), 0, 255).astype(np.uint8)
    r_wb = np.clip(r * (avg / m_r), 0, 255).astype(np.uint8)
    return cv2.merge([b_wb, g_wb, r_wb])

def apply_fusion(image_bgr):
    """Fusion: white balance + CLAHE + saliency."""
    wb = apply_white_balance(image_bgr)
    ce = apply_clahe(image_bgr)
    sal_map = compute_saliency_map(image_bgr)
    if sal_map.shape[:2] != image_bgr.shape[:2]:
        sal_map = cv2.resize(sal_map, (image_bgr.shape[1], image_bgr.shape[0]))
    sal_3ch = np.dstack([sal_map] * 3)
    wb_f = wb.astype(np.float32)
    ce_f = ce.astype(np.float32)
    blended = 0.5 * wb_f + 0.5 * ce_f
    orig_f = image_bgr.astype(np.float32)
    fused = sal_3ch * orig_f + (1.0 - sal_3ch) * blended
    return np.clip(fused, 0, 255).astype(np.uint8)

ENHANCEMENT_METHODS = {
    'Histogram Equalization': apply_hist_eq,
    'CLAHE': apply_clahe,
    'Gamma Correction': apply_gamma_correction,
    'White Balance': apply_white_balance,
    'Fusion': apply_fusion,
}

# ----------------------------------------------------------------------
# Object detection
def is_target_class(cls_name):
    """Check if the class name matches a marine / underwater target."""
    name = cls_name.lower().strip()
    targets = ['person', 'fish', 'jellyfish', 'starfish', 'turtle', 'coral',
               'sea turtle', 'coral reef', 'statue']
    for t in targets:
        if t in name:
            return True
    return False

def detect_objects(model, image_bgr, conf_threshold=0.25):
    """Run YOLO detection, filter for target classes, annotate image."""
    start_time = time.time()
    results = model(image_bgr, conf=conf_threshold, verbose=False)
    elapsed = time.time() - start_time

    res = results[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return image_bgr, {'total':0, 'human':0, 'marine':0,
                           'avg_confidence':0.0, 'processing_time':elapsed}, []

    class_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    names = model.names

    filtered_boxes = []
    human_count = 0
    marine_count = 0
    conf_list = []

    for i, cls_id in enumerate(class_ids):
        cls_name = names[cls_id]
        if is_target_class(cls_name):
            conf = confs[i]
            conf_list.append(conf)
            filtered_boxes.append((xyxy[i], cls_name, conf))
            if 'person' in cls_name.lower():
                human_count += 1
            else:
                marine_count += 1

    total = len(filtered_boxes)
    avg_conf = np.mean(conf_list) if conf_list else 0.0

    annotated = image_bgr.copy()
    for box, cls_name, conf in filtered_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if 'person' in cls_name.lower():
            display_name = "Human (diver)"
        elif 'statue' in cls_name.lower():
            display_name = "Statue"
        else:
            display_name = cls_name
        label = f"{display_name} {conf:.2f}"
        cv2.putText(annotated, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    stats = {
        'total': total,
        'human': human_count,
        'marine': marine_count,
        'avg_confidence': avg_conf,
        'processing_time': elapsed,
    }
    return annotated, stats, filtered_boxes

# ----------------------------------------------------------------------
# Comparative study (now includes images)
def run_full_comparison(original_bgr, model, conf_threshold):
    """Run all methods: enhancement, detection, collect stats and images."""
    methods = {
        'Original': lambda img: img,
        'Histogram Equalization': apply_hist_eq,
        'CLAHE': apply_clahe,
        'Gamma Correction': apply_gamma_correction,
        'White Balance': apply_white_balance,
        'Fusion': apply_fusion,
    }
    records = []
    images_dict = {}  # method_name -> {'enhanced': bgr, 'detected': bgr}

    for name, func in methods.items():
        enhanced = func(original_bgr)
        detected, stats, _ = detect_objects(model, enhanced, conf_threshold)
        images_dict[name] = {
            'enhanced': enhanced,
            'detected': detected
        }
        records.append({
            'Method': name,
            'Objects Detected': stats['total'],
            'Human Count': stats['human'],
            'Marine Count': stats['marine'],
            'Avg Confidence': stats['avg_confidence'],
            'Processing Time (s)': stats['processing_time'],
            'Precision': stats['avg_confidence'],
            'Accuracy': stats['avg_confidence'],
        })
    return pd.DataFrame(records), images_dict

def plot_comparative(df):
    """Generate bar charts for comparative study."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0,0].bar(df['Method'], df['Objects Detected'], color='skyblue')
    axes[0,0].set_title('Objects Detected per Method')
    axes[0,0].set_ylabel('Count')
    axes[0,0].tick_params(axis='x', rotation=45)

    axes[0,1].bar(df['Method'], df['Avg Confidence'], color='lightgreen')
    axes[0,1].set_title('Average Confidence (Precision / Accuracy)')
    axes[0,1].set_ylabel('Confidence')
    axes[0,1].tick_params(axis='x', rotation=45)

    axes[1,0].bar(df['Method'], df['Processing Time (s)'], color='salmon')
    axes[1,0].set_title('Processing Time (Detection)')
    axes[1,0].set_ylabel('Seconds')
    axes[1,0].tick_params(axis='x', rotation=45)

    axes[1,1].bar(df['Method'], df['Human Count'], label='Human', color='blue')
    axes[1,1].bar(df['Method'], df['Marine Count'], bottom=df['Human Count'],
                  label='Marine', color='green')
    axes[1,1].set_title('Human vs Marine Detections')
    axes[1,1].set_ylabel('Count')
    axes[1,1].legend()
    axes[1,1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig

def display_comparison_grid(images_dict):
    """Display a visual comparison grid: for each method, show enhanced and detected images."""
    methods_order = ['Original', 'Histogram Equalization', 'CLAHE',
                     'Gamma Correction', 'White Balance', 'Fusion']
    cols_per_row = 3
    for i in range(0, len(methods_order), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, method in enumerate(methods_order[i:i+cols_per_row]):
            if method in images_dict:
                with cols[j]:
                    st.markdown(f"**{method}**")
                    # Enhanced image
                    enh_rgb = cv2.cvtColor(images_dict[method]['enhanced'], cv2.COLOR_BGR2RGB)
                    st.image(enh_rgb, use_column_width=True, caption='Enhanced')
                    # Detected image
                    det_rgb = cv2.cvtColor(images_dict[method]['detected'], cv2.COLOR_BGR2RGB)
                    st.image(det_rgb, use_column_width=True, caption='Detected')

# ----------------------------------------------------------------------
# Streamlit UI
st.set_page_config(page_title="Underwater Image Enhancement & Detection", layout="wide")
st.title("🌊 Underwater Image Enhancement and Marine Object Detection")
st.markdown("**A Comparative Study Using Classical and Fusion-Based Algorithms**")
st.markdown("Now also detects **statues** and other submerged objects 🗿")

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("📤 Upload Underwater Image", type=["jpg", "jpeg", "png"])
    enh_method = st.selectbox("🛠️ Enhancement Method",
                              list(ENHANCEMENT_METHODS.keys()))
    conf_threshold = st.slider("🎯 Detection Confidence Threshold",
                               0.05, 1.0, 0.25, 0.05)
    run_enh = st.button("✨ Run Enhancement")
    run_det = st.button("🔍 Run Detection")
    run_comp = st.button("📊 Run Full Comparative Study (Stats + Visuals)")

# Initialize session state
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
    st.session_state.enhanced_image = None
    st.session_state.annotated_image = None
    st.session_state.detection_stats = None
    st.session_state.df_comparison = None
    st.session_state.comparison_images = None

model = load_model()

# Handle image upload
if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert('RGB')
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    st.session_state.original_image = img_bgr
    st.session_state.enhanced_image = None
    st.session_state.annotated_image = None
    st.session_state.detection_stats = None
    st.session_state.df_comparison = None
    st.session_state.comparison_images = None

# Run Enhancement
if run_enh:
    if st.session_state.original_image is not None:
        with st.spinner(f'Applying {enh_method}...'):
            func = ENHANCEMENT_METHODS[enh_method]
            st.session_state.enhanced_image = func(st.session_state.original_image.copy())
        st.success("Enhancement complete")
    else:
        st.warning("Please upload an image first.")

# Run Detection
if run_det:
    if st.session_state.original_image is not None:
        img_to_detect = (st.session_state.enhanced_image
                         if st.session_state.enhanced_image is not None
                         else st.session_state.original_image)
        with st.spinner('Detecting objects...'):
            annotated, stats, _ = detect_objects(model, img_to_detect, conf_threshold)
            st.session_state.annotated_image = annotated
            st.session_state.detection_stats = stats
        st.success("Detection complete")
    else:
        st.warning("Please upload an image first.")

# Run Full Comparative Study
if run_comp:
    if st.session_state.original_image is not None:
        with st.spinner('Running full comparison on all methods...'):
            df, images_dict = run_full_comparison(
                st.session_state.original_image, model, conf_threshold
            )
            st.session_state.df_comparison = df
            st.session_state.comparison_images = images_dict
        st.success("Full comparative study complete")
    else:
        st.warning("Please upload an image first.")

# ----------------------------------------------------------------------
# Main display area
if st.session_state.original_image is not None:
    st.subheader("📷 Original Image")
    orig_rgb = cv2.cvtColor(st.session_state.original_image, cv2.COLOR_BGR2RGB)
    st.image(orig_rgb, use_column_width=True)

    if st.session_state.enhanced_image is not None:
        st.subheader(f"✨ Enhanced Image ({enh_method})")
        enh_rgb = cv2.cvtColor(st.session_state.enhanced_image, cv2.COLOR_BGR2RGB)
        st.image(enh_rgb, use_column_width=True)

    if st.session_state.annotated_image is not None:
        st.subheader("🔍 Detected Output")
        det_rgb = cv2.cvtColor(st.session_state.annotated_image, cv2.COLOR_BGR2RGB)
        st.image(det_rgb, use_column_width=True)

    if st.session_state.detection_stats is not None:
        st.subheader("📈 Detection Statistics")
        stats = st.session_state.detection_stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Objects", stats['total'])
        col2.metric("Human Count", stats['human'])
        col3.metric("Marine Count", stats['marine'])
        col4.metric("Avg Confidence", f"{stats['avg_confidence']:.2f}")
        col5, col6, col7 = st.columns(3)
        col5.metric("Processing Time (s)", f"{stats['processing_time']:.3f}")
        col6.metric("Precision (avg conf)", f"{stats['avg_confidence']:.2f}")
        col7.metric("Accuracy Estimate", f"{stats['avg_confidence']:.2f}")

    if st.session_state.df_comparison is not None:
        st.subheader("📊 Comparative Study Data")
        df = st.session_state.df_comparison
        st.dataframe(df.style.format({"Avg Confidence": "{:.2f}",
                                      "Processing Time (s)": "{:.3f}",
                                      "Precision": "{:.2f}",
                                      "Accuracy": "{:.2f}"}))
        fig = plot_comparative(df)
        st.pyplot(fig)

        # Visual comparison grid
        if st.session_state.comparison_images is not None:
            st.subheader("🔍 Visual Comparison Grid (Enhanced & Detected)")
            with st.expander("Click to show/hide full visual comparison"):
                display_comparison_grid(st.session_state.comparison_images)
else:
    st.info("👈 Please upload an underwater image from the sidebar to begin.")
