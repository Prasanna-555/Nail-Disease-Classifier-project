import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import tempfile
from fpdf import FPDF
from datetime import datetime
import pandas as pd
import io
import numpy as np
import os

st.set_page_config(page_title="Nail Disease Classifier 💅", page_icon="💅", layout="wide", initial_sidebar_state="expanded")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Enhanced CSS with animations and modern design
if st.session_state.dark_mode:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Poppins:wght@500;700&display=swap');

    body, .reportview-container, .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1c2526 100%);
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
        overflow-x: hidden;
    }
    .sidebar .sidebar-content {
        background: #1c2526;
        border-right: 1px solid #2e2e2e;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .header {
        background: linear-gradient(90deg, #4b0082, #7c4dff);
        padding: 30px;
        border-radius: 16px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        animation: fadeIn 1s ease-in-out;
    }
    .header h1 {
        color: white;
        margin: 0;
        font-family: 'Poppins', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .prediction-card {
        background: #1c2526;
        padding: 30px;
        margin-bottom: 25px;
        border-radius: 16px;
        border: 2px solid transparent;
        background-clip: padding-box;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.5);
        border-color: #7c4dff;
    }
    .confidence-bar {
        height: 14px;
        border-radius: 7px;
        background-color: #2e2e2e;
        margin: 15px 0;
        overflow: hidden;
        position: relative;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 7px;
        background: linear-gradient(90deg, #6200ea, #7c4dff);
        transition: width 0.8s ease-in-out;
    }
    .prediction-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 12px;
        letter-spacing: 0.01em;
        display: flex;
        align-items: center;
    }
    .prediction-title::before {
        content: '💅';
        margin-right: 10px;
        font-size: 1.2rem;
    }
    .precautions-text {
        font-size: 1rem;
        line-height: 1.8;
        margin-left: 15px;
        color: #b0bec5;
    }
    .stButton > button {
        background: linear-gradient(90deg, #4b0082, #7c4dff);
        color: white;
        border-radius: 12px;
        padding: 14px 28px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        border: none;
        transition: background 0.3s, transform 0.2s, box-shadow 0.3s;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #7c4dff, #b388ff);
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.4);
    }
    .stFileUploader > div > div > div {
        border: 3px dashed #444;
        border-radius: 16px;
        background: #1c2526;
        padding: 25px;
        transition: border-color 0.3s, transform 0.3s;
        animation: pulse 2s infinite;
    }
    .stFileUploader > div > div > div:hover {
        border-color: #7c4dff;
        transform: scale(1.02);
    }
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .footer {
        text-align: center;
        padding: 30px;
        color: #b0bec5;
        font-size: 0.95rem;
        margin-top: 50px;
        border-top: 1px solid #2e2e2e;
        display: flex;
        justify-content: center;
        gap: 20px;
    }
    .footer a {
        color: #7c4dff;
        text-decoration: none;
        transition: color 0.3s;
    }
    .footer a:hover {
        color: #b388ff;
    }
    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #7c4dff;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.01); }
        100% { transform: scale(1); }
    }
    .css-1lcbmhc.e1fqkh3o3 > div:nth-child(2) {
        min-width: 500px !important;
        padding-left: 60px;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Poppins:wght@500;700&display=swap');

    body, .reportview-container, .main {
        background: linear-gradient(135deg, #f7f9fc 0%, #e0e7ff 100%);
        color: #1a1a1a;
        font-family: 'Inter', sans-serif;
        overflow-x: hidden;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
        border-right: 1px solid #e0e7ff;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .header {
        background: linear-gradient(90deg, #4b0082, #7c4dff);
        padding: 30px;
        border-radius: 16px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        animation: fadeIn 1s ease-in-out;
    }
    .header h1 {
        color: white;
        margin: 0;
        font-family: 'Poppins', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .prediction-card {
        background: #ffffff;
        padding: 30px;
        margin-bottom: 25px;
        border-radius: 16px;
        border: 2px solid transparent;
        background-clip: padding-box;
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.15);
        border-color: #7c4dff;
    }
    .confidence-bar {
        height: 14px;
        border-radius: 7px;
        background-color: #e0e7ff;
        margin: 15px 0;
        overflow: hidden;
        position: relative;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 7px;
        background: linear-gradient(90deg, #4b0082, #7c4dff);
        transition: width 0.8s ease-in-out;
    }
    .prediction-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 12px;
        letter-spacing: 0.01em;
        display: flex;
        align-items: center;
    }
    .prediction-title::before {
        content: '💅';
        margin-right: 10px;
        font-size: 1.2rem;
    }
    .precautions-text {
        font-size: 1rem;
        line-height: 1.8;
        margin-left: 15px;
        color: #4b5e6a;
    }
    .stButton > button {
        background: linear-gradient(90deg, #4b0082, #7c4dff);
        color: white;
        border-radius: 12px;
        padding: 14px 28px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        border: none;
        transition: background 0.3s, transform 0.2s, box-shadow 0.3s;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #7c4dff, #b388ff);
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.3);
    }
    .stFileUploader > div > div > div {
        border: 3px dashed #c7d2fe;
        border-radius: 16px;
        background: #ffffff;
        padding: 25px;
        transition: border-color 0.3s, transform 0.3s;
        animation: pulse 2s infinite;
    }
    .stFileUploader > div > div > div:hover {
        border-color: #7c4dff;
        transform: scale(1.02);
    }
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .footer {
        text-align: center;
        padding: 30px;
        color: #4b5e6a;
        font-size: 0.95rem;
        margin-top: 50px;
        border-top: 1px solid #e0e7ff;
        display: flex;
        justify-content: center;
        gap: 20px;
    }
    .footer a {
        color: #7c4dff;
        text-decoration: none;
        transition: color 0.3s;
    }
    .footer a:hover {
        color: #b388ff;
    }
    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #7c4dff;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.01); }
        100% { transform: scale(1); }
    }
    .css-1lcbmhc.e1fqkh3o3 > div:nth-child(2) {
        min-width: 500px !important;
        padding-left: 60px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar with theme toggle and language selection
st.sidebar.markdown("### Settings ⚙")
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode, on_change=toggle_theme)
lang_options = {"English": "en", "Hindi": "hi"}
lang = st.sidebar.selectbox("Select Language 🌐", list(lang_options.keys()))
current_lang = lang_options[lang]

# Text translations
TEXTS = {
    "title": {
        "en": "Nail Disease Classifier 💅",
        "hi": "नाखून रोग वर्गीकरण 💅",
    },
    "upload_prompt": {
        "en": "Upload a nail image (JPG, PNG)",
        "hi": "नाखून की छवि अपलोड करें (JPG, PNG)",
    },
    "no_image": {
        "en": "Please upload an image to see predictions.",
        "hi": "कृपया भविष्यवाणियाँ देखने के लिए एक छवि अपलोड करें।",
    },
    "top_predictions": {
        "en": "Top 3 Predictions",
        "hi": "शीर्ष 3 भविष्यवाणियाँ",
    },
    "precautions": {
        "en": "Precautions & Suggestions",
        "hi": "सावधानियाँ और सुझाव",
    },
    "prediction_history": {
        "en": "Prediction History",
        "hi": "भविष्यवाणी इतिहास",
    },
    "download_csv": {
        "en": "Download History as CSV",
        "hi": "इतिहास CSV के रूप में डाउनलोड करें",
    },
    "download_pdf": {
        "en": "Download PDF Report",
        "hi": "PDF रिपोर्ट डाउनलोड करें",
    },
    "unsupported_file": {
        "en": "Unsupported file type. Please upload JPG or PNG.",
        "hi": "असमर्थित फ़ाइल प्रकार। कृपया JPG या PNG अपलोड करें।",
    },
    "not_nail": {
        "en": "This does not appear to be a nail image. Please upload a nail image.",
        "hi": "यह नाखून की छवि प्रतीत नहीं होती। कृपया नाखून की छवि अपलोड करें।",
    },
    "font_warning": {
        "en": "Warning: Unicode font not found. PDF will use English text only. Install 'DejaVuSans.ttf' for Hindi support.",
        "hi": "चेतावनी: यूनिकोड फॉन्ट नहीं मिला। PDF में केवल अंग्रेजी टेक्स्ट होगा। हिंदी समर्थन के लिए 'DejaVuSans.ttf' इंस्टॉल करें।"
    }
}

def t(key):
    return TEXTS[key][current_lang]

# Model and class definitions
num_classes = 7
class_names = [
    'Acral_Lentiginous_Melanoma',
    'Healthy_Nail',
    'Onychogryphosis',
    'beau',
    'blue_finger',
    'clubbing',
    'other'
]

precautions = {
    'Acral_Lentiginous_Melanoma': {
        "en": "- Consult a dermatologist immediately; this is a serious form of melanoma.\n- Avoid sun exposure and protect your nails.\n- Follow up regularly for monitoring and treatment.",
        "hi": "- तुरंत त्वचा रोग विशेषज्ञ से परामर्श करें; यह मेलेनोमा का एक गंभीर रूप है।\n- सूरज की रोशनी से बचें और अपने नाखूनों की सुरक्षा करें।\n- नियमित निगरानी और उपचार के लिए फॉलो-अप करें।"
    },
    'Healthy_Nail': {
        "en": "- No issues detected.\n- Maintain good nail hygiene.\n- Keep nails clean and moisturized.",
        "hi": "- कोई समस्या नहीं मिली।\n- अच्छी नाखून स्वच्छता बनाए रखें।\n- नाखूनों को साफ और मॉइस्चराइज रखें।"
    },
    'Onychogryphosis': {
        "en": "- Consult a healthcare provider for nail trimming or treatment.\n- Avoid trauma or injury to nails.\n- Maintain regular nail care routines.",
        "hi": "- नाखून काटने या उपचार के लिए स्वास्थ्य सेवा प्रदाता से परामर्श करें।\n- नाखूनों को चोट या आघात से बचाएं।\n- नियमित नाखून देखभाल बनाए रखें।"
    },
    'beau': {
        "en": "- Monitor for any progression or changes.\n- Keep nails clean and dry.\n- Consult a dermatologist if symptoms worsen.",
        "hi": "- किसी भी प्रगति या परिवर्तन के लिए निगरानी करें।\n- नाखूनों को साफ और सूखा रखें।\n- लक्षण बिगड़ने पर त्वचा विशेषज्ञ से परामर्श करें।"
    },
    'blue_finger': {
        "en": "- Check for circulatory or respiratory issues.\n- Consult a healthcare professional if persistent.\n- Avoid cold exposure and smoking.",
        "hi": "- परिसंचरण या श्वसन समस्याओं की जांच करें।\n- यदि समस्या बनी रहे तो स्वास्थ्य पेशेवर से परामर्श करें。\n- ठंड के संपर्क और धूम्रपान से बचें।"
    },
    'clubbing': {
        "en": "- Could indicate underlying health problems.\n- See a physician for proper diagnosis.\n- Monitor for related symptoms like breathlessness.",
        "hi": "- यह अंतर्निहित स्वास्थ्य समस्याओं का संकेत हो सकता है。\n- उचित निदान के लिए चिकित्सक से मिलें。\n- सांस फूलने जैसे संबंधित लक्षणों की निगरानी करें।"
    },
    'other': {
        "en": "- Seek professional advice for an accurate diagnosis.\n- Avoid self-treatment.\n- Keep nails clean and protected.",
        "hi": "- सटीक निदान के लिए पेशेवर सलाह लें。\n- स्वयं उपचार से बचें。\n- नाखूनों को साफ और सुरक्षित रखें।"
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
try:
    model.load_state_dict(torch.load("multiclass_model.pth", map_location=device))
except Exception as e:
    st.error(f"⚠ Model loading failed: {e}")
    st.stop()
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Improved placeholder function to check if the image contains a nail
def is_nail_image(img):
    """
    Improved placeholder for nail detection. Checks image size, aspect ratio, and color distribution.
    Replace with a proper nail detection model for better accuracy.
    """
    try:
        img_array = np.array(img)
        height, width = img_array.shape[:2]

        # Check image size and aspect ratio (nail images typically have a reasonable aspect ratio)
        if height < 50 or width < 50 or abs(height / width - 1) > 2:  # Avoid extreme aspect ratios
            return False

        # Check color distribution (refined range for skin tones and nail-like colors)
        mean_color = np.mean(img_array, axis=(0, 1))
        if not (120 < mean_color[0] < 220 and 80 < mean_color[1] < 180 and 80 < mean_color[2] < 180):
            return False

        # Check for high contrast (nails often have a distinct edge against skin)
        std_color = np.std(img_array, axis=(0, 1))
        if np.all(std_color < 20):  # Low contrast might indicate a non-nail image
            return False

        return True
    except Exception:
        return False

def risk_color(confidence):
    if confidence > 80:
        return "#4caf50"  # Green
    elif confidence > 50:
        return "#ff9800"  # Orange
    else:
        return "#f44336"  # Red

def create_pdf(pred_class, confidence, suggestion, image_path):
    pdf = FPDF()
    pdf.add_page()
    
    # Attempt to add and use a Unicode font (DejaVuSans)
    font_path = os.path.join(os.path.dirname(_file_), "DejaVuSans.ttf")
    unicode_font_available = False
    try:
        if os.path.exists(font_path):
            pdf.add_font('DejaVu', '', font_path, uni=True)
            unicode_font_available = True
        else:
            st.warning(t("font_warning"))
    except Exception as e:
        st.warning(f"{t('font_warning')} (Error: {e})")

    # Set font based on language and font availability
    if unicode_font_available and current_lang == "hi":
        pdf.set_font('DejaVu', 'B', 16)
        pdf.cell(0, 10, "Nail Disease Classification Report", ln=True, align="C")
        pdf.ln(10)
        pdf.set_font('DejaVu', '', 12)
    else:
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 10, "Nail Disease Classification Report", ln=True, align="C")
        pdf.ln(10)
        pdf.set_font('Helvetica', '', 12)
        # Fallback to English if Hindi font is unavailable
        if current_lang == "hi":
            suggestion = precautions[pred_class]["en"]

    pdf.cell(0, 10, f"Prediction: {pred_class}", ln=True)
    pdf.cell(0, 10, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Precautions & Suggestions:\n{suggestion}")
    pdf.ln(10)
    pdf.image(image_path, x=pdf.get_x(), w=100)
    return pdf

# Header
st.markdown(f"""
<div class="header">
    <h1>{t("title")}</h1>
    <p style="color: #e0e0e0; font-size: 1.1rem; margin-top: 10px;">Upload a nail image to detect potential conditions</p>
</div>
""", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

# File uploader and prediction section
uploaded_file = st.file_uploader(t("upload_prompt"), type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error(t("unsupported_file"))
        st.stop()

    # Check if the image is a nail
    if not is_nail_image(image):
        st.error(t("not_nail"))
        st.stop()

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner(""):
            st.markdown('<div class="spinner"></div>', unsafe_allow_html=True)
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        top3_idx = probabilities.argsort()[-3:][::-1]
        top3_probs = probabilities[top3_idx]
        top3_classes = [class_names[i] for i in top3_idx]

        st.markdown(f"### {t('top_predictions')}", unsafe_allow_html=True)

        for i in range(3):
            color = risk_color(top3_probs[i]*100)
            confidence_pct = top3_probs[i]*100

            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-title" style="color:{color};">{top3_classes[i]}</div>
                <div>Confidence: {confidence_pct:.2f}%</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width:{confidence_pct}%; background: linear-gradient(90deg, #4b0082, #7c4dff);"></div>
                </div>
                <div class="precautions-text">
                    <strong>{t('precautions')}</strong>
            """, unsafe_allow_html=True)

            for line in precautions[top3_classes[i]][current_lang].split('\n'):
                st.markdown(f"- {line.strip('- ')}")

            st.markdown("</div></div>", unsafe_allow_html=True)

        pred_class = top3_classes[0]
        confidence = top3_probs[0]*100
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Current time: 02:38 PM IST, July 24, 2025
        if not st.session_state.history or st.session_state.history[-1]['Timestamp'] != timestamp:
            st.session_state.history.append({
                'Timestamp': timestamp,
                'Prediction': pred_class,
                'Confidence (%)': f"{confidence:.2f}"
            })

        suggestion = precautions[pred_class][current_lang]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            image.save(tmpfile.name)
            pdf = create_pdf(pred_class, confidence, suggestion, tmpfile.name)
            pdf_output = bytes(pdf.output(dest='S'))

        st.download_button(
            label=t("download_pdf"),
            data=pdf_output,
            file_name=f"nail_disease_report_{timestamp}.pdf",
            mime="application/pdf"
        )

else:
    st.info(t("no_image"))

# Prediction history section
if st.session_state.history:
    st.markdown(f"## {t('prediction_history')}", unsafe_allow_html=True)
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history, use_container_width=True)

    csv_buffer = io.StringIO()
    df_history.to_csv(csv_buffer, index=False)
    st.download_button(
        label=t("download_csv"),
        data=csv_buffer.getvalue(),
        file_name="prediction_history.csv",
        mime="text/csv"
    )

# Footer with social icons
st.markdown("""
<div class="footer">
    <span>© 2025 Nail Disease Classifier | Powered by xAI</span>
    <a href="https://x.com" target="_blank">X</a>
    <a href="https://github.com" target="_blank">GitHub</a>
    <a href="https://linkedin.com" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)