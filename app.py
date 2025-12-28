import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time

# --------------------------------------------------
# 1. PAGE CONFIG & STYLING
# --------------------------------------------------
st.set_page_config(page_title="Dental AI Diagnostics", page_icon="🦷", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    .main-title {
        text-align: center; font-size: 56px; font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .main-subtitle { text-align: center; color: #94a3b8; font-size: 22px; margin-bottom: 40px; }
    
    /* GREEN PRECAUTION BOX */
    .bulb-box {
        background: linear-gradient(135deg, #065f46 0%, #064e3b 100%);
        border-left: 8px solid #10b981;
        padding: 30px; border-radius: 20px; margin-top: 25px;
        box-shadow: 0 15px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .bulb-header { display: flex; align-items: center; margin-bottom: 15px; }
    .bulb-icon { font-size: 38px; margin-right: 18px; }
    .bulb-text-title { color: #a7f3d0; font-weight: 800; font-size: 28px; }
    .precaution-item { color: #ecfdf5; font-size: 19px; margin-bottom: 12px; line-height: 1.5; }

    /* RESULT CARDS */
    .card {
        background-color: #1e293b; border: 1px solid #334155;
        border-radius: 20px; padding: 35px; margin-bottom: 25px;
    }
    .stat-label { color: #38bdf8; font-size: 16px; text-transform: uppercase; font-weight: 700; margin-bottom: 12px; }
    .stat-value { color: #f8fafc; font-size: 38px; font-weight: 800; }
    .confidence-value { color: #22c55e; font-size: 38px; font-weight: 800; }
    .treatment-text { color: #f1f5f9; font-size: 21px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 2. DATA ENGINE (Clinical Knowledge)
# --------------------------------------------------
dental_knowledge = {
    "Calculus": {
        "precautions": [
            "Schedule professional scaling (deep cleaning) within 14 days.",
            "Use a tartar-control toothpaste with pyrophosphates.",
            "Switch to an electric toothbrush for more effective plaque removal.",
            "Floss daily to prevent buildup between teeth.",
            "Rinse with an antimicrobial mouthwash to inhibit bacteria."
        ],
        "treatment": "Professional Scaling and Root Planing (Deep Cleaning)."
    },
    "Caries": {
        "precautions": [
            "Reduce intake of sugary and acidic beverages immediately.",
            "Brush with high-fluoride toothpaste (1450ppm+) twice daily.",
            "Use interdental brushes to clean hidden decay areas.",
            "Chew Xylitol-based sugar-free gum after meals.",
            "Rinse with a neutral pH mouthwash to protect enamel."
        ],
        "treatment": "Dental Restoration (Filling) or Fluoride Therapy."
    },
    "Hypodontia": {
        "precautions": [
            "Consult an orthodontist to manage spacing issues.",
            "Monitor primary teeth that may be retained longer than usual.",
            "Maintain strict hygiene for adjacent teeth to prevent tilting.",
            "Discuss dental implant or bridge options for adult teeth gaps.",
            "Avoid excessive pressure on gum areas where teeth are missing."
        ],
        "treatment": "Space Management or Dental Prosthetics (Implants/Bridges)."
    },
    "Mouth Ulcer": {
        "precautions": [
            "Avoid spicy, acidic, or very hot foods for 1 week.",
            "Apply a topical oral numbing gel to reduce pain.",
            "Rinse with a warm salt-water solution 3 times daily.",
            "Switch to a SLS-free (Sodium Lauryl Sulfate) toothpaste.",
            "Increase intake of Vitamin B12 and Iron-rich foods."
        ],
        "treatment": "Topical Corticosteroids or Antimicrobial Gels."
    },
    "Tooth Discoloration": {
        "precautions": [
            "Limit stains from coffee, red wine, and tobacco.",
            "Rinse with water immediately after drinking dark liquids.",
            "Use a gentle whitening toothpaste to remove surface stains.",
            "Maintain regular 6-month professional cleaning sessions.",
            "Avoid abrasive home-whitening kits that damage enamel."
        ],
        "treatment": "Professional Bleaching, Microabrasion, or Veneers."
    }
}

classes = list(dental_knowledge.keys())

# --------------------------------------------------
# 3. MODEL LOADING & PREDICTION
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    # Initialize your model architecture (matching your training)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    # Load your weights
    try:
        model.load_state_dict(torch.load("dental_model.pth", map_location=device))
    except FileNotFoundError:
        st.error("Error: 'dental_model.pth' not found in directory.")
    model.to(device)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --------------------------------------------------
# 4. USER INTERFACE
# --------------------------------------------------
st.markdown("<h1 class='main-title'>DentalAI Diagnostics</h1>", unsafe_allow_html=True)
st.markdown("<p class='main-subtitle'>Neural Network Analysis for Precision Care</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Dental Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)
    
    # ACTUAL INFERENCE PROCESS
    with st.spinner('🔍 Detecting Disease...'):
        time.sleep(1.0) # Slight delay for UX
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            
        detected_class = classes[pred_idx]
        confidence = probs[pred_idx].item() * 100

    # DISPLAY DYNAMIC RESULTS
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='card'><div class='stat-label'>Detected</div><div class='stat-value'>{detected_class}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='card'><div class='stat-label'>Confidence</div><div class='confidence-value'>{confidence:.2f}%</div></div>", unsafe_allow_html=True)

    # TREATMENT CARD
    st.markdown(f"""
        <div class="card">
            <div class="stat-label">Clinical Action</div>
            <div class="treatment-text">{dental_knowledge[detected_class]['treatment']}</div>
        </div>
    """, unsafe_allow_html=True)

    # PRECAUTIONS (Green box at bottom, only after detection)
    p_items = "".join([f"<div class='precaution-item'>• {item}</div>" for item in dental_knowledge[detected_class]['precautions']])
    st.markdown(f"""
        <div class="bulb-box">
            <div class="bulb-header">
                <span class="bulb-icon">💡</span>
                <span class="bulb-text-title">Top 5 Precautions</span>
            </div>
            {p_items}
        </div>
    """, unsafe_allow_html=True)
else:
    st.info("Please upload a dental scan to begin.")
