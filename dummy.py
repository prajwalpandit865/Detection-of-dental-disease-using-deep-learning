import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ------------------ Page Config ------------------
st.set_page_config(page_title="🦷 Tooth Disease Detector", page_icon="🦷", layout="centered")

# ------------------ Custom CSS ------------------
st.markdown("""
    <style>
        body {background-color: #0a0a0a; font-family: 'Segoe UI', sans-serif; color:white;}
        .top-emoji {text-align:center; font-size:80px; margin-top:20px; margin-bottom:10px;}
        .title {text-align:center; color:#00ffff; font-size:42px; font-weight:800; margin-top:0px;}
        .subtitle {text-align:center; color:#ff99cc; font-size:20px; margin-bottom:40px;}
        .prediction-box {
            display: flex;
            justify-content: space-around;
            background-color: #1a1a1a;
            border-radius: 20px;
            box-shadow: 0 0 10px #222222;
            padding: 30px;
            margin: 20px auto;
            width:90%;
            max-width:900px;
        }
        .half-box {flex:1; text-align:center;}
        .header-label {font-size:18px; color:#ffffff; margin-bottom:10px; font-weight:600;}
        .predicted-disease {font-size:32px; font-weight:700; color:#8B0000;} /* Blood red */
        .confidence {font-size:32px; font-weight:700; color:#33ccff;}
        .precaution {
            background-color:#1a2a1a;
            border-left:6px solid #33cc33;
            padding:25px;
            border-radius:15px;
            margin:25px auto;
            width:90%;
            max-width:900px;
            font-size:18px;
            color:#33ff77;
            box-shadow: 0 0 10px #33ff77;
        }
        .precaution ul {padding-left:20px;}
        .precaution li {margin-bottom:10px;}
        .center-img {display:block; margin-left:auto; margin-right:auto;}
    </style>
""", unsafe_allow_html=True)

# ------------------ Top Tooth Emoji ------------------
st.markdown('<div class="top-emoji">🪵</div>', unsafe_allow_html=True)

# ------------------ Title ------------------
st.markdown('<div class="title">Wood Defect Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a wood image to identify possible defects.</div>', unsafe_allow_html=True)

# ------------------ Model Setup ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)
model.load_state_dict(torch.load("dental_model.pth", map_location=device))
model.to(device)
model.eval()

classes = ['Calculus', 'Caries', 'Hypodontia', 'Mouth Ulcer', 'Tooth Discoloration']

precautions = {
    'Calculus': [
        "Brush teeth twice daily with a soft-bristled toothbrush.",
        "Floss daily to remove plaque between teeth.",
        "Visit a dentist for cleaning every 6 months.",
        "Avoid excessive sugary or sticky foods.",
        "Maintain a balanced diet with enough calcium and vitamins."
    ],
    'Caries': [
        "Brush with fluoride toothpaste twice daily.",
        "Limit sugary snacks and drinks.",
        "Use antibacterial mouthwash.",
        "Get regular dental check-ups for early intervention.",
        "Ensure good oral hygiene from an early age."
    ],
    'Hypodontia': [
        "Consult an orthodontist or dentist for diagnosis.",
        "Consider implants or prosthetics for missing teeth.",
        "Keep remaining teeth clean and healthy.",
        "Follow professional advice for braces or corrective treatment.",
        "Regular dental check-ups for monitoring."
    ],
    'Mouth Ulcer': [
        "Avoid spicy, acidic, or rough foods.",
        "Rinse mouth with saline or antiseptic solution.",
        "Stay hydrated and maintain a balanced diet.",
        "Use dentist-recommended topical treatments.",
        "Consult a dentist if ulcers persist or recur."
    ],
    'Tooth Discoloration': [
        "Reduce intake of coffee, tea, and tobacco.",
        "Brush and floss twice daily.",
        "Consider professional cleaning or whitening.",
        "Maintain overall oral hygiene.",
        "Consult dentist if discoloration persists."
    ]
}

# ------------------ Transform ------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ------------------ Upload Image ------------------
uploaded_file = st.file_uploader("📸 Upload a dental image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # --- Center the image using Streamlit columns ---
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        st.write("")
    with col2:
        st.image(image, width=300)
    with col3:
        st.write("")
    # -----------------------------------------------

    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)
        predicted_class = classes[preds.item()]
    confidence = conf.item() * 100

    # Full-width Prediction + Confidence box
    st.markdown(f"""
        <div class='prediction-box'>
            <div class='half-box'>
                <div class='header-label'>Predicted Disease</div>
                <div class='predicted-disease'>{predicted_class}</div>
            </div>
            <div class='half-box'>
                <div class='header-label'>Confidence</div>
                <div class='confidence'>{confidence:.2f}%</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Precautions
    precaution_points = "".join([f"<li>{step}</li>" for step in precautions[predicted_class]])
    st.markdown(f"<div class='precaution'><strong>💡 Precautions & Steps to Take:</strong><ul>{precaution_points}</ul></div>", unsafe_allow_html=True)
