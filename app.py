import os
import cv2
import torch
import numpy as np
import pydicom
import tempfile
import base64
import datetime
from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from groq import Groq
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
from fpdf import FPDF

load_dotenv()
app = Flask(__name__)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "judge_model_epoch5.pth")
RL_AGENT_PATH = os.path.join(BASE_DIR, "strict_rl_agent")

# --- GROQ SETUP ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = None

if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq API Connected")
    except Exception as e:
        print(f"❌ Groq Error: {e}")

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CLASS_MAPPING = {
    1: 'Aortic enlargement', 2: 'Atelectasis', 3: 'Calcification', 4: 'Cardiomegaly',
    5: 'Consolidation', 6: 'ILD', 7: 'Infiltration', 8: 'Lung Opacity',
    9: 'Nodule/Mass', 10: 'Other lesion', 11: 'Pleural effusion', 12: 'Pleural thickening',
    13: 'Pneumothorax', 14: 'Pulmonary fibrosis'
}

# --- MODEL LOADING ---
def get_detector():
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 15)
    return model

detector = get_detector()
if os.path.exists(MODEL_PATH):
    detector.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
detector.to(DEVICE)
detector.eval()

class DummyEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=255, shape=(512, 512, 3), dtype=np.uint8)

agent = None
if os.path.exists(RL_AGENT_PATH + ".zip") or os.path.exists(RL_AGENT_PATH):
    try:
        agent = PPO.load(RL_AGENT_PATH, env=DummyEnv(), device=DEVICE)
    except: pass

# --- HELPERS ---
def encode_img(img_np):
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

def preprocess_upload(file):
    if file.filename.lower().endswith(('.dcm', '.dicom')):
        dcm = pydicom.dcmread(file)
        try: img = apply_voi_lut(dcm.pixel_array, dcm)
        except: img = dcm.pixel_array
        if dcm.PhotometricInterpretation == "MONOCHROME1": img = np.amax(img) - img
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = Image.open(file).convert('RGB')
        img = np.array(img)
    return cv2.resize(img, (512, 512))

def create_heatmap(boxes, scores):
    heatmap = np.zeros((512, 512), dtype=np.float32)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(512, x2), min(512, y2)
        try: heatmap[y1:y2, x1:x2] += score
        except: pass
    heatmap = cv2.GaussianBlur(heatmap, (101, 101), 0)
    if np.max(heatmap) > 0: heatmap = heatmap / np.max(heatmap)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    return cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

# --- PROFESSIONAL HOSPITAL PDF REPORT ---
class HospitalReport(FPDF):
    def header(self):
        # Hospital Logo/Header
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'CITY GENERAL HOSPITAL', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 5, 'Department of Radiology & Diagnostic Imaging', 0, 1, 'C')
        self.cell(0, 5, '123 Medical Center Dr, Metropolis, NY 10012', 0, 1, 'C')
        self.line(10, 30, 200, 30)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Authorized by: Senior Radiologist (ID: 4492) - Page {self.page_no()}', 0, 0, 'C')

    def patient_info_block(self):
        self.set_font('Arial', 'B', 10)
        self.cell(30, 6, 'Patient Name:', 0, 0)
        self.set_font('Arial', '', 10)
        self.cell(60, 6, 'DOE, JOHN (M/45)', 0, 0)
        
        self.set_font('Arial', 'B', 10)
        self.cell(25, 6, 'Exam Date:', 0, 0)
        self.set_font('Arial', '', 10)
        self.cell(40, 6, datetime.date.today().strftime("%Y-%m-%d"), 0, 1)

        self.set_font('Arial', 'B', 10)
        self.cell(30, 6, 'Patient ID:', 0, 0)
        self.set_font('Arial', '', 10)
        self.cell(60, 6, 'MRN-884291', 0, 0)

        self.set_font('Arial', 'B', 10)
        self.cell(25, 6, 'Modality:', 0, 0)
        self.set_font('Arial', '', 10)
        self.cell(40, 6, 'Chest X-Ray (PA View)', 0, 1)
        self.line(10, 55, 200, 55)
        self.ln(10)

    def add_section(self, title, body):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 8, title, 0, 1, 'L', fill=True)
        self.ln(2)
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln(5)

def generate_pdf(data):
    pdf = HospitalReport()
    pdf.add_page()
    pdf.patient_info_block()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        enh_path = os.path.join(tmpdir, 'enhanced.jpg')
        heat_path = os.path.join(tmpdir, 'heatmap.jpg')
        
        Image.fromarray(data['enhanced_img']).save(enh_path)
        Image.fromarray(cv2.cvtColor(data['heatmap_img'], cv2.COLOR_BGR2RGB)).save(heat_path)

        # 1. TEXT REPORT
        # We split the report text if it has sections, otherwise dump it
        pdf.add_section("RADIOLOGICAL REPORT", data['report_text'])
        
        pdf.ln(5)
        pdf.set_font('Arial', 'I', 9)
        pdf.multi_cell(0, 5, "This report was generated with AI assistance. Clinical correlation is recommended.")

        # 2. IMAGES PAGE
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "DIAGNOSTIC IMAGERY", 0, 1, 'C')
        
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(95, 5, "Figure 1: Enhanced Frontal Radiograph", 0, 0, 'C')
        pdf.cell(95, 5, "Figure 2: Computer-Aided Detection (Heatmap)", 0, 1, 'C')
        
        pdf.image(enh_path, x=10, y=40, w=90)
        pdf.image(heat_path, x=110, y=40, w=90)
        
        # 3. CROPS (If any)
        if data['crops']:
            pdf.ln(100) # Move down below images
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 10, "Localized Regions of Interest", 0, 1, 'L')
            
            y_pos = pdf.get_y()
            for i, (crop_img, name) in enumerate(data['crops']):
                # Simple layout: 3 per row
                if i > 2: break # Limit for PDF cleanliness
                
                c_path = os.path.join(tmpdir, f'crop_{i}.jpg')
                Image.fromarray(crop_img).save(c_path)
                
                x_pos = 10 + (i * 65)
                pdf.image(c_path, x=x_pos, y=y_pos, w=60)
                pdf.set_xy(x_pos, y_pos + 62)
                pdf.set_font('Arial', '', 8)
                pdf.cell(60, 5, name, 0, 0, 'C')

        pdf_path = os.path.join(tmpdir, 'Hospital_Report.pdf')
        pdf.output(pdf_path)
        with open(pdf_path, "rb") as f:
            return f.read()

# --- ROUTES ---

@app.route('/')
def landing(): return render_template('landing.html')

@app.route('/app')
def dashboard(): return render_template('dashboard.html')

@app.route('/process', methods=['POST'])
def process():
    file = request.files['file']
    threshold = float(request.form.get('threshold', 0.2))
    
    # 1. Preprocess
    original = preprocess_upload(file)
    
    # 2. GROQ VISION ADVICE (Hidden from report, used for Dashboard)
    llm_advice = "System Ready."
    if groq_client:
        try:
            # Using Llama 3.2 11B Vision for image quality assessment
            base64_img = encode_img(original)
            resp = groq_client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Act as a radiologist. Briefly assess image quality. One sentence."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_img}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=100
            )
            llm_advice = resp.choices[0].message.content
        except Exception as e:
            print(f"Groq Vision Error: {e}")
            pass

    # 3. RL AGENT (Still runs, but we don't talk about it in the report)
    current = original.copy()
    frames = [encode_img(current)]
    actions = []
    
    if agent:
        for _ in range(5):
            action, _ = agent.predict(current)
            if action == 0: current = np.clip(current + 15, 0, 255).astype(np.uint8); actions.append("Bright +")
            elif action == 1: current = np.clip(current - 15, 0, 255).astype(np.uint8); actions.append("Bright -")
            elif action == 2: current = np.clip((current - 127.5)*1.2 + 127.5, 0, 255).astype(np.uint8); actions.append("Contrast +")
            elif action == 3: current = np.clip((current - 127.5)*0.8 + 127.5, 0, 255).astype(np.uint8); actions.append("Contrast -")
            elif action == 4: actions.append("Submit"); break
            frames.append(encode_img(current))
    
    enhanced = current

    # 4. DETECTOR
    inp = torch.tensor(enhanced).permute(2, 0, 1).float()/255.0; inp = inp.unsqueeze(0).to(DEVICE)
    with torch.no_grad(): preds = detector(inp)[0]
    
    keep = preds['scores'] > threshold
    if keep.sum() > 0:
        nms_idx = torchvision.ops.nms(preds['boxes'][keep], preds['scores'][keep], 0.2)
        boxes = preds['boxes'][keep][nms_idx].cpu().numpy()
        scores = preds['scores'][keep][nms_idx].cpu().numpy()
        labels = preds['labels'][keep][nms_idx].cpu().numpy()
    else: boxes, scores, labels = [], [], []

    # 5. ANNOTATE & CROP
    annotated = enhanced.copy()
    findings_list = []
    crops = []; crops_b64 = []

    for box, lbl, sc in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        name = CLASS_MAPPING.get(int(lbl), "Unknown")
        confidence = f"{sc:.1%}"
        findings_list.append(f"{name} (Confidence: {confidence})")
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(annotated, name, (x1, max(y1-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Crop
        pad = 30
        c_img = enhanced[max(0, y1-pad):min(512, y2+pad), max(0, x1-pad):min(512, x2+pad)]
        if c_img.size > 0:
            crops.append((c_img, name))
            crops_b64.append({'img': encode_img(c_img), 'name': name})

    # 6. HEATMAP
    heatmap_img = create_heatmap(boxes, scores)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(enhanced_bgr, 0.6, heatmap_img, 0.4, 0)

    # 7. GENERATE PROFESSIONAL REPORT
    report = "No acute abnormalities detected. Lungs appear clear."
    if findings_list and groq_client:
        try:
            # STRICT MEDICAL PROMPT - NO AI TALK
            # Using Llama 3.3 70B Versatile for Report Generation
            prompt = f"""
            Role: Senior Consultant Radiologist.
            Task: Write the BODY of a formal DIAGNOSTIC REPORT for a referring physician.
            
            Visual Findings detected by system: {findings_list}.
            
            Structure required:
            1. FINDINGS: Describe the location and characteristics of ALL detected anomalies. Address each finding in the list. Use professional medical terminology. Mention the confidence level as 'High probability' or 'Suspicion of'.
            2. IMPRESSION: A concise summary diagnosis.
            3. PATIENT CONDITION & PATHOPHYSIOLOGY: Explain the likely physical condition of the patient given these findings. Describe the pathophysiology (how the disease processes).
            
            RULES:
            - Do NOT include a header, patient name, date, or referring physician. Start directly with 'FINDINGS'.
            - Address ALL detected findings.
            - Do NOT mention 'RL agent', 'AI', 'Algorithm', 'Steps', or 'Computer Vision'.
            - Write as if YOU are the doctor looking at the X-ray.
            - Tone: Formal, Clinical, Direct.
            """
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            report = resp.choices[0].message.content
        except Exception as e:
            print(f"Groq Report Error: {e}")
            pass
    elif findings_list:
        report = "FINDINGS:\n" + "\n".join(findings_list)

    # Store for PDF
    global last_pdf_data
    last_pdf_data = {
        'enhanced_img': annotated, 
        'heatmap_img': overlay,
        'crops': crops, 
        'report_text': report
    }

    return jsonify({
        "frames": frames, 
        "actions": actions, 
        "llm_advice": llm_advice,
        "final_result": encode_img(annotated), 
        "heatmap": encode_img(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)),
        "crops": crops_b64, 
        "report": report
    })

@app.route('/download_pdf')
def download_pdf():
    if 'last_pdf_data' not in globals(): return "No data", 400
    pdf_bytes = generate_pdf(last_pdf_data)
    return (pdf_bytes, 200, {'Content-Type': 'application/pdf', 'Content-Disposition': 'attachment; filename=Hospital_Report.pdf'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)