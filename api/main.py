from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import sys
import os

# Add root project dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector.mlp import MLPDetector
from gan.generator import Generator

app = FastAPI(title="DDoS-GAN Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load real data
try:
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    real_ddos_idx = np.where(y_test == 1)[0]
except Exception as e:
    print(f"[WARNING] Could not load processed data: {e}")
    X_test = np.zeros((10, 80))
    real_ddos_idx = np.array([0,1,2])

# Initialize models
print("\n[INFO] Loading models...")
input_dim = 80
latent_dim = 64
hidden_dims = [256, 128, 64]

# Detectors
detector_v1 = MLPDetector(input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.3).to(DEVICE)
detector_v1.load_state_dict(torch.load("detector/detector_best.pt", map_location=DEVICE))
detector_v1.eval()

detector_v2 = MLPDetector(input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.3).to(DEVICE)
detector_v2.load_state_dict(torch.load("detector/detector_adv_r1.pt", map_location=DEVICE))
detector_v2.eval()

# Generators
gen_r0 = Generator(latent_dim=latent_dim, output_dim=input_dim).to(DEVICE)
gen_r0.load_state_dict(torch.load("gan/generator_r0.pt", map_location=DEVICE))
gen_r0.eval()

gen_r2 = Generator(latent_dim=latent_dim, output_dim=input_dim).to(DEVICE)
gen_r2.load_state_dict(torch.load("gan/generator_r2.pt", map_location=DEVICE))
gen_r2.eval()
print("[INFO] All models loaded successfully!\n")

@app.get("/api/sample/real")
def get_real_sample():
    idx = np.random.choice(real_ddos_idx)
    sample = X_test[idx]
    return {"features": sample.tolist(), "type": "Real DDoS"}

@app.post("/api/generate")
def generate_fake(round: int = 0):
    z = torch.randn(1, latent_dim, device=DEVICE)
    with torch.no_grad():
        if round == 0:
            fake = gen_r0(z)
        elif round == 2:
            fake = gen_r2(z)
        else:
            return {"error": "Invalid round"}
    return {"features": fake.cpu().numpy()[0].tolist(), "type": f"GAN Round {round}"}

from typing import Dict, Any

@app.post("/api/detect")
def detect(model: str, req: Dict[str, Any]):
    features = req.get("features", [])
    x = torch.tensor([features], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        if model == "v1":
            logits = detector_v1(x)
        elif model == "v2":
            logits = detector_v2(x)
        else:
            return {"error": "Invalid model"}
        prob = torch.sigmoid(logits).item()
    
    status = "BLOCKED" if prob >= 0.5 else "BYPASSED"
    return {"probability": prob, "status": status}
