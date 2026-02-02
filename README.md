# An Explainable AI Decision Support System for Mitotic Detection using B-cos Networks

## Abstract
Mitotic count is a critical prognostic factor in breast cancer grading (Nottingham Grading System). However, manual counting is labor-intensive, subjective, and prone to inter-observer variability.

This project introduces a Transparent Decision Support System that automates mitotic figure detection while prioritizing clinical trust. Unlike standard "Black Box" AI, our system uses B-cos Networks to provide "faithful-by-design" pixel-level explanations for every prediction.

## Key Features
* Scalable Detection (Stage 1): Utilizes YOLOv11L integrated with SAHI (Slicing Aided Hyper Inference) to detect small mitotic figures in high-resolution Whole Slide Images (WSIs) without downsampling artifacts.
* Interpretable Classification (Stage 2): A B-cos ResNet50 classifier that distinguishes mitotic cells from mimics (e.g., apoptotic bodies) with 94% Precision.
* Faithful Explainability: Generates high-fidelity saliency maps directly from the model's forward pass, avoiding the inaccuracies of post-hoc methods like Grad-CAM.
* Smart Focus Visualization: A novel clinical visualization tool that uses saliency maps as alpha masks to blur background noise and spotlight diagnostic chromatin features.

## Tech Stack
* Deep Learning: PyTorch, Ultralytics YOLOv11
* Data Processing: NumPy, Pillow, SAHI
* Backend: FastAPI (Python)
* Frontend: Next.js (React)
* Visualization: Matplotlib
* Hardware: Optimized for NVIDIA T4 GPU

## Pipeline Overview
1. Input: High-resolution Histopathology ROI (TUPAC16 Dataset).
2. Preprocessing: Normalization & stain-preserving augmentation.
3. Candidate Identification: SAHI slices the image -> YOLOv11 detects candidates -> Global Stitching & NMS.
4. Candidate Recognition: B-cos ResNet50 classifies crops & generates explanation maps.
5. Output: Final annotated image + "Smart Focus" interactive overlay.

## Performance Metrics
Our system achieves a competitive balance of sensitivity and specificity:

| Stage | Model | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| Stage 1 | YOLOv11L + SAHI | 0.82 | 0.74 | 0.78 |
| Stage 2 | B-cos ResNet50 | 0.94 | 0.83 | 0.88 |
| Pipeline | End-to-End | 0.94 | 0.62 | 0.75 |

## Installation & Setup

### Prerequisites
* Python 3.8+
* Node.js & npm
* Git LFS (Large File Storage)

### 1. Clone the Repository
git clone https://github.com/oxihigh/mitotic-detection-system.git
cd mitotic-detection-system
git lfs pull

### 2. Backend Setup (Python)
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt

### 3. Frontend Setup (Next.js)
cd ../app
npm install

## Usage
1. Start the Backend API:
   cd backend
   uvicorn main:app --reload

2. Start the Frontend UI:
   cd app
   npm run dev

3. Access the Dashboard:
   Open http://localhost:3000 in your browser. Upload a sample ROI image to see the detection and Smart Focus visualization in action.

## Team
Department of Information Science and Engineering
Jyothy Institute of Technology, Bengaluru

* Rajesh Kumar K (1JT22IS036)
* Sai Skanda A (1JT22IS042)
* Skanda S (1JT22IS048)
* Dhruthi Halibandi (1JT22IS014)

Guide: Asst. Prof. Rashmi K S

## License
This project is developed for academic research purposes.
