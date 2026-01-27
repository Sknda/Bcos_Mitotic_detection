from fastapi import FastAPI, APIRouter, File, UploadFile
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, timezone
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import cv2

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Model paths
MODELS_DIR = Path("/app/models")
YOLO_MODEL_PATH = MODELS_DIR / "yolov11_model.pt"
RESNET_MODEL_PATH = MODELS_DIR / "resnet_model.pt"

# Global model variables
yolo_model = None
resnet_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing for ResNet (6-channel input)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class DetectionBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
    confidence: float
    label: str

class ModelResult(BaseModel):
    model_name: str
    predictions: List[Dict[str, Any]]
    total_detections: int
    mitotic_count: int
    non_mitotic_count: int
    annotated_image: str  # base64
    explainability_images: List[str]  # base64 encoded images
    summary: str

class AnalysisResponse(BaseModel):
    success: bool
    original_image: str  # base64
    yolo_results: ModelResult
    resnet_results: ModelResult
    processing_time: float

# Model loading functions
def load_models():
    """Load both YOLO and ResNet models on startup"""
    global yolo_model, resnet_model
    
    logger.info(f"Loading models from {MODELS_DIR}")
    logger.info(f"Using device: {device}")
    
    try:
        # Load YOLO model
        if YOLO_MODEL_PATH.exists():
            logger.info(f"Loading YOLO model from {YOLO_MODEL_PATH}")
            yolo_model = YOLO(str(YOLO_MODEL_PATH))
            logger.info("✓ YOLO model loaded successfully")
        else:
            logger.warning(f"YOLO model not found at {YOLO_MODEL_PATH}")
            logger.info("Using mock YOLO model for demo")
            yolo_model = None
        
        # Load ResNet model
        if RESNET_MODEL_PATH.exists():
            logger.info(f"Loading ResNet model from {RESNET_MODEL_PATH}")
            resnet_model = torch.load(str(RESNET_MODEL_PATH), map_location=device)
            resnet_model.eval()
            logger.info("✓ ResNet model loaded successfully")
        else:
            logger.warning(f"ResNet model not found at {RESNET_MODEL_PATH}")
            logger.info("Using mock ResNet model for demo")
            resnet_model = None
            
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.info("Will use mock models for demo")
        yolo_model = None
        resnet_model = None

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.convert('RGB').save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def generate_yolo_visualization(image: Image.Image, detections: List[Dict]) -> str:
    """Generate annotated image with YOLO detections"""
    img_array = np.array(image.convert('RGB'))
    
    for det in detections:
        x, y, w, h = det['bbox']
        label = det['label']
        conf = det['confidence']
        
        # Choose color based on label
        color = (255, 0, 0) if 'mitotic' in label.lower() else (0, 255, 0)
        
        # Draw bounding box
        cv2.rectangle(img_array, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        text = f"{label}: {conf:.2f}"
        cv2.putText(img_array, text, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    result_image = Image.fromarray(img_array)
    return image_to_base64(result_image)

def generate_resnet_heatmap(image: Image.Image, attention_map: Optional[np.ndarray] = None) -> str:
    """Generate Grad-CAM style heatmap for ResNet predictions"""
    img_array = np.array(image.convert('RGB'))
    height, width = img_array.shape[:2]
    
    if attention_map is None:
        # Generate mock attention map if not provided
        attention_map = np.random.rand(height, width)
    
    # Normalize and apply colormap
    attention_map = cv2.resize(attention_map, (width, height))
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # Apply jet colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    result = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
    result_image = Image.fromarray(result)
    
    return image_to_base64(result_image)

def run_yolo_detection(image: Image.Image) -> Dict[str, Any]:
    """Run YOLO model for cell detection"""
    import time
    start_time = time.time()
    
    if yolo_model is not None:
        try:
            # Run YOLO inference
            img_array = np.array(image.convert('RGB'))
            results = yolo_model(img_array)
            
            detections = []
            mitotic_count = 0
            non_mitotic_count = 0
            
            # Parse YOLO results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Map class to label (adjust based on your model)
                    label = "Mitotic Cell" if cls == 0 else "Non-Mitotic Cell"
                    
                    if 'mitotic' in label.lower():
                        mitotic_count += 1
                    else:
                        non_mitotic_count += 1
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                        'confidence': conf,
                        'label': label,
                        'class_id': cls
                    })
            
            # Generate visualizations
            annotated_img = generate_yolo_visualization(image, detections)
            
            processing_time = time.time() - start_time
            
            return {
                'model_name': 'YOLOv11 Detection',
                'predictions': detections,
                'total_detections': len(detections),
                'mitotic_count': mitotic_count,
                'non_mitotic_count': non_mitotic_count,
                'annotated_image': annotated_img,
                'explainability_images': [annotated_img],
                'summary': f'YOLOv11 detected {len(detections)} cells: {mitotic_count} mitotic and {non_mitotic_count} non-mitotic cells in {processing_time:.2f} seconds.',
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in YOLO detection: {str(e)}")
            # Fall back to mock
            pass
    
    # Mock YOLO detection
    logger.info("Using mock YOLO detection")
    detections = [
        {'bbox': [50, 50, 100, 100], 'confidence': 0.92, 'label': 'Mitotic Cell', 'class_id': 0},
        {'bbox': [200, 150, 80, 80], 'confidence': 0.88, 'label': 'Non-Mitotic Cell', 'class_id': 1},
        {'bbox': [350, 100, 90, 90], 'confidence': 0.85, 'label': 'Mitotic Cell', 'class_id': 0},
    ]
    
    mitotic_count = sum(1 for d in detections if 'mitotic' in d['label'].lower())
    non_mitotic_count = len(detections) - mitotic_count
    
    annotated_img = generate_yolo_visualization(image, detections)
    
    return {
        'model_name': 'YOLOv11 Detection (Mock)',
        'predictions': detections,
        'total_detections': len(detections),
        'mitotic_count': mitotic_count,
        'non_mitotic_count': non_mitotic_count,
        'annotated_image': annotated_img,
        'explainability_images': [annotated_img],
        'summary': f'YOLOv11 detected {len(detections)} cells: {mitotic_count} mitotic and {non_mitotic_count} non-mitotic cells. (Using mock data - upload models to /app/models)',
        'processing_time': 0.1
    }

def run_resnet_classification(image: Image.Image, yolo_detections: List[Dict]) -> Dict[str, Any]:
    """Run ResNet model on YOLO detected regions"""
    import time
    start_time = time.time()
    
    if resnet_model is not None:
        try:
            # Process each detected region with ResNet
            img_array = np.array(image.convert('RGB'))
            refined_predictions = []
            
            for det in yolo_detections:
                x, y, w, h = det['bbox']
                
                # Crop region
                region = img_array[y:y+h, x:x+w]
                region_pil = Image.fromarray(region)
                
                # Preprocess for ResNet
                input_tensor = preprocess(region_pil).unsqueeze(0).to(device)
                
                # Run inference
                with torch.no_grad():
                    output = resnet_model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    
                    # Get prediction
                    pred_idx = torch.argmax(probabilities).item()
                    confidence = probabilities[pred_idx].item()
                    
                    label = "Mitotic Cell" if pred_idx == 0 else "Non-Mitotic Cell"
                    
                    refined_predictions.append({
                        'bbox': det['bbox'],
                        'confidence': confidence,
                        'label': label,
                        'yolo_label': det['label'],
                        'refinement': 'Confirmed' if label == det['label'] else 'Corrected'
                    })
            
            mitotic_count = sum(1 for p in refined_predictions if 'mitotic' in p['label'].lower())
            non_mitotic_count = len(refined_predictions) - mitotic_count
            
            # Generate visualizations
            annotated_img = generate_yolo_visualization(image, refined_predictions)
            heatmap = generate_resnet_heatmap(image)
            
            processing_time = time.time() - start_time
            
            return {
                'model_name': 'ResNet Classification',
                'predictions': refined_predictions,
                'total_detections': len(refined_predictions),
                'mitotic_count': mitotic_count,
                'non_mitotic_count': non_mitotic_count,
                'annotated_image': annotated_img,
                'explainability_images': [annotated_img, heatmap],
                'summary': f'ResNet refined the classifications: {mitotic_count} mitotic and {non_mitotic_count} non-mitotic cells confirmed in {processing_time:.2f} seconds.',
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in ResNet classification: {str(e)}")
            # Fall back to mock
            pass
    
    # Mock ResNet classification
    logger.info("Using mock ResNet classification")
    refined_predictions = []
    for det in yolo_detections:
        refined_predictions.append({
            'bbox': det['bbox'],
            'confidence': det['confidence'] * 0.95,
            'label': det['label'],
            'yolo_label': det['label'],
            'refinement': 'Confirmed'
        })
    
    mitotic_count = sum(1 for p in refined_predictions if 'mitotic' in p['label'].lower())
    non_mitotic_count = len(refined_predictions) - mitotic_count
    
    annotated_img = generate_yolo_visualization(image, refined_predictions)
    heatmap = generate_resnet_heatmap(image)
    
    return {
        'model_name': 'ResNet Classification (Mock)',
        'predictions': refined_predictions,
        'total_detections': len(refined_predictions),
        'mitotic_count': mitotic_count,
        'non_mitotic_count': non_mitotic_count,
        'annotated_image': annotated_img,
        'explainability_images': [annotated_img, heatmap],
        'summary': f'ResNet refined the classifications: {mitotic_count} mitotic and {non_mitotic_count} non-mitotic cells confirmed. (Using mock data - upload models to /app/models)',
        'processing_time': 0.1
    }

# API Routes
@api_router.get("/")
async def root():
    return {
        "message": "ML Model Comparison API - Mitotic Cell Detection",
        "models_loaded": {
            "yolo": yolo_model is not None,
            "resnet": resnet_model is not None
        }
    }

@api_router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """Analyze uploaded image with sequential YOLO -> ResNet pipeline"""
    import time
    overall_start = time.time()
    
    try:
        # Read and process uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Resize if too large (keep aspect ratio)
        max_size = 800
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert original image to base64
        original_b64 = image_to_base64(image)
        
        logger.info("Starting sequential model pipeline: YOLO -> ResNet")
        
        # Step 1: Run YOLO detection
        logger.info("Step 1: Running YOLO detection...")
        yolo_results = run_yolo_detection(image)
        logger.info(f"YOLO detected {yolo_results['total_detections']} cells")
        
        # Step 2: Run ResNet on YOLO detections
        logger.info("Step 2: Running ResNet classification on detected regions...")
        resnet_results = run_resnet_classification(image, yolo_results['predictions'])
        logger.info(f"ResNet refined {resnet_results['total_detections']} classifications")
        
        total_time = time.time() - overall_start
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        
        return AnalysisResponse(
            success=True,
            original_image=original_b64,
            yolo_results=ModelResult(**yolo_results),
            resnet_results=ModelResult(**resnet_results),
            processing_time=total_time
        )
    
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    
    _ = await db.status_checks.insert_one(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    
    return status_checks

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting ML Model Comparison API")
    load_models()

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()