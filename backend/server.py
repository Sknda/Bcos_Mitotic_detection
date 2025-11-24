from fastapi import FastAPI, APIRouter, File, UploadFile
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any
import uuid
from datetime import datetime, timezone
import base64
import io
from PIL import Image, ImageDraw, ImageFilter
import random
import numpy as np

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

# Define Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class ModelResult(BaseModel):
    prediction: str
    confidence: float
    explainability_images: List[str]  # base64 encoded images
    explainability_summary: str

class AnalysisResponse(BaseModel):
    success: bool
    original_image: str  # base64
    model_a: ModelResult
    model_b: ModelResult

# ML Model Simulation Functions
def generate_mock_gradcam(image: Image.Image, intensity: float = 0.6) -> str:
    """Generate a mock Grad-CAM heatmap overlay"""
    # Create a copy of the image
    img_array = np.array(image.convert('RGB'))
    height, width = img_array.shape[:2]
    
    # Create heatmap with random hot spots
    heatmap = np.zeros((height, width))
    num_spots = random.randint(2, 4)
    
    for _ in range(num_spots):
        cx, cy = random.randint(0, width-1), random.randint(0, height-1)
        y, x = np.ogrid[:height, :width]
        mask = ((x - cx)**2 + (y - cy)**2) <= (min(width, height) / 4)**2
        heatmap[mask] += random.uniform(0.5, 1.0)
    
    # Normalize heatmap
    heatmap = np.clip(heatmap, 0, 1)
    
    # Apply colormap (red-yellow for hot regions)
    heatmap_colored = np.zeros((height, width, 3), dtype=np.uint8)
    heatmap_colored[:, :, 0] = (heatmap * 255).astype(np.uint8)  # Red channel
    heatmap_colored[:, :, 1] = (heatmap * 200).astype(np.uint8)  # Green channel
    
    # Blend with original image
    result = (img_array * (1 - intensity) + heatmap_colored * intensity).astype(np.uint8)
    result_image = Image.fromarray(result)
    
    # Convert to base64
    buffer = io.BytesIO()
    result_image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def generate_mock_segmentation(image: Image.Image) -> str:
    """Generate a mock segmentation mask"""
    img = image.convert('RGB')
    width, height = img.size
    
    # Create a new image for segmentation
    seg_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(seg_img)
    
    # Draw random segmentation regions
    num_regions = random.randint(3, 6)
    colors = [
        (255, 100, 100, 180),  # Red
        (100, 255, 100, 180),  # Green
        (100, 100, 255, 180),  # Blue
        (255, 255, 100, 180),  # Yellow
        (255, 100, 255, 180),  # Magenta
        (100, 255, 255, 180),  # Cyan
    ]
    
    for i in range(num_regions):
        x = random.randint(0, width - 50)
        y = random.randint(0, height - 50)
        w = random.randint(50, width // 2)
        h = random.randint(50, height // 2)
        color = colors[i % len(colors)]
        draw.ellipse([x, y, x + w, y + h], fill=color)
    
    # Composite with original
    result = Image.alpha_composite(img.convert('RGBA'), seg_img)
    
    # Convert to base64
    buffer = io.BytesIO()
    result.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def run_modelA(image: Image.Image) -> Dict[str, Any]:
    """Simulate Model A inference"""
    categories = ['Cat', 'Dog', 'Bird', 'Car', 'Bicycle', 'Person', 'Tree', 'Building']
    prediction = random.choice(categories)
    confidence = round(random.uniform(0.82, 0.97), 4)
    
    # Generate explainability outputs
    gradcam = generate_mock_gradcam(image, intensity=0.5)
    segmentation = generate_mock_segmentation(image)
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'explainability_images': [gradcam, segmentation],
        'explainability_summary': f'Model A identified key features in the {prediction.lower()} with high attention to central regions and edge patterns.'
    }

def run_modelB(image: Image.Image) -> Dict[str, Any]:
    """Simulate Model B inference"""
    categories = ['Feline', 'Canine', 'Avian', 'Vehicle', 'Cycle', 'Human', 'Flora', 'Structure']
    prediction = random.choice(categories)
    confidence = round(random.uniform(0.78, 0.95), 4)
    
    # Generate explainability outputs
    gradcam = generate_mock_gradcam(image, intensity=0.6)
    segmentation = generate_mock_segmentation(image)
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'explainability_images': [gradcam, segmentation],
        'explainability_summary': f'Model B focused on texture and shape characteristics, emphasizing the {prediction.lower()} classification through pattern recognition.'
    }

# API Routes
@api_router.get("/")
async def root():
    return {"message": "ML Model Comparison API"}

@api_router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """Analyze uploaded image with both models"""
    try:
        # Read and process uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Resize if too large
        max_size = 800
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert original image to base64
        buffer = io.BytesIO()
        image.convert('RGB').save(buffer, format='PNG')
        original_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Run both models
        result_a = run_modelA(image)
        result_b = run_modelB(image)
        
        return AnalysisResponse(
            success=True,
            original_image=original_b64,
            model_a=ModelResult(**result_a),
            model_b=ModelResult(**result_b)
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

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()