# Model Files Directory

Place your PyTorch model files here:

## Required Files:

1. **yolov11_model.pt** - YOLOv11 model for cell detection
   - Purpose: Detects and localizes mitotic and non-mitotic cells
   - Expected output: Bounding boxes with class predictions

2. **resnet_model.pt** - ResNet model for classification
   - Purpose: Classifies detected cells as mitotic or non-mitotic
   - Input size: 224x224
   - Expected output: Binary classification (mitotic/non-mitotic)

## Upload Instructions:

### Option 1: Using terminal/SSH
```bash
scp /local/path/yolov11_model.pt user@server:/app/models/
scp /local/path/resnet_model.pt user@server:/app/models/
```

### Option 2: Using file transfer tools
- Upload both .pt files to this directory (/app/models/)
- Ensure files are named exactly as: `yolov11_model.pt` and `resnet_model.pt`

### After uploading:
```bash
sudo supervisorctl restart backend
```

## Model Loading:

The backend will automatically load models on startup:
- YOLOv11: `YOLO(str(YOLO_MODEL_PATH))`
- ResNet: `torch.load(str(RESNET_MODEL_PATH), map_location=device)`

## Note:
If models are not present, the system will use mock models for demonstration purposes.