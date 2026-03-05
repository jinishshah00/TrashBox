# TrashBox Waste Classifier

A complete image classification app that identifies **7 types of waste** from photos and tells you how to dispose of them properly.

| Class | Examples |
|-------|----------|
| Cardboard | Boxes, packaging |
| E-waste | Cables, chips, laptops, phones, appliances |
| Glass | Bottles, jars |
| Medical | Syringes, gloves, masks, medicines |
| Metal | Cans, scrap, spray cans, containers |
| Paper | Newspaper, cups, tetra pak |
| Plastic | Bags, bottles, containers, cups, cigarette butts |

Built on the [TrashBox dataset](https://github.com/nikhilvenkatkumsetty/TrashBox) (17,785 images) using transfer learning with EfficientNet-B0.

---

## Quick Start

### 1. Install dependencies

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Train the model

Make sure the dataset folders (`TrashBox/` and `TrashBox-testandvalid/`) are in the project root.

```bash
python train.py
```

This will:
- Load 14,282 training images and 1,781 validation images
- Train EfficientNet-B0 with frozen backbone → then fine-tune all layers
- Save the best model to `model_output/best_model.pth`
- Print test accuracy at the end

**Training options:**

```bash
# Use MobileNetV3 (faster, lower accuracy)
python train.py --arch mobilenet

# Use ResNet-50 (heavier but robust)
python train.py --arch resnet

# Custom hyperparameters
python train.py --epochs 20 --batch-size 64 --lr 0.0005

# Don't freeze backbone (train from scratch)
python train.py --no-freeze
```

Training takes roughly **15–30 minutes** on Apple Silicon (MPS) or a GPU, longer on CPU.

### 3. Launch the web app

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

---

## Project Structure

```
trash-classifier/
├── train.py              # Training script (EfficientNet / MobileNet / ResNet)
├── app.py                # FastAPI backend serving predictions
├── static/
│   └── index.html        # Web UI (single-page, no build step)
├── requirements.txt      # Python dependencies
├── .gitignore
├── model_output/         # Created after training
│   ├── best_model.pth    # Best model weights
│   ├── model_meta.json   # Arch, classes, normalization params
│   ├── history.json      # Training metrics per epoch
│   └── checkpoint.pth    # Full checkpoint for resuming
├── TrashBox/             # Training dataset (cloned separately)
└── TrashBox-testandvalid/ # Test/val dataset (cloned separately)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Health check (model loaded?) |
| `GET` | `/classes` | List class names + disposal tips |
| `POST` | `/predict` | Upload image file → classification |
| `POST` | `/predict/base64` | Send base64 image → classification |

### Example API call

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg"
```

Response:
```json
{
  "success": true,
  "prediction": {
    "class_name": "plastic",
    "confidence": 94.32,
    "bin": "Recycling (check local rules)",
    "tip": "Check the resin code...",
    "icon": "🧴"
  },
  "all_scores": [...]
}
```

## Features

- **Drag & drop** or click to upload images
- **Camera capture** on mobile devices
- **Real-time classification** with confidence scores
- **Disposal tips** for each waste category
- **Responsive** dark-themed UI
- Supports **JPEG, PNG, WEBP** up to 10 MB

## Dataset

- **Training:** 14,282 images across 7 classes
- **Validation:** 1,781 images
- **Test:** 1,793 images

Citation: Kumsetty, N.V., Nekkare, A.B., S, A.K., & Bahera, A. (2022). *TrashBox: Trash Detection and Classification using Quantum Transfer Learning.* 31st Conference of Open Innovations Association (FRUCT), IEEE.
