# TrashBox — AI Trash Classification & Disposal Guide

A hierarchical image classification system that identifies **12 categories** and **32 subcategories** of waste from a single photo, provides disposal instructions, and flags multiple items in one image — all powered by EfficientNet-B0 transfer learning and served through a FastAPI web app.

<br>

## Performance

| Model | Classes | Best Val Acc | Test Acc | Test Samples |
|-------|---------|-------------|----------|-------------|
| **V1** (Parent) | 12 | 94.0% | **94.8%** | 4,010 / 4,231 |
| **V2** (Subcategory) | 32 | 88.4% | **88.1%** | 1,422 / 1,615 |

<br>

## Taxonomy

The classifier uses a two-level hierarchy. V1 predicts the parent category; V2 refines it into a subcategory when applicable.

| Parent Category | Subcategories |
|----------------|---------------|
| Cardboard | — *(flat)* |
| Clothes | — *(flat)* |
| E-Waste | Electrical Cables · Electronic Chips · Laptops · Small Appliances · Smartphones |
| Food & Organic | — *(flat)* |
| Glass | — *(flat)* |
| Hazardous | Batteries · Light Bulbs · Paints · Pesticides |
| Medical | Gloves · Masks · Medicines · Syringe |
| Metal | Beverage Cans · Construction Scrap · Metal Containers · Other Metal Objects · Spray Cans |
| Non-Recyclable | Ceramics · Diapers · Multi-Layer Wrappers · Sanitary Napkins · Styrofoam |
| Paper | Newspaper · Paper · Paper Cups · Tetra Pak |
| Plastic | Cigarette Butts · Plastic Bags · Plastic Bottles · Plastic Containers · Plastic Cups |
| Shoes | — *(flat)* |

> *Flat* categories have no subcategories — classification stops at V1.

<br>

## Architecture

### Model

Both V1 and V2 use the same architecture — **EfficientNet-B0** with ImageNet pre-trained weights and a custom classifier head:

```
EfficientNet-B0 Backbone (frozen epochs 1-5, unfrozen 6-15)
    │
    ▼
┌─────────────────────┐
│  Dropout(p=0.3)     │
│  Linear(1280 → 512) │
│  ReLU               │
│  Dropout(p=0.2)     │
│  Linear(512 → N)    │    N = 12 (V1) or 32 (V2)
└─────────────────────┘
    │
    ▼
  Softmax → Class Probabilities
```

### Inference Pipeline

```
                        ┌──────────────┐
   Input Image ───────►│  V1 (Parent)  │──── Top-1 category + all softmax scores
     224×224            │  12 classes   │
                        └──────┬───────┘
                               │
                 ┌─────────────┼──────────────┐
                 │             │              │
          Has subcats?    Secondary      No subcats
                 │        threshold       (flat class)
                 ▼          > 15%             │
         ┌──────────────┐    │                │
         │  V2 (Subcat)  │   ▼                │
         │  32 classes   │  Also flagged      │
         └──────┬───────┘  with tip           │
                │                             │
                ▼                             ▼
         Primary result              Primary result
       (parent + subcat              (parent only
        + disposal tip)              + disposal tip)
```

### Multi-Item Detection

When an image contains multiple types of waste (e.g., medicines inside a plastic bag), the system uses **softmax thresholding** on V1 scores:

$$P(\text{class}_i) = \frac{e^{z_i}}{\sum_{j=1}^{12} e^{z_j}}$$

Any secondary category with $P > 0.15$ (15%) is flagged as "Also Detected" with its own subcategory classification and disposal tip. No retraining required — this leverages the existing probability distribution.

### Training Strategy

| Phase | Epochs | Backbone | Learning Rate | Scheduler |
|-------|--------|----------|--------------|-----------|
| **1 — Head only** | 1–5 | Frozen | $1 \times 10^{-3}$ | ReduceLROnPlateau (patience=2, factor=0.5) |
| **2 — Full fine-tune** | 6–15 | Unfrozen | $1 \times 10^{-4}$ | ReduceLROnPlateau (patience=2, factor=0.5) |

**Data handling:**
- V1 training caps at **3,000 images/class** to prevent dominance by large categories (e.g., `food_organic` has 12,565 images)
- **WeightedRandomSampler** balances class frequencies during training
- Categories without dedicated test/val splits are **auto-split 80/10/10**

**Augmentation (training only):**
- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation: 0.2)

**Normalization (ImageNet stats):**

$$x_{\text{norm}} = \frac{x - \mu}{\sigma}, \quad \mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

<br>

## Dataset

| Split | V1 (Parent) | V2 (Subcategory) |
|-------|------------|-----------------|
| Train | 24,376 | 12,837 |
| Test | 4,231 | 1,615 |
| Val | 4,074 | 1,591 |

Based on the [TrashBox dataset](https://github.com/nikhilvenkatkumsetty/TrashBox), extended with additional categories for hazardous waste, non-recyclable items, clothes, shoes, and food/organic.

> Kumsetty, N.V., Nekkare, A.B., S, A.K., & Bahera, A. (2022). *TrashBox: Trash Detection and Classification using Quantum Transfer Learning.* 31st Conference of Open Innovations Association (FRUCT), IEEE.

<br>

## Training Curves

```
V1 (Parent — 12 classes)                    V2 (Subcategory — 32 classes)

Val Acc                                     Val Acc
 0.94 ┤                        ★★           0.88 ┤                        ★★
 0.92 ┤                  ★ ★ ★               0.87 ┤                  ★ ★ ★
 0.91 ┤               ★                      0.85 ┤               ★
 0.86 ┤            ★                          0.80 ┤            ★
 0.85 ┤   ★ ★ ★ ★                            0.79 ┤            
 0.84 ┤ ★                                    0.77 ┤   ★ ★ ★ ★
      └──┬──┬──┬──┬──┬──┬──┬──┬──┬──►             └──┬──┬──┬──┬──┬──┬──┬──┬──┬──►
         2  4  6  8  10 12 14  Epoch                  2  4  6  8  10 12 14  Epoch
              ▲ unfreeze                                   ▲ unfreeze
```

<br>

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/jinishshah00/trash-classifier.git
cd trash-classifier

python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train (optional — skip if using pre-trained weights)

Place the dataset folders in the project root:

```
trash-classifier/
├── TrashBox/                    # Training data
│   ├── TrashBox_train_set/
│   └── TrashBox_train_dataset_subfolders/
└── TrashBox-testandvalid/       # Test & validation data
    ├── TrashBox_testandvalid_set/
    └── TrashBox_testandvalid_dataset_subfolders/
```

```bash
python train_full.py
```

This runs a 4-phase pipeline: scan → V1 train (15 epochs) → V2 train (15 epochs) → save. Models are saved to `model_output_final/`.

### 3. Run the app

```bash
python app.py
```

Open **http://localhost:8000** — upload an image via drag-and-drop, file picker, or camera.

<br>

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Health check — model status, device, class counts |
| `GET` | `/classes` | List all parent classes with their subcategories |
| `POST` | `/predict` | Upload an image file → classification result |
| `POST` | `/predict/base64` | Send base64-encoded image → classification result |

### Example

```bash
curl -X POST http://localhost:8000/predict -F "file=@photo.jpg"
```

```json
{
  "parent_category": "plastic",
  "parent_display": "Plastic",
  "parent_confidence": 0.687,
  "subcategory": "plastic_bags",
  "subcategory_display": "Plastic Bags",
  "subcategory_confidence": 0.812,
  "disposal_tip": "Check the recycling number. Rinse containers.",
  "subcategory_tip": "Return to store drop-off bins. Do not put in curbside recycling.",
  "has_subcategories": true,
  "all_parent_scores": { "plastic": 0.687, "medical": 0.257, "..." : "..." },
  "all_subcategory_scores": { "plastic_bags": 0.812, "..." : "..." },
  "secondary_detections": [
    {
      "parent_category": "medical",
      "parent_display": "Medical",
      "parent_confidence": 0.257,
      "subcategory": "medicines",
      "subcategory_display": "Medicines",
      "disposal_tip": "Use designated medical-waste bins.",
      "subcategory_tip": "Return to a pharmacy take-back programme. Do not flush."
    }
  ]
}
```

<br>

## Project Structure

```
trash-classifier/
├── train_full.py             # Unified training script (V1 + V2)
├── app.py                    # FastAPI server with prediction endpoints
├── requirements.txt          # Python dependencies
├── static/
│   └── index.html            # Web UI (single-page, no build step)
├── model_output_final/       # Generated after training
│   ├── v1/
│   │   ├── best_model.pth    # V1 weights (12 classes)
│   │   ├── class_names.json
│   │   └── training_info.json
│   ├── v2/
│   │   ├── best_model.pth    # V2 weights (32 classes)
│   │   ├── class_names.json
│   │   ├── parent_map.json   # Parent → subcategory mapping
│   │   └── training_info.json
│   └── display_names.json    # Normalized → display name map
├── TrashBox/                  # Training dataset (not committed)
└── TrashBox-testandvalid/     # Test/val dataset (not committed)
```

<br>

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | EfficientNet-B0 (torchvision) |
| Framework | PyTorch 2.x |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Image Processing | Pillow, torchvision transforms |
| Language | Python 3.12 |

<br>

## License

This project is licensed under the [MIT License](LICENSE).

Dataset: This project uses the [TrashBox dataset](https://github.com/nikhilvenkatkumsetty/TrashBox) — see their repository for dataset licensing terms.
