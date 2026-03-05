"""
app.py — FastAPI server for the TrashBox classifier.

Loads V1 (parent) and optional V2 (subcategory) models from model_output_final/.
Serves a web UI at / and a prediction API at /predict.
"""

import json
from io import BytesIO
from pathlib import Path
import base64

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model_output_final"
IMG_SIZE = 224

DEVICE = torch.device(
    "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

DISPLAY_OVERRIDE = {
    'stroform_product': 'Styrofoam',
    'e_waste': 'E-Waste',
    'food_organic': 'Food & Organic',
    'non_recyclable': 'Non-Recyclable',
    'tetra_pak': 'Tetra Pak',
    'news_paper': 'Newspaper',
    'multi_layered_wrappers': 'Multi-Layer Wrappers',
    'other_metal_objects': 'Other Metal Objects',
    'cigarette_butt': 'Cigarette Butts',
    'small_appliances': 'Small Appliances',
    'electronic_chips': 'Electronic Chips',
    'electrical_cables': 'Electrical Cables',
    'ceramic_product': 'Ceramics',
    'sanitary_napkin': 'Sanitary Napkins',
    'spray_cans': 'Spray Cans',
    'beverage_cans': 'Beverage Cans',
    'construction_scrap': 'Construction Scrap',
    'metal_containers': 'Metal Containers',
    'plastic_bags': 'Plastic Bags',
    'plastic_bottles': 'Plastic Bottles',
    'plastic_containers': 'Plastic Containers',
    'plastic_cups': 'Plastic Cups',
    'paper_cups': 'Paper Cups',
    'light_bulbs': 'Light Bulbs',
}

DISPOSAL_TIPS = {
    'cardboard':       'Flatten boxes and recycle. Remove tape and staples. Keep dry.',
    'clothes':         'Donate wearable items. Use textile recycling for worn clothing. Do not put in regular recycling.',
    'e_waste':         'Take to a certified e‑waste recycler or retailer take‑back programme. Never put in regular trash.',
    'food_organic':    'Compost at home or use your municipal organic‑waste bin. Avoid plastic bags for collection.',
    'glass':           'Rinse and recycle by colour if required. Remove caps/lids.',
    'hazardous':       'Take to a hazardous‑waste collection facility. Never pour down the drain or put in regular trash.',
    'medical':         'Use designated medical‑waste bins. Place sharps in puncture‑proof containers.',
    'metal':           'Rinse containers and recycle. Crush cans to save space.',
    'non_recyclable':  'Place in general waste. These items cannot be recycled through standard programmes.',
    'paper':           'Recycle if clean and dry. Shred sensitive documents first. Remove plastic windows from envelopes.',
    'plastic':         'Check the recycling number. Rinse containers. Remove caps if your area requires it.',
    'shoes':           'Donate wearable pairs. Some brands offer take‑back programmes. Otherwise general waste.',
}

SUBCATEGORY_TIPS = {
    'batteries':              'Take to a battery‑recycling drop‑off point. Never put in regular trash — fire risk.',
    'beverage_cans':          'Rinse and crush. Aluminium is infinitely recyclable.',
    'ceramic_product':        'Cannot be recycled with glass. Place in general waste or construction debris.',
    'cigarette_butt':         'Dispose in designated bins. Some programmes (e.g. TerraCycle) recycle these.',
    'construction_scrap':     'Take to a metal‑recycling facility or scrap yard.',
    'diapers':                'Place in general waste. Cannot be recycled. Consider compostable alternatives.',
    'electrical_cables':      'Strip copper/metal for recycling. Take to an e‑waste facility.',
    'electronic_chips':       'E‑waste recycler. Contains recoverable precious metals.',
    'gloves':                 'Medical‑waste bin if clinical; otherwise general waste.',
    'laptops':                'E‑waste recycling or manufacturer take‑back. Wipe data first.',
    'light_bulbs':            'CFLs/fluorescents → hazardous waste (mercury). LEDs → e‑waste. Incandescent → general waste.',
    'masks':                  'Medical‑waste bin. Cut ear‑loops before disposal to protect wildlife.',
    'medicines':              'Return to a pharmacy take‑back programme. Do not flush.',
    'metal_containers':       'Rinse and recycle with regular metal recycling.',
    'multi_layered_wrappers': 'Not recyclable in most programmes. Place in general waste.',
    'news_paper':             'Recycle — bundle or bag loosely. Keep dry.',
    'other_metal_objects':    'Scrap yard or bulky‑metal collection depending on size.',
    'paints':                 'Latex paint: dry out and general waste. Oil‑based: hazardous‑waste facility.',
    'paper':                  'Recycle if clean. Shred sensitive documents first.',
    'paper_cups':             'Often not recyclable due to plastic lining. Check local guidelines.',
    'pesticides':             'Hazardous‑waste facility. Never pour down the drain.',
    'plastic_bags':           'Return to store drop‑off bins. Do not put in curbside recycling.',
    'plastic_bottles':        'Rinse, remove cap, recycle. Most widely accepted plastic type.',
    'plastic_containers':     'Check recycling number, rinse, recycle. Remove lids if required.',
    'plastic_cups':           'Check if your area accepts them. Rinse first.',
    'sanitary_napkin':        'Place in general waste. Cannot be recycled. Wrap before disposal.',
    'small_appliances':       'E‑waste recycler or retailer take‑back programme.',
    'smartphones':            'E‑waste recycling. Many carriers offer trade‑in. Wipe data first.',
    'spray_cans':             'Empty completely, then recycle if your area allows it.',
    'stroform_product':       'Most programmes don\'t accept styrofoam. Look for specialist drop‑off centres.',
    'syringe':                'Sharps container only. Never put loose in trash. Pharmacy disposal.',
    'tetra_pak':              'Rinse and flatten. Recyclable in many areas — check locally.',
}


# ══════════════════════════════════════════════════════════════════════
# DISPLAY HELPER
# ══════════════════════════════════════════════════════════════════════

def pretty(name: str) -> str:
    """Normalised name → human‑friendly display string."""
    if name in DISPLAY_OVERRIDE:
        return DISPLAY_OVERRIDE[name]
    return name.replace('_', ' ').title()


# ══════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════

def load_model(model_dir: Path, tag: str):
    """Load an EfficientNet‑B0 checkpoint + class list."""
    with open(model_dir / "class_names.json") as f:
        class_names = json.load(f)
    n = len(class_names)

    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(1280, 512), nn.ReLU(),
        nn.Dropout(0.2), nn.Linear(512, n),
    )
    state = torch.load(model_dir / "best_model.pth",
                       map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval().to(DEVICE)
    print(f"  ✓ {tag}: {n} classes from {model_dir.name}/")
    return model, class_names


print("Loading models …")
v1_model, v1_classes = None, []
v2_model, v2_classes = None, []
parent_map = {}

v1_dir = MODEL_DIR / "v1"
if v1_dir.exists() and (v1_dir / "best_model.pth").exists():
    v1_model, v1_classes = load_model(v1_dir, "V1 (parent)")

v2_dir = MODEL_DIR / "v2"
if v2_dir.exists() and (v2_dir / "best_model.pth").exists():
    v2_model, v2_classes = load_model(v2_dir, "V2 (subcategory)")
    pm_path = v2_dir / "parent_map.json"
    if pm_path.exists():
        with open(pm_path) as f:
            parent_map = json.load(f)

v2_idx = {c: i for i, c in enumerate(v2_classes)} if v2_classes else {}


# ══════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════

inference_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


SECONDARY_THRESHOLD = 0.15   # flag any parent category above 15 %


def _resolve_subcategory(parent_name: str, tensor):
    """Return (sub_name, sub_conf, all_sub_scores) for a given parent."""
    has_subs = parent_name in parent_map and v2_model is not None
    if not has_subs:
        return None, None, None, False

    with torch.no_grad():
        v2_logits = v2_model(tensor)
    v2_probs = torch.softmax(v2_logits, dim=1)[0]
    relevant = parent_map[parent_name]
    sub_scores = {s: round(v2_probs[v2_idx[s]].item(), 5)
                  for s in relevant if s in v2_idx}
    if sub_scores:
        sub_name = max(sub_scores, key=sub_scores.get)
        return sub_name, sub_scores[sub_name], sub_scores, True
    return None, None, None, True


def classify(image: Image.Image) -> dict:
    if v1_model is None:
        raise RuntimeError("V1 model not loaded — run train_full.py first")

    img = image.convert('RGB')
    tensor = inference_tfm(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        v1_logits = v1_model(tensor)
    v1_probs = torch.softmax(v1_logits, dim=1)[0]
    parent_idx = v1_probs.argmax().item()
    parent_name = v1_classes[parent_idx]
    parent_conf = v1_probs[parent_idx].item()
    all_parent = {v1_classes[i]: round(v1_probs[i].item(), 5)
                  for i in range(len(v1_classes))}

    # primary subcategory
    sub_name, sub_conf, all_sub, has_subs = _resolve_subcategory(parent_name, tensor)

    # ── secondary detections ──────────────────────────────────────
    # Any other parent category scoring above the threshold is likely
    # also present in the image.  We include them with their own
    # subcategory + disposal tip so the user can act on each item.
    secondary = []
    for i, score in enumerate(v1_probs):
        sc = score.item()
        cat = v1_classes[i]
        if cat == parent_name or sc < SECONDARY_THRESHOLD:
            continue
        s_sub, s_sub_conf, s_all_sub, s_has = _resolve_subcategory(cat, tensor)
        secondary.append({
            'parent_category':       cat,
            'parent_display':        pretty(cat),
            'parent_confidence':     round(sc, 5),
            'subcategory':           s_sub,
            'subcategory_display':   pretty(s_sub) if s_sub else None,
            'subcategory_confidence': round(s_sub_conf, 5) if s_sub_conf is not None else None,
            'disposal_tip':          DISPOSAL_TIPS.get(cat, 'Check local waste guidelines.'),
            'subcategory_tip':       SUBCATEGORY_TIPS.get(s_sub) if s_sub else None,
            'has_subcategories':     s_has,
            'all_subcategory_scores': s_all_sub,
        })
    # sort strongest first
    secondary.sort(key=lambda d: d['parent_confidence'], reverse=True)

    return {
        'parent_category':       parent_name,
        'parent_display':        pretty(parent_name),
        'parent_confidence':     round(parent_conf, 5),
        'subcategory':           sub_name,
        'subcategory_display':   pretty(sub_name) if sub_name else None,
        'subcategory_confidence': round(sub_conf, 5) if sub_conf is not None else None,
        'disposal_tip':          DISPOSAL_TIPS.get(parent_name, 'Check local waste guidelines.'),
        'subcategory_tip':       SUBCATEGORY_TIPS.get(sub_name) if sub_name else None,
        'has_subcategories':     has_subs,
        'all_parent_scores':     all_parent,
        'all_subcategory_scores': all_sub,
        'secondary_detections':  secondary,
    }


# ══════════════════════════════════════════════════════════════════════
# FASTAPI
# ══════════════════════════════════════════════════════════════════════

app = FastAPI(title="TrashBox Classifier")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

static_dir = BASE / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(static_dir / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok",
            "v1_loaded": v1_model is not None,
            "v2_loaded": v2_model is not None,
            "parent_classes": len(v1_classes),
            "subcategory_classes": len(v2_classes),
            "device": str(DEVICE)}


@app.get("/classes")
async def classes():
    return {"parents": [{"name": c, "display": pretty(c),
                         "subcategories": parent_map.get(c)}
                        for c in v1_classes]}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    try:
        img = Image.open(BytesIO(data))
    except Exception:
        raise HTTPException(400, "Invalid image file")
    return classify(img)


@app.post("/predict/base64")
async def predict_base64(body: dict):
    b64 = body.get("image", "")
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        img = Image.open(BytesIO(base64.b64decode(b64)))
    except Exception:
        raise HTTPException(400, "Invalid base64 image")
    return classify(img)


# ── run directly ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
