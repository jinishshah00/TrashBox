#!/usr/bin/env python3
"""
train_full.py — Unified trash‑classifier training.

Trains two EfficientNet‑B0 models:
  V1  – Parent‑category classifier   (12 classes)
  V2  – Sub‑category classifier      (32 classes, only for parents that have subs)

Features:
  • Auto‑discovers categories from the dataset folders on disk
  • Auto‑splits 80/10/10 when test or val data is missing
  • Caps V1 training images per class to keep CPU training feasible
  • Weighted sampling to handle class imbalance
  • Freeze→unfreeze backbone schedule
"""

import json, random, time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
BASE = Path(__file__).resolve().parent

TRAIN_FLAT = BASE / "TrashBox" / "TrashBox_train_set"
TRAIN_SUB  = BASE / "TrashBox" / "TrashBox_train_dataset_subfolders"
TV_FLAT    = BASE / "TrashBox-testandvalid" / "TrashBox_testandvalid_set"
TV_SUB     = BASE / "TrashBox-testandvalid" / "TrashBox_testandvalid_dataset_subfolders"
OUT        = BASE / "model_output_final"

IMG_SIZE    = 224
BATCH       = 32
WORKERS     = 0          # 0 is safest on macOS
EPOCHS      = 15
LR_HEAD     = 1e-3       # learning‑rate while backbone is frozen
LR_FINE     = 1e-4       # learning‑rate after unfreezing
FREEZE_EP   = 5          # unfreeze backbone after this many epochs
MAX_V1      = 3000       # cap per‑class train images for V1
SEED        = 42

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
SKIP = {'.DS_Store', '__MACOSX', '.ipynb_checkpoints'}

random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def norm(name: str) -> str:
    """Normalize a folder name: lowercase, spaces/hyphens → underscores."""
    return name.lower().replace(' ', '_').replace('-', '_')


def list_imgs(folder: Path | None) -> list[Path]:
    """Return sorted image paths inside *folder*."""
    if folder is None or not folder.exists():
        return []
    return sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in EXTS
    )


def split3(imgs: list[Path]):
    """Shuffle *imgs* and return (train, test, val) with 80/10/10 split."""
    xs = list(imgs)
    random.shuffle(xs)
    n = len(xs)
    nt = max(1, int(n * 0.10))
    nv = max(1, int(n * 0.10))
    return xs[nt + nv:], xs[:nt], xs[nt:nt + nv]


def weighted_sampler(samples):
    """Create a WeightedRandomSampler that balances classes."""
    labels = [s[1] for s in samples]
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, len(weights))


# ══════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════

class TrashDataset(Dataset):
    """Simple dataset from a list of (path, class_index) tuples."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        if self.transform:
            img = self.transform(img)
        return img, label


# ══════════════════════════════════════════════════════════════════════
# TRANSFORMS
# ══════════════════════════════════════════════════════════════════════

train_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ══════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════

def collect_samples():
    """
    Scan every data directory and return:
        v1_samples  – {train/test/val: [(path, idx), ...]}
        v2_samples  – same, for subcategory model
        v1_classes  – sorted list of normalised parent names
        v2_classes  – sorted list of normalised subcategory names
        parent_map  – {parent_norm: [sub_norm, ...]}  (only parents with subs)
        display     – {norm_name: original_folder_name}
    """
    # ── Step 1: discover parents & subcategories ──────────────────────
    parents = {}   # norm → {display, flat_train, subs}

    # 1a. flat train set
    if TRAIN_FLAT.exists():
        for d in sorted(TRAIN_FLAT.iterdir()):
            if d.is_dir() and d.name not in SKIP:
                n = norm(d.name)
                parents[n] = {'display': d.name, 'flat_train': d, 'subs': {}}

    # 1b. subfolder train set (adds subcategories)
    if TRAIN_SUB.exists():
        for pd in sorted(TRAIN_SUB.iterdir()):
            if not pd.is_dir() or pd.name in SKIP:
                continue
            pn = norm(pd.name)
            if pn not in parents:
                parents[pn] = {'display': pd.name, 'flat_train': None, 'subs': {}}
            for sd in sorted(pd.iterdir()):
                if sd.is_dir() and sd.name not in SKIP:
                    sn = norm(sd.name)
                    parents[pn]['subs'][sn] = {
                        'display': sd.name, 'train': sd,
                        'test': None, 'val': None,
                    }

    # 1c. subfolder test/val
    if TV_SUB.exists():
        for pd in sorted(TV_SUB.iterdir()):
            if not pd.is_dir():
                continue
            pn = norm(pd.name)
            if pn not in parents:
                continue
            for split in ('test', 'val'):
                sp = pd / split
                if not sp.exists():
                    continue
                for sd in sorted(sp.iterdir()):
                    if sd.is_dir() and sd.name not in SKIP:
                        sn = norm(sd.name)
                        if sn in parents[pn]['subs']:
                            parents[pn]['subs'][sn][split] = sd

    # ── Step 2: build class lists ─────────────────────────────────────
    v1_classes = sorted(parents.keys())
    v2_set = set()
    for pi in parents.values():
        for sn in pi['subs']:
            v2_set.add(sn)
    v2_classes = sorted(v2_set)
    v1_idx = {c: i for i, c in enumerate(v1_classes)}
    v2_idx = {c: i for i, c in enumerate(v2_classes)}

    # ── Step 3: build V2 samples (subcategory level) ──────────────────
    v2_samples = {'train': [], 'test': [], 'val': []}
    parent_agg = {}  # pn → {train/test/val: [paths]}
    parent_map = {}
    display = {}

    for pn in v1_classes:
        pi = parents[pn]
        display[pn] = pi['display']
        if not pi['subs']:
            continue
        parent_map[pn] = sorted(pi['subs'].keys())
        parent_agg[pn] = {'train': [], 'test': [], 'val': []}

        for sn in sorted(pi['subs']):
            si = pi['subs'][sn]
            display[sn] = si['display']
            tr = list_imgs(si['train'])
            te = list_imgs(si['test'])
            va = list_imgs(si['val'])

            # auto‑split when test/val are missing
            if not te and not va:
                tr, te, va = split3(tr)
                print(f"  ↳ auto‑split {pn}/{sn}: {len(tr)}t/{len(te)}te/{len(va)}v")
            elif not va:
                random.shuffle(tr)
                nv = max(1, len(tr) // 10)
                va, tr = tr[:nv], tr[nv:]
                print(f"  ↳ auto‑split val  {pn}/{sn}: {len(va)} val from train")
            elif not te:
                random.shuffle(tr)
                nt = max(1, len(tr) // 10)
                te, tr = tr[:nt], tr[nt:]

            v2_samples['train'].extend((p, v2_idx[sn]) for p in tr)
            v2_samples['test'].extend((p, v2_idx[sn]) for p in te)
            v2_samples['val'].extend((p, v2_idx[sn]) for p in va)
            parent_agg[pn]['train'].extend(tr)
            parent_agg[pn]['test'].extend(te)
            parent_agg[pn]['val'].extend(va)

    # ── Step 4: build V1 samples (parent level) ──────────────────────
    v1_samples = {'train': [], 'test': [], 'val': []}

    for pn in v1_classes:
        pi = parents[pn]
        has_subs = bool(pi['subs'])

        if has_subs:
            # prefer flat sets when they exist; otherwise aggregate V2 splits
            ft_dir = TV_FLAT / "test" / pi['display'] if TV_FLAT.exists() else None
            fv_dir = TV_FLAT / "val"  / pi['display'] if TV_FLAT.exists() else None
            if (pi['flat_train']
                    and ft_dir and ft_dir.exists()
                    and fv_dir and fv_dir.exists()):
                tr = list_imgs(pi['flat_train'])
                te = list_imgs(ft_dir)
                va = list_imgs(fv_dir)
            else:
                tr = parent_agg[pn]['train']
                te = parent_agg[pn]['test']
                va = parent_agg[pn]['val']
        else:
            tr = list_imgs(pi['flat_train'])
            ft_dir = TV_FLAT / "test" / pi['display'] if TV_FLAT.exists() else None
            fv_dir = TV_FLAT / "val"  / pi['display'] if TV_FLAT.exists() else None
            te = list_imgs(ft_dir) if ft_dir and ft_dir.exists() else []
            va = list_imgs(fv_dir) if fv_dir and fv_dir.exists() else []

            if not te and not va:
                tr, te, va = split3(tr)
                print(f"  ↳ auto‑split V1 {pn}: {len(tr)}t/{len(te)}te/{len(va)}v")
            elif not va:
                random.shuffle(tr)
                nv = max(1, len(tr) // 10)
                va, tr = tr[:nv], tr[nv:]
                print(f"  ↳ auto‑split V1 val {pn}: {len(va)} from train")
            elif not te:
                random.shuffle(tr)
                nt = max(1, len(tr) // 10)
                te, tr = tr[:nt], tr[nt:]

        # cap
        if len(tr) > MAX_V1:
            random.shuffle(tr)
            tr = tr[:MAX_V1]

        v1_samples['train'].extend((p, v1_idx[pn]) for p in tr)
        v1_samples['test'].extend((p, v1_idx[pn]) for p in te)
        v1_samples['val'].extend((p, v1_idx[pn]) for p in va)

    return v1_samples, v2_samples, v1_classes, v2_classes, parent_map, display


# ══════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════

def make_model(num_classes: int):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return model


# ══════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def train(model, train_ld, val_ld, test_ld, class_names, out_dir, tag=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()

    # freeze backbone
    for p in model.features.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(model.classifier.parameters(), lr=LR_HEAD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)

    model.to(DEVICE)
    best_val = 0.0

    for ep in range(EPOCHS):
        t0 = time.time()

        # unfreeze at FREEZE_EP
        if ep == FREEZE_EP:
            print(f"  ── unfreezing backbone at epoch {ep + 1} ──")
            for p in model.features.parameters():
                p.requires_grad = True
            opt = torch.optim.Adam(model.parameters(), lr=LR_FINE)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)

        # --- train ---
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for imgs, labels in train_ld:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)

        # --- validate ---
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for imgs, labels in val_ld:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                vc += (model(imgs).argmax(1) == labels).sum().item()
                vt += labels.size(0)
        val_acc = vc / vt if vt else 0
        sched.step(1 - val_acc)

        elapsed = time.time() - t0
        star = ""
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            star = " ★"

        print(f"  {tag}Ep {ep + 1:2d}/{EPOCHS} │ {elapsed:5.0f}s │ "
              f"loss {loss_sum / total:.4f}  train {correct / total:.4f}  "
              f"val {val_acc:.4f}{star}")

    # --- test ---
    model.load_state_dict(torch.load(out_dir / "best_model.pth",
                                     map_location=DEVICE, weights_only=True))
    model.eval()
    tc, tt = 0, 0
    with torch.no_grad():
        for imgs, labels in test_ld:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            tc += (model(imgs).argmax(1) == labels).sum().item()
            tt += labels.size(0)
    test_acc = tc / tt if tt else 0
    print(f"\n  {tag}Test accuracy: {test_acc:.4f}  ({tc}/{tt})")

    # save metadata
    with open(out_dir / "class_names.json", 'w') as f:
        json.dump(class_names, f, indent=2)
    info = {'num_classes': len(class_names), 'best_val_acc': best_val,
            'test_acc': test_acc, 'epochs': EPOCHS, 'image_size': IMG_SIZE}
    with open(out_dir / "training_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    return model, test_acc


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("═" * 56)
    print("   TRASH CLASSIFIER — FULL TRAINING")
    print("═" * 56)
    print(f"Device : {DEVICE}")
    print(f"Epochs : {EPOCHS}   Batch : {BATCH}   Max/class(V1) : {MAX_V1}")

    # ── collect ───────────────────────────────────────────────────────
    print("\n▸ Scanning & splitting data …")
    v1_s, v2_s, v1_c, v2_c, pmap, disp = collect_samples()

    print(f"\n  V1 — {len(v1_c)} parent classes:")
    for c in v1_c:
        subs = pmap.get(c)
        label = f"→ {', '.join(subs)}" if subs else "(flat)"
        print(f"    {c:20s} {label}")

    print(f"\n  V2 — {len(v2_c)} subcategory classes")
    for split in ('train', 'test', 'val'):
        print(f"    V1 {split:5s}: {len(v1_s[split]):>6,}    "
              f"V2 {split:5s}: {len(v2_s[split]):>6,}")

    # ── V1 ────────────────────────────────────────────────────────────
    print(f"\n▸ Phase 1 — Training V1 (parent classifier, {len(v1_c)} classes) …\n")
    v1_model = make_model(len(v1_c))

    tr_ds = TrashDataset(v1_s['train'], train_tfm)
    va_ds = TrashDataset(v1_s['val'],   val_tfm)
    te_ds = TrashDataset(v1_s['test'],  val_tfm)

    tr_ld = DataLoader(tr_ds, BATCH, sampler=weighted_sampler(v1_s['train']),
                       num_workers=WORKERS)
    va_ld = DataLoader(va_ds, BATCH, shuffle=False, num_workers=WORKERS)
    te_ld = DataLoader(te_ds, BATCH, shuffle=False, num_workers=WORKERS)

    train(v1_model, tr_ld, va_ld, te_ld, v1_c, OUT / "v1", tag="[V1] ")

    # ── V2 ────────────────────────────────────────────────────────────
    if v2_s['train']:
        print(f"\n▸ Phase 2 — Training V2 (subcategory classifier, {len(v2_c)} classes) …\n")
        v2_model = make_model(len(v2_c))

        tr_ds = TrashDataset(v2_s['train'], train_tfm)
        va_ds = TrashDataset(v2_s['val'],   val_tfm)
        te_ds = TrashDataset(v2_s['test'],  val_tfm)

        tr_ld = DataLoader(tr_ds, BATCH, sampler=weighted_sampler(v2_s['train']),
                           num_workers=WORKERS)
        va_ld = DataLoader(va_ds, BATCH, shuffle=False, num_workers=WORKERS)
        te_ld = DataLoader(te_ds, BATCH, shuffle=False, num_workers=WORKERS)

        train(v2_model, tr_ld, va_ld, te_ld, v2_c, OUT / "v2", tag="[V2] ")

        # save parent→sub mapping
        with open(OUT / "v2" / "parent_map.json", 'w') as f:
            json.dump(pmap, f, indent=2)
    else:
        print("\n  (no subcategories found — skipping V2)")

    # save display‑name mapping
    with open(OUT / "display_names.json", 'w') as f:
        json.dump(disp, f, indent=2)

    print("\n" + "═" * 56)
    print("   TRAINING COMPLETE")
    print("═" * 56)


if __name__ == "__main__":
    main()
