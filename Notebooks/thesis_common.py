# =====================================================================================
# thesis_common.py  —  Shared library for the 10-CLASS Threat-Aware Waste pipeline.
#   (WithShorterClassification edition: 18 -> 10 materials, + YOLO detection, + fusion model)
#
# Every notebook (1_data_prep .. 8_detection_yolo) does:  import thesis_common as T
# Edit logic HERE once -> re-upload this one file -> all modules update.
# Upload this file as its own tiny Kaggle dataset (e.g. "thesis-common-10c") and add to every notebook.
# =====================================================================================
import importlib, subprocess, sys, os, glob, json, math, random, warnings, shutil

def _ensure(pkg, pip_name=None):
    try: importlib.import_module(pkg)
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or pkg], check=False)
_ensure("timm"); _ensure("albumentations")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
from PIL import Image, ImageOps
from sklearn.metrics import (f1_score, accuracy_score, cohen_kappa_score, roc_auc_score,
                             confusion_matrix, precision_recall_fscore_support)
warnings.filterwarnings("ignore")

# =====================================================================================
# 1. CONFIG  — 10-class label space
# =====================================================================================
MATERIALS = ["plastic","paper","metal","glass","textile",
             "e_waste","battery","sharps","hazardous_chemical","composite"]
CONDITIONS = ["Good","Poor"]
ROUTES     = ["Reuse","Recycle","Landfill","Special"]
NUM_THREAT = 4                                # ordinal 0..3 (LII). data uses 1..3; 0 reserved.
HAZARD = ["sharps","e_waste","battery","hazardous_chemical"]   # + glass-when-broken (via condition)

SEED = 42
IMG_SIZE, BATCH_SIZE, EPOCHS = 224, 32, 24   # v3 "strong" recipe: more epochs (EMA+aug+reg prevent overfit)
BASE_LR, BB_LR = 3e-4, 3e-5
WEIGHT_DECAY, WARMUP_EPOCHS, NUM_WORKERS = 1e-3, 2, 2   # mild WD (v2 used 1e-4); gentle extra regularization only
LOSS_W = {"route":1.0, "condition":0.5, "material":1.0, "threat":1.0}   # fallback if uncertainty-weighting off
SEL_W  = {"route":0.4, "threat":0.4, "material":0.2}   # checkpoint-selection blend; override (e.g. material=1) for single-task ablations
NORM_MEAN, NORM_STD = [0.485,0.456,0.406], [0.229,0.224,0.225]
BACKBONES = ["efficientnet_b0","convnext_tiny","resnet50","mobilenetv3_large_100",
             "convnext_small.fb_in22k_ft_in1k","vit_small_patch14_dinov2.lvd142m"]  # +stronger-pretrained (SC12/SC13)
GRADCAM_BACKBONE = "efficientnet_b0"
FUSION_BACKBONES = ["efficientnet_b0","convnext_tiny","resnet50"]  # module 7 default (override in module 1)
FUSION_FREEZE = True                          # freeze backbones, train fusion+heads only (cheap)

# ---- v3 strengthening switches (all citable; toggle in Module SC1) ----
# --- reliable wins: ON by default (each adds upside with little/no downside on small data) ---
USE_EMA            = True     # exponential moving average of weights (decay warmed up -> no init-lag)
EMA_DECAY          = 0.998
USE_TTA            = True     # test-time augmentation (hflip) at eval -> more stable test logits
USE_CB_FOCAL       = True     # class-balanced focal for material (helps rare hazardous_chemical / composite)
LABEL_SMOOTH       = 0.05     # light label smoothing on the route head
EARLY_STOP_PATIENCE = 6       # stop if val score hasn't improved for N epochs (0 disables). 24 is a ceiling.
# --- experimental: OFF by default (could hurt the strong threat head on this tiny dataset) — flip for ablations ---
USE_UNCERTAINTY_W  = False    # learnable per-task weights (Kendall&Gal 2018); off -> proven fixed LOSS_W
STRONG_AUG         = False    # heavier aug; off -> the proven v2 augmentation (safer for subtle threat cues)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT_DIR = {"train":"train", "val":"val", "test":"test"}

# ---- policy rules (rewritten for 10 classes; see 01_ROUTING_AND_THREAT_REFERENCES_10class.md) ----
POLICY = {
    "version": "2.1-10class",
    "confidence_threshold_route": 0.45,   # lowered from 0.55: 10-class softmax is often <0.55 even when correct
    "reuse_max_threat": 1,
    "threat_force_special_level": 3,   # threat-aware backstop: predicted threat>=3 -> Special (catches
                                       # hazards the material head misclassifies; threat-3 == Special in this taxonomy)
    "material_force_special": ["e_waste","battery","sharps","hazardous_chemical"],
    "material_force_special_priority1": ["battery"],     # lithium folded into battery
    "recyclables": ["plastic","paper","metal","glass","textile"],
    "material_default_threat": {
        "plastic":2,"paper":1,"metal":1,"glass":2,"textile":1,
        "e_waste":3,"battery":3,"sharps":3,"hazardous_chemical":3,"composite":2},
    "material_default_route_good": {
        "plastic":"Recycle","paper":"Recycle","metal":"Recycle","glass":"Recycle","textile":"Reuse",
        "e_waste":"Special","battery":"Special","sharps":"Special","hazardous_chemical":"Special","composite":"Landfill"},
    "material_default_route_poor": {
        "plastic":"Landfill","paper":"Landfill","metal":"Recycle","glass":"Special","textile":"Recycle",
        "e_waste":"Special","battery":"Special","sharps":"Special","hazardous_chemical":"Special","composite":"Landfill"},
}

def set_seed(seed=None):
    seed = SEED if seed is None else seed
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

CONFIG_KEYS = ["MATERIALS","CONDITIONS","ROUTES","NUM_THREAT","HAZARD","SEED","IMG_SIZE","BATCH_SIZE",
               "EPOCHS","BASE_LR","BB_LR","WEIGHT_DECAY","WARMUP_EPOCHS","NUM_WORKERS","LOSS_W",
               "NORM_MEAN","NORM_STD","BACKBONES","GRADCAM_BACKBONE","FUSION_BACKBONES","FUSION_FREEZE",
               "LABEL_SMOOTH","USE_CB_FOCAL","USE_UNCERTAINTY_W","USE_EMA","EMA_DECAY","USE_TTA","STRONG_AUG",
               "EARLY_STOP_PATIENCE"]

def dump_config(path):
    json.dump({k: globals()[k] for k in CONFIG_KEYS}, open(path, "w"), indent=2); return path

def load_config(path):
    cfg = json.load(open(path))
    for k, v in cfg.items(): globals()[k] = v
    global train_tf, eval_tf, apply_tf
    train_tf, eval_tf, apply_tf = build_transforms(globals()["IMG_SIZE"])
    return cfg

# =====================================================================================
# 2. INPUT DISCOVERY
# =====================================================================================
def find_one(filename_glob, required=True):
    MTIME_TIE_SEC = 300   # candidates ingested within this window of each other are treated
                          # as a tie (Kaggle can stamp near-identical mtimes when a dataset
                          # version is processed) — break ties by file size (larger = more
                          # complete) instead of trusting a few-second mtime gap.
    for base in ["/kaggle/input", ".", "..", "/kaggle/working"]:
        hits = glob.glob(os.path.join(base, "**", filename_glob), recursive=True)
        if hits:
            info = [(p, os.path.getmtime(p), os.path.getsize(p)) for p in hits]
            newest_mtime = max(t for _, t, _ in info)
            hits_sorted = sorted(
                info,
                key=lambda x: (round((newest_mtime - x[1]) / MTIME_TIE_SEC), -x[2], -x[1]),
            )
            if len(hits_sorted) > 1:
                print(f"find_one('{filename_glob}'): {len(hits_sorted)} candidates found "
                      f"(likely a stale/duplicate dataset mount) — picking largest within the "
                      f"newest {MTIME_TIE_SEC}s ingestion window:")
                for p, t, sz in hits_sorted:
                    marker = "-> " if (p, t, sz) == hits_sorted[0] else "   "
                    print(f"    {marker}{p}  (mtime={t:.0f}, size={sz} bytes)")
                print("    If this repeats, clean up the dataset on Kaggle so only one copy exists.")
            return hits_sorted[0][0]
    if required:
        raise FileNotFoundError(f"Could not find '{filename_glob}' under /kaggle/input — add the dataset.")
    return None

def find_image_root():
    return os.path.dirname(find_one("train.csv"))

# =====================================================================================
# 3. PREPROCESSING — EXIF, object-aware crop, Gray-World
# =====================================================================================
def load_rgb(path):
    img = ImageOps.exif_transpose(Image.open(path))
    return np.asarray(img.convert("RGB"))

def object_crop(img, bbox, pad=0.2):
    H, W = img.shape[:2]
    bad = bbox is None or any((v is None) or (isinstance(v, float) and math.isnan(v)) for v in bbox)
    if bad:
        side = min(H, W); y1 = (H-side)//2; x1 = (W-side)//2
        return img[y1:y1+side, x1:x1+side]
    x, y, w, h = bbox
    x1, y1, x2, y2 = x*W, y*H, (x+w)*W, (y+h)*H
    cx, cy = (x1+x2)/2, (y1+y2)/2
    side = max(x2-x1, y2-y1) * (1+pad)
    xs, ys = max(0, int(cx-side/2)), max(0, int(cy-side/2))
    xe, ye = min(W, int(cx+side/2)), min(H, int(cy+side/2))
    crop = img[ys:ye, xs:xe]
    return crop if crop.size else img

def gray_world(img):
    f = img.astype(np.float32)
    mean = f.reshape(-1, 3).mean(0) + 1e-6
    f = f * (mean.mean() / mean)
    return np.clip(f, 0, 255).astype(np.uint8)

# =====================================================================================
# 4. TRANSFORMS
# =====================================================================================
USE_ALB = True
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception:
    USE_ALB = False

def build_transforms(img_size=None):
    img_size = img_size or IMG_SIZE
    if USE_ALB:
        if STRONG_AUG:
            train = A.Compose([
                A.Resize(img_size, img_size), A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.12, rotate_limit=15, border_mode=0, p=0.6),
                A.ColorJitter(0.3, 0.3, 0.3, 0.08, p=0.6),
                A.RandomBrightnessContrast(0.2, 0.2, p=0.4),
                A.OneOf([A.GaussianBlur(blur_limit=(3,5)), A.MotionBlur(blur_limit=5), A.GaussNoise()], p=0.3),
                A.CoarseDropout(max_holes=2, max_height=48, max_width=48, p=0.3),
                A.Normalize(NORM_MEAN, NORM_STD), ToTensorV2()])
        else:
            train = A.Compose([
                A.Resize(img_size, img_size), A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, border_mode=0, p=0.5),
                A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.5),
                A.GaussianBlur(blur_limit=(3,5), p=0.2),
                A.CoarseDropout(max_holes=1, max_height=48, max_width=48, p=0.25),
                A.Normalize(NORM_MEAN, NORM_STD), ToTensorV2()])
        ev = A.Compose([A.Resize(img_size, img_size), A.Normalize(NORM_MEAN, NORM_STD), ToTensorV2()])
        def ap(tf, img): return tf(image=img)["image"]
    else:
        import torchvision.transforms as TT
        train = TT.Compose([TT.ToPILImage(), TT.Resize((img_size,img_size)), TT.RandomHorizontalFlip(0.5),
                            TT.ColorJitter(0.2,0.2,0.2,0.05), TT.ToTensor(), TT.Normalize(NORM_MEAN,NORM_STD)])
        ev = TT.Compose([TT.ToPILImage(), TT.Resize((img_size,img_size)),
                         TT.ToTensor(), TT.Normalize(NORM_MEAN,NORM_STD)])
        def ap(tf, img): return tf(img)
    return train, ev, ap

train_tf, eval_tf, apply_tf = build_transforms()

def denorm(t):
    x = t.numpy().transpose(1,2,0) * np.array(NORM_STD) + np.array(NORM_MEAN)
    return np.clip(x, 0, 1)

# =====================================================================================
# 5. MANIFEST LOADING
# =====================================================================================
def _basename_index(image_root, split):
    idx, root = {}, os.path.join(image_root, "Raw", SPLIT_DIR[split])
    for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if p.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):
            idx.setdefault(os.path.basename(p), p)
    return idx

def _resolve(image_root, split, item_id, image_path, bidx):
    ip = str(image_path); base = os.path.basename(ip)
    for c in [os.path.join(image_root, ip),
              os.path.join(image_root, "Raw", SPLIT_DIR[split], ip),
              os.path.join(image_root, "Raw", SPLIT_DIR[split], str(item_id), base)]:
        if os.path.exists(c): return c
    if base in bidx: return bidx[base]
    g = glob.glob(os.path.join(image_root, "Raw", SPLIT_DIR[split], str(item_id), "*"))
    return g[0] if g else None

def prepare_split(csv_path, split, image_root, max_items=None):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if max_items is not None:
        keep = df["item_id"].drop_duplicates().head(max_items)
        df = df[df["item_id"].isin(keep)].reset_index(drop=True)
    df["y_material"]  = df["material"].astype(str).map({m:i for i,m in enumerate(MATERIALS)})
    df["y_condition"] = df["condition"].astype(str).map({c:i for i,c in enumerate(CONDITIONS)})
    df["y_route"]     = df["route"].astype(str).map({r:i for i,r in enumerate(ROUTES)})
    df["y_threat"]    = pd.to_numeric(df["threat"], errors="coerce").clip(0, NUM_THREAT-1)
    bidx = _basename_index(image_root, split)
    df["abspath"] = [_resolve(image_root, split, it, ip, bidx)
                     for it, ip in zip(df["item_id"], df["image_path"])]
    miss = df["abspath"].isna().sum()
    if miss: print(f"[{split}] WARNING: {miss} images unresolved (dropped)")
    bad_mat = df["y_material"].isna().sum()
    if bad_mat: print(f"[{split}] WARNING: {bad_mat} rows have material not in MATERIALS (dropped)")
    df = df.dropna(subset=["abspath","y_material","y_condition","y_route","y_threat"]).reset_index(drop=True)
    df["y_threat"] = df["y_threat"].astype(int)
    for c in ["bbox_x","bbox_y","bbox_w","bbox_h"]:
        if c not in df.columns: df[c] = np.nan
    return df

def load_manifests(image_root, max_items=None):
    df_tr = prepare_split(os.path.join(image_root, "train.csv"),      "train", image_root, max_items)
    df_va = prepare_split(os.path.join(image_root, "validation.csv"), "val",   image_root, max_items)
    df_te = prepare_split(os.path.join(image_root, "test.csv"),       "test",  image_root, max_items)
    print("rows:", {"train":len(df_tr), "val":len(df_va), "test":len(df_te)})
    return df_tr, df_va, df_te

def check_leakage(df_tr, df_va, df_te):
    s_tr, s_va, s_te = set(df_tr.item_id), set(df_va.item_id), set(df_te.item_id)
    assert s_tr.isdisjoint(s_va) and s_tr.isdisjoint(s_te) and s_va.isdisjoint(s_te), "ITEM LEAKAGE DETECTED!"
    print("Leakage check passed. items train/val/test:", len(s_tr), len(s_va), len(s_te))

# =====================================================================================
# 6. DATASET + LOADERS
# =====================================================================================
class WasteDataset(Dataset):
    def __init__(self, df, train=False):
        self.df = df.reset_index(drop=True); self.train = train
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = load_rgb(r["abspath"])
        img = object_crop(img, (r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]))
        img = gray_world(img)
        x = apply_tf(train_tf if self.train else eval_tf, img)
        y = {"material":  torch.tensor(int(r["y_material"])),
             "condition": torch.tensor(float(r["y_condition"])),
             "threat":    torch.tensor(int(r["y_threat"])),
             "route":     torch.tensor(int(r["y_route"]))}
        return x, y, str(r["item_id"])

def make_sampler(df):
    counts = df["y_material"].value_counts().to_dict()
    w = df["y_material"].map(lambda m: 1.0/counts[m]).to_numpy()
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(df), replacement=True)

def make_loaders(df_tr, df_va, df_te):
    tr = DataLoader(WasteDataset(df_tr, True), batch_size=BATCH_SIZE, sampler=make_sampler(df_tr),
                    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    va = DataLoader(WasteDataset(df_va, False), batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)
    te = DataLoader(WasteDataset(df_te, False), batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)
    return tr, va, te

# =====================================================================================
# 7. MODELS — (a) shared backbone + 4 heads   (b) multi-backbone FUSION
# =====================================================================================
def _feat_dim(backbone):
    """Actual pooled-feature width by a dummy forward — robust to models (e.g. mobilenetv3) whose
    `.num_features` (960) differs from the real forward output (1280 via conv_head)."""
    was = backbone.training; backbone.eval()
    with torch.no_grad():
        d = backbone(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)).shape[-1]
    backbone.train(was)
    return int(d)

def _make_backbone(name, pretrained=True):
    """Create a timm backbone returning pooled (B,D) features. Tries img_size (so ViTs like DINOv2,
    pretrained at 518, interpolate their pos-embeddings to IMG_SIZE) with avg pooling first — this keeps
    the existing CNNs byte-identical (they reject img_size -> fall through to the plain avg-pool attempt).
    Falls back to the model's DEFAULT pooling when avg conflicts with the pretrained norm: DINOv2 ships a
    CLS-token `norm`, not the `fc_norm` that global_pool='avg' expects, which raised a state_dict error."""
    base = dict(pretrained=pretrained, num_classes=0)
    for extra in (dict(img_size=IMG_SIZE, global_pool="avg"),   # CNNs (reject img_size) + standard ViTs
                  dict(img_size=IMG_SIZE),                       # ViT default pooling (DINOv2 -> CLS token)
                  dict(global_pool="avg"),                       # CNNs without an img_size kwarg
                  dict()):                                       # bare default
        try:
            return timm.create_model(name, **base, **extra)
        except (TypeError, RuntimeError):
            continue
    return timm.create_model(name, **base)                       # last resort: surface a clean error

class Head(nn.Module):
    def __init__(self, d_in, d_out, hidden=256, p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden),
                                 nn.GELU(), nn.Dropout(p), nn.Linear(hidden, d_out))
    def forward(self, x): return self.net(x)

class MultiTaskNet(nn.Module):
    """Single shared backbone -> 4 heads (modules 2-5)."""
    def __init__(self, backbone_name):
        super().__init__()
        self.backbone = _make_backbone(backbone_name)
        d = _feat_dim(self.backbone)
        self.h_material  = Head(d, len(MATERIALS))
        self.h_condition = Head(d, 1)
        self.h_route     = Head(d, len(ROUTES))
        self.h_threat    = Head(d, NUM_THREAT-1)
    def forward(self, x):
        f = self.backbone(x)
        return {"material":  self.h_material(f), "condition": self.h_condition(f).squeeze(1),
                "route":     self.h_route(f),    "threat":    self.h_threat(f)}

class MultiBackboneFusion(nn.Module):
    """Feature-level FUSION (module 7): concat pooled features of several backbones -> fusion MLP -> 4 heads.
    With freeze=True the backbones are frozen ImageNet/feature extractors and only the fusion+heads train
    (cheap, ~few MB of grads). Set freeze=False to fine-tune everything (heavier)."""
    def __init__(self, backbone_names, freeze=True, proj=256, weight_paths=None,
                 match_checkpoint=False):
        super().__init__()
        self.names = list(backbone_names); self.freeze = freeze
        self.bbs = nn.ModuleList(); dims = []
        for nm in self.names:
            # if a fine-tuned MultiTaskNet weight is supplied, start from its DOMAIN-ADAPTED backbone
            # (much stronger than frozen ImageNet features); else fall back to ImageNet pretrained.
            wp = (weight_paths or {}).get(nm)
            m = _make_backbone(nm, pretrained=(wp is None))
            if wp:
                sd = torch.load(wp, map_location="cpu")
                bb_sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
                if match_checkpoint:
                    # With pretrained=False no construction attempt fails, so _make_backbone may pick a
                    # DIFFERENT variant than the one the checkpoint was trained with (e.g. DINOv2 trains
                    # with CLS-token `norm`, but the offline rebuild gets `fc_norm`+avg-pool and silently
                    # drops the trained norm). Try the same construction ladder and keep the variant whose
                    # state_dict matches the checkpoint exactly.
                    best, best_bad = m, None
                    for extra in (dict(img_size=IMG_SIZE, global_pool="avg"), dict(img_size=IMG_SIZE),
                                  dict(global_pool="avg"), dict()):
                        try:
                            cand = timm.create_model(nm, pretrained=False, num_classes=0, **extra)
                        except (TypeError, RuntimeError):
                            continue
                        r = cand.load_state_dict(bb_sd, strict=False)
                        bad = len(r.missing_keys) + len(r.unexpected_keys)
                        if best_bad is None or bad < best_bad:
                            best, best_bad = cand, bad
                        if bad == 0:
                            break
                    m = best
                    if best_bad:
                        print(f"  fusion[{nm}] WARNING: no construction matches the checkpoint exactly "
                              f"({best_bad} mismatched tensors) - using the closest variant.")
                res = m.load_state_dict(bb_sd, strict=False)
                n_bad = len(res.missing_keys) + len(res.unexpected_keys)
                print(f"  fusion[{nm}] loaded fine-tuned backbone ({len(bb_sd)} tensors"
                      + (f", {n_bad} MISMATCHED" if n_bad else "") + ")")
            if freeze:
                m.eval()
                for p in m.parameters(): p.requires_grad = False
            self.bbs.append(m); dims.append(_feat_dim(m))
        D = sum(dims)
        self.fuse = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, proj*2), nn.GELU(),
                                  nn.Dropout(0.3), nn.Linear(proj*2, proj), nn.GELU())
        self.h_material  = Head(proj, len(MATERIALS))
        self.h_condition = Head(proj, 1)
        self.h_route     = Head(proj, len(ROUTES))
        self.h_threat    = Head(proj, NUM_THREAT-1)
    def _extract(self, x):
        feats = []
        for m in self.bbs:
            if self.freeze:
                with torch.no_grad(): feats.append(m(x))
            else:
                feats.append(m(x))
        return torch.cat(feats, dim=1)
    def forward(self, x):
        f = self.fuse(self._extract(x))
        return {"material":  self.h_material(f), "condition": self.h_condition(f).squeeze(1),
                "route":     self.h_route(f),    "threat":    self.h_threat(f)}

# =====================================================================================
# 8. LOSSES
# =====================================================================================
def focal_loss(logits, target, gamma=2.0, alpha=None):
    logp = F.log_softmax(logits, dim=1)
    logpt = logp.gather(1, target.view(-1,1)).squeeze(1); pt = logpt.exp()
    loss = -((1-pt)**gamma) * logpt
    if alpha is not None: loss = loss * alpha[target]      # class-balanced focal (helps rare classes)
    return loss.mean()

def corn_loss(logits, y, num_classes=None):
    num_classes = num_classes or NUM_THREAT
    total, n = 0.0, 0
    for k in range(num_classes-1):
        mask = y > (k-1)
        if mask.sum() == 0: continue
        z = logits[mask, k]; t = (y[mask] > k).float()
        total = total + F.binary_cross_entropy_with_logits(z, t, reduction="sum"); n += int(mask.sum())
    return total / max(n, 1)

def cb_weights(df, col, n):
    cnt = np.bincount(df[col].to_numpy(), minlength=n).astype(np.float32) + 1.0
    w = 1.0/cnt; w = w * (n / w.sum())
    return torch.tensor(w, device=DEVICE)

# =====================================================================================
# 9. CORN DECODING + numeric helpers + metrics
# =====================================================================================
def corn_cum_probs(logits): return torch.cumprod(torch.sigmoid(logits), dim=1)
def corn_levels(logits):    return (corn_cum_probs(logits) > 0.5).sum(dim=1)
def corn_class_probs(logits):
    cum = corn_cum_probs(logits); N = logits.shape[0]
    p = torch.zeros(N, NUM_THREAT, device=logits.device)
    p[:,0] = 1 - cum[:,0]
    for k in range(1, NUM_THREAT-1): p[:,k] = cum[:,k-1] - cum[:,k]
    p[:,NUM_THREAT-1] = cum[:,-1]
    return p.clamp_min(1e-6)

def softmax_np(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def sigmoid_np(z): return 1.0/(1.0+np.exp(-z))
def qwk(a, b):
    try:    return cohen_kappa_score(a, b, weights="quadratic", labels=list(range(NUM_THREAT)))
    except Exception: return float("nan")

def expected_calibration_error(probs, labels, bins=15):
    conf = probs.max(1); pred = probs.argmax(1); acc = (pred == labels).astype(float)
    edges = np.linspace(0, 1, bins+1); ece = 0.0
    for b in range(bins):
        m = (conf > edges[b]) & (conf <= edges[b+1])
        if m.sum() > 0: ece += abs(acc[m].mean() - conf[m].mean()) * m.sum() / len(labels)
    return ece

def per_class_report(y_true, y_pred, labels):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(labels)), zero_division=0)
    return {labels[i]: {"precision":round(float(p[i]),4), "recall":round(float(r[i]),4),
                        "f1":round(float(f[i]),4), "support":int(s[i])} for i in range(len(labels))}

def confusion(y_true, y_pred, n): return confusion_matrix(y_true, y_pred, labels=range(n))

def aggregate_by_item(logits, items):
    uniq = list(dict.fromkeys(items.tolist() if hasattr(items, "tolist") else list(items)))
    idx = {it:i for i,it in enumerate(uniq)}
    agg = {k: np.zeros((len(uniq),) + v.shape[1:], dtype=np.float64) for k,v in logits.items()}
    cnt = np.zeros(len(uniq))
    for j, it in enumerate(items):
        i = idx[it]; cnt[i] += 1
        for k in logits: agg[k][i] += logits[k][j]
    for k in agg: agg[k] = agg[k] / (cnt[:,None] if agg[k].ndim == 2 else cnt)
    return {k: v.astype(np.float32) for k,v in agg.items()}, uniq, idx

def item_labels(ys, items, idx):
    lab = {k: np.zeros(len(idx), dtype=int) for k in ys}; seen = set()
    for j, it in enumerate(items):
        if it in seen: continue
        seen.add(it)
        for k in ys: lab[k][idx[it]] = int(ys[k][j])
    return lab

def item_metrics(logits, ys, items):
    agg, uniq, idx = aggregate_by_item(logits, items)
    lab = item_labels(ys, items, idx)
    mp = softmax_np(agg["material"]); mpred = mp.argmax(1)
    mat = {"acc": accuracy_score(lab["material"], mpred),
           "macro_f1": f1_score(lab["material"], mpred, average="macro", labels=range(len(MATERIALS)), zero_division=0),
           "ece": expected_calibration_error(mp, lab["material"])}
    cp = sigmoid_np(agg["condition"]); cpred = (cp > 0.5).astype(int)
    try:    auroc = roc_auc_score(lab["condition"], cp)
    except Exception: auroc = float("nan")
    cond = {"acc": accuracy_score(lab["condition"], cpred), "f1": f1_score(lab["condition"], cpred, zero_division=0),
            "auroc": auroc, "ece": expected_calibration_error(np.stack([1-cp, cp], 1), lab["condition"])}
    rp = softmax_np(agg["route"]); rpred = rp.argmax(1)
    rt = {"acc": accuracy_score(lab["route"], rpred),
          "macro_f1": f1_score(lab["route"], rpred, average="macro", labels=range(len(ROUTES)), zero_division=0),
          "ece": expected_calibration_error(rp, lab["route"])}
    tlog = torch.tensor(agg["threat"]); tpred = corn_levels(tlog).numpy(); tp = corn_class_probs(tlog).numpy()
    th = {"qwk": qwk(lab["threat"], tpred), "mae": float(np.abs(lab["threat"]-tpred).mean()),
          "acc": accuracy_score(lab["threat"], tpred), "ece": expected_calibration_error(tp, lab["threat"])}
    return {"material":mat, "condition":cond, "route":rt, "threat":th}

def full_report(logits, ys, items):
    """Rich diagnostics for a results notebook: scalar metrics + per-class reports + confusion matrices
    + threat per-level accuracy. Everything JSON/CSV-friendly so problems are easy to spot."""
    agg, uniq, idx = aggregate_by_item(logits, items)
    lab = item_labels(ys, items, idx)
    out = {"scalars": item_metrics(logits, ys, items), "per_class": {}, "confusion": {}}
    mpred = softmax_np(agg["material"]).argmax(1)
    rpred = softmax_np(agg["route"]).argmax(1)
    tpred = corn_levels(torch.tensor(agg["threat"])).numpy()
    cpred = (sigmoid_np(agg["condition"]) > 0.5).astype(int)
    out["per_class"]["material"]  = per_class_report(lab["material"], mpred, MATERIALS)
    out["per_class"]["route"]     = per_class_report(lab["route"], rpred, ROUTES)
    out["per_class"]["condition"] = per_class_report(lab["condition"], cpred, CONDITIONS)
    out["per_class"]["threat"]    = per_class_report(lab["threat"], tpred, [str(i) for i in range(NUM_THREAT)])
    out["confusion"]["material"]  = confusion(lab["material"], mpred, len(MATERIALS)).tolist()
    out["confusion"]["route"]     = confusion(lab["route"], rpred, len(ROUTES)).tolist()
    out["confusion"]["threat"]    = confusion(lab["threat"], tpred, NUM_THREAT).tolist()
    # threat per-level accuracy (where the ordinal head struggles)
    lvl = {}
    for L in range(NUM_THREAT):
        m = lab["threat"] == L
        lvl[L] = {"support": int(m.sum()), "acc": float((tpred[m]==L).mean()) if m.sum() else None}
    out["threat_per_level"] = lvl
    out["n_items"] = len(uniq)
    return out

def evaluate_logits(model, loader, tta=None):
    tta = USE_TTA if tta is None else tta
    model.eval()
    out = {k:[] for k in ["material","condition","route","threat"]}
    ys  = {k:[] for k in ["material","condition","route","threat"]}; items = []
    with torch.no_grad():
        for x, y, it in loader:
            x = x.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=DEVICE=="cuda"):
                o = model(x)
                if tta:                                   # average logits with a horizontal flip
                    o2 = model(torch.flip(x, dims=[3]))
                    o = {k: (o[k] + o2[k]) / 2 for k in o}
            for k in out: out[k].append(o[k].float().cpu())
            for k in ys:  ys[k].append(y[k])
            items += list(it)
    out = {k: torch.cat(v).numpy() for k,v in out.items()}
    ys  = {k: torch.cat(v).numpy() for k,v in ys.items()}
    return out, ys, np.array(items)

def combined_val_loss(logits, ys):
    """Validation loss with the SAME components as training (fixed LOSS_W, for a comparable train-vs-val curve)."""
    zr = torch.tensor(logits["route"]); zc = torch.tensor(logits["condition"])
    zm = torch.tensor(logits["material"]); zt = torch.tensor(logits["threat"])
    yr = torch.tensor(ys["route"]).long();  yc = torch.tensor(ys["condition"]).float()
    ym = torch.tensor(ys["material"]).long(); yt = torch.tensor(ys["threat"]).long()
    L = (LOSS_W["route"]*F.cross_entropy(zr, yr) + LOSS_W["condition"]*F.binary_cross_entropy_with_logits(zc, yc)
         + LOSS_W["material"]*focal_loss(zm, ym) + LOSS_W["threat"]*corn_loss(zt, yt))
    return float(L)

# =====================================================================================
# 10. TRAINING — generic loop used by both single-backbone and fusion
# =====================================================================================
def _fit(model, df_tr, df_va, df_te, tag="model"):
    set_seed()
    tr, va, te = make_loaders(df_tr, df_va, df_te if df_te is not None else df_va)
    head_p = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith("bbs.")
              and not n.startswith("backbone.")]
    bb_p   = [p for n,p in model.named_parameters() if p.requires_grad and (n.startswith("bbs.")
              or n.startswith("backbone."))]
    groups = [{"params":head_p, "lr":BASE_LR}]
    if bb_p: groups.append({"params":bb_p, "lr":BB_LR})
    # learnable per-task loss weights (Kendall & Gal 2018): s_i = log(sigma_i^2); no weight decay on them
    log_vars = torch.zeros(4, device=DEVICE, requires_grad=True) if USE_UNCERTAINTY_W else None
    if log_vars is not None: groups.append({"params":[log_vars], "lr":BASE_LR, "weight_decay":0.0})
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    steps = max(1, len(tr)); total = EPOCHS*steps
    def lr_scale(step):
        wu = WARMUP_EPOCHS*steps
        if step < wu: return step/max(1, wu)
        return 0.5*(1 + math.cos(math.pi*(step-wu)/max(1, total-wu)))
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE=="cuda")
    w_route   = cb_weights(df_tr, "y_route", len(ROUTES))
    alpha_mat = cb_weights(df_tr, "y_material", len(MATERIALS)) if USE_CB_FOCAL else None
    ema = None
    if USE_EMA:
        try:
            from timm.utils import ModelEmaV2
            ema = ModelEmaV2(model, decay=EMA_DECAY)
        except Exception as e:
            print("  EMA unavailable, continuing without:", e)
    hist, best, best_state, g, no_improve = [], -1e9, None, 0, 0
    for ep in range(EPOCHS):
        model.train(); run = 0.0
        for x, y, _ in tr:
            x = x.to(DEVICE)
            ym = y["material"].to(DEVICE); yc = y["condition"].to(DEVICE)
            yt = y["threat"].to(DEVICE);   yr = y["route"].to(DEVICE)
            s = lr_scale(g)
            opt.param_groups[0]["lr"] = BASE_LR*s
            if bb_p: opt.param_groups[1]["lr"] = BB_LR*s
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=DEVICE=="cuda"):
                o = model(x)
                losses = [F.cross_entropy(o["route"], yr, weight=w_route, label_smoothing=LABEL_SMOOTH),
                          F.binary_cross_entropy_with_logits(o["condition"], yc),
                          focal_loss(o["material"], ym, alpha=alpha_mat),
                          corn_loss(o["threat"], yt)]
                if log_vars is not None:
                    L = sum(torch.exp(-log_vars[i])*losses[i] + log_vars[i] for i in range(4))
                else:
                    L = (LOSS_W["route"]*losses[0] + LOSS_W["condition"]*losses[1]
                         + LOSS_W["material"]*losses[2] + LOSS_W["threat"]*losses[3])
            scaler.scale(L).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            if ema is not None:
                ema.decay = min(EMA_DECAY, (1 + g) / (10 + g))   # decay warmup -> early EMA tracks the model
                ema.update(model)
            g += 1; run += float(L.item())
        eval_model = ema.module if ema is not None else model
        vlog, vy, vit = evaluate_logits(eval_model, va); m = item_metrics(vlog, vy, vit)
        vloss = combined_val_loss(vlog, vy)
        score = SEL_W["route"]*m["route"]["macro_f1"] + SEL_W["threat"]*m["threat"]["qwk"] + SEL_W["material"]*m["material"]["macro_f1"]
        hist.append({"epoch":ep, "train_loss":run/steps, "val_loss":vloss,
                     "val_route_f1":m["route"]["macro_f1"], "val_threat_qwk":m["threat"]["qwk"],
                     "val_material_f1":m["material"]["macro_f1"], "val_condition_f1":m["condition"]["f1"],
                     "val_material_acc":m["material"]["acc"], "val_threat_mae":m["threat"]["mae"]})
        print(f"  [{tag}] ep{ep}: tr_loss={run/steps:.3f} val_loss={vloss:.3f}  routeF1={m['route']['macro_f1']:.3f}  "
              f"threatQWK={m['threat']['qwk']:.3f}  matF1={m['material']['macro_f1']:.3f}  "
              f"matAcc={m['material']['acc']:.3f}  condF1={m['condition']['f1']:.3f}")
        if score > best:
            best = score; best_state = {k:v.detach().cpu().clone() for k,v in eval_model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if EARLY_STOP_PATIENCE and no_improve >= EARLY_STOP_PATIENCE:
                print(f"  [{tag}] early stop at ep{ep} (no val gain for {EARLY_STOP_PATIENCE} epochs)")
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist)

def train_backbone(name, df_tr, df_va, df_te=None):
    print(f"\n=== Training single backbone: {name} ===")
    model = MultiTaskNet(name).to(DEVICE)
    return _fit(model, df_tr, df_va, df_te, tag=name)

def train_fusion(backbone_names=None, df_tr=None, df_va=None, df_te=None, freeze=None, weight_paths=None):
    names = backbone_names or FUSION_BACKBONES
    freeze = FUSION_FREEZE if freeze is None else freeze
    print(f"\n=== Training FUSION: {names} (freeze={freeze}, fine-tuned_backbones={bool(weight_paths)}) ===")
    model = MultiBackboneFusion(names, freeze=freeze, weight_paths=weight_paths).to(DEVICE)
    return _fit(model, df_tr, df_va, df_te, tag="fusion")

# =====================================================================================
# 10b. OPTIMIZED ViT FINE-TUNING (ADDITIVE — used by SC23-SC25)
# -------------------------------------------------------------------------------------
# Nothing here touches _fit / train_backbone / train_fusion, so every existing backbone and
# fusion result stays byte-identical. This is a ViT-specialised trainer that layers the standard
# small-data transformer tricks on top of the SAME 4 heads / focal+CORN losses / EMA / hflip-TTA /
# early-stop machinery, so its outputs (logits_<name>.npz, metrics_<name>.json) slot straight into
# SC6/SC17 next to the CNNs. Tricks (all citable):
#   * Layer-wise LR decay (LLRD)         — early blocks keep generic pretrained features, later blocks adapt.
#   * Linear-probe-then-finetune (LP-FT) — warm the heads with the backbone frozen, THEN unfreeze at a low
#                                          LR (Kumar et al. 2022: preserves pretrained features, better OOD).
#   * Optional permanent block freezing  — freeze patch-embed + first N blocks entirely (heavy overfit brake).
#   * Stochastic depth (drop_path)       — mild transformer-native regularisation.
#   * Optional higher input resolution   — resolve fine detail (hazard labels/text) the DINOv2 pos-embed can carry.
# =====================================================================================
class MultiTaskViT(nn.Module):
    """A pretrained ViT backbone (drop_path + optional higher img_size) -> the same 4 Heads as
    MultiTaskNet. Separate class so drop_path_rate/img_size can be set at creation without touching
    _make_backbone or MultiTaskNet (keeps the CNN path byte-identical)."""
    def __init__(self, name, drop_path=0.1, img_size=None):
        super().__init__()
        self._img = int(img_size or IMG_SIZE)
        self.backbone = None
        base = dict(pretrained=True, num_classes=0, drop_path_rate=drop_path)
        for extra in (dict(img_size=self._img, global_pool="avg"), dict(img_size=self._img),
                      dict(global_pool="avg"), dict()):
            try:
                self.backbone = timm.create_model(name, **base, **extra); break
            except (TypeError, RuntimeError):
                continue
        if self.backbone is None:                                  # last resort (surface a clean error)
            self.backbone = timm.create_model(name, pretrained=True, num_classes=0)
        was = self.backbone.training; self.backbone.eval()
        with torch.no_grad():
            d = int(self.backbone(torch.zeros(1, 3, self._img, self._img)).shape[-1])
        self.backbone.train(was)
        self.h_material  = Head(d, len(MATERIALS))
        self.h_condition = Head(d, 1)
        self.h_route     = Head(d, len(ROUTES))
        self.h_threat    = Head(d, NUM_THREAT-1)
    def forward(self, x):
        f = self.backbone(x)
        return {"material":  self.h_material(f), "condition": self.h_condition(f).squeeze(1),
                "route":     self.h_route(f),    "threat":    self.h_threat(f)}

def _vit_set_backbone_trainable(model, flag, freeze_blocks):
    """Enable/disable grads on the backbone. When enabling, the first `freeze_blocks` transformer
    blocks (and the patch-embed/pos-embed if freeze_blocks>0) stay frozen. Heads are never touched."""
    nb = len(model.backbone.blocks) if hasattr(model.backbone, "blocks") else 0
    for pname, p in model.named_parameters():
        if not pname.startswith("backbone."):
            continue
        sub = pname[len("backbone."):]
        if sub.startswith("blocks."):
            bidx = int(sub.split(".")[1])
            p.requires_grad = bool(flag) and (bidx >= freeze_blocks)
        else:                                                      # patch_embed / tokens / pos_embed / norms
            p.requires_grad = bool(flag) and (freeze_blocks == 0)
    return nb

def _vit_param_groups(model, base_lr, bb_lr, layer_decay, weight_decay):
    """Layer-wise-decayed AdamW groups. Heads @ base_lr; each backbone block gets
    bb_lr * layer_decay**(depth+1-layer) (deepest decay on the patch embed); no weight-decay on
    biases / 1-D norm params. Only requires_grad params are included (so frozen blocks are skipped)."""
    nb = len(model.backbone.blocks) if hasattr(model.backbone, "blocks") else 0
    def layer_of(pname):
        if not pname.startswith("backbone."):
            return nb + 1                                          # heads = top
        sub = pname[len("backbone."):]
        if sub.startswith("blocks."):
            return int(sub.split(".")[1]) + 1
        if sub.startswith(("norm.", "fc_norm.")):
            return nb                                              # final norm just below heads
        return 0                                                  # patch_embed / tokens / pos_embed / norm_pre
    groups = {}
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        L = layer_of(pname)
        lr = base_lr if L == nb + 1 else bb_lr * (layer_decay ** (nb + 1 - L))
        no_decay = (p.ndim == 1) or pname.endswith(".bias")
        key = (L, no_decay)
        groups.setdefault(key, {"params": [], "lr": lr,
                                "weight_decay": 0.0 if no_decay else weight_decay})["params"].append(p)
    return list(groups.values())

def _fit_vit(model, df_tr, df_va, df_te, tag, layer_decay, lp_epochs, freeze_blocks):
    """LP-FT training loop: mirrors _fit (same losses/EMA/TTA/early-stop/val-score) but with an
    optional linear-probe phase (backbone frozen, heads warm up) followed by an LLRD fine-tune phase."""
    set_seed()
    tr, va, te = make_loaders(df_tr, df_va, df_te if df_te is not None else df_va)
    w_route   = cb_weights(df_tr, "y_route", len(ROUTES))
    alpha_mat = cb_weights(df_tr, "y_material", len(MATERIALS)) if USE_CB_FOCAL else None
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE == "cuda")
    ema = None
    if USE_EMA:
        try:
            from timm.utils import ModelEmaV2
            ema = ModelEmaV2(model, decay=EMA_DECAY)
        except Exception as e:
            print("  EMA unavailable, continuing without:", e)

    steps = max(1, len(tr))
    lp_epochs = max(0, min(int(lp_epochs), EPOCHS - 1))            # keep at least 1 fine-tune epoch

    def build_opt(phase):
        if phase == "probe":                                      # heads only, backbone frozen
            _vit_set_backbone_trainable(model, False, freeze_blocks)
            hp = [p for n, p in model.named_parameters() if p.requires_grad]
            return torch.optim.AdamW([{"params": hp, "lr": BASE_LR}], weight_decay=WEIGHT_DECAY)
        _vit_set_backbone_trainable(model, True, freeze_blocks)   # unfreeze (minus frozen blocks)
        return torch.optim.AdamW(_vit_param_groups(model, BASE_LR, BB_LR, layer_decay, WEIGHT_DECAY),
                                 weight_decay=WEIGHT_DECAY)

    phase = "probe" if lp_epochs > 0 else "finetune"
    opt = build_opt(phase)
    base_lrs = [g["lr"] for g in opt.param_groups]
    def cos_scale(step, warm, total):
        if step < warm: return step / max(1, warm)
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    hist, best, best_state, no_improve = [], -1e9, None, 0
    for ep in range(EPOCHS):
        if lp_epochs > 0 and ep == lp_epochs:                     # LP -> FT switch: rebuild opt+schedule
            phase = "finetune"; opt = build_opt(phase); base_lrs = [g["lr"] for g in opt.param_groups]
        if phase == "probe":
            warm, total, local = max(1, steps), lp_epochs * steps, ep * steps
        else:
            ft_ep = EPOCHS - lp_epochs
            warm, total, local = max(1, steps), ft_ep * steps, (ep - lp_epochs) * steps
        model.train(); run = 0.0
        for x, y, _ in tr:
            x = x.to(DEVICE)
            ym = y["material"].to(DEVICE); yc = y["condition"].to(DEVICE)
            yt = y["threat"].to(DEVICE);   yr = y["route"].to(DEVICE)
            s = cos_scale(local, warm, total)
            for gi, g in enumerate(opt.param_groups): g["lr"] = base_lrs[gi] * s
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=DEVICE == "cuda"):
                o = model(x)
                L = (LOSS_W["route"]*F.cross_entropy(o["route"], yr, weight=w_route, label_smoothing=LABEL_SMOOTH)
                     + LOSS_W["condition"]*F.binary_cross_entropy_with_logits(o["condition"], yc)
                     + LOSS_W["material"]*focal_loss(o["material"], ym, alpha=alpha_mat)
                     + LOSS_W["threat"]*corn_loss(o["threat"], yt))
            scaler.scale(L).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            if ema is not None:
                ema.decay = min(EMA_DECAY, (1 + local) / (10 + local)); ema.update(model)
            local += 1; run += float(L.item())
        eval_model = ema.module if ema is not None else model
        vlog, vy, vit = evaluate_logits(eval_model, va); m = item_metrics(vlog, vy, vit)
        vloss = combined_val_loss(vlog, vy)
        score = SEL_W["route"]*m["route"]["macro_f1"] + SEL_W["threat"]*m["threat"]["qwk"] + SEL_W["material"]*m["material"]["macro_f1"]
        hist.append({"epoch": ep, "phase": phase, "train_loss": run/steps, "val_loss": vloss,
                     "val_route_f1": m["route"]["macro_f1"], "val_threat_qwk": m["threat"]["qwk"],
                     "val_material_f1": m["material"]["macro_f1"], "val_condition_f1": m["condition"]["f1"],
                     "val_material_acc": m["material"]["acc"], "val_threat_mae": m["threat"]["mae"]})
        print(f"  [{tag}] ep{ep}({phase[:2]}): tr_loss={run/steps:.3f} val_loss={vloss:.3f}  "
              f"routeF1={m['route']['macro_f1']:.3f}  threatQWK={m['threat']['qwk']:.3f}  "
              f"matF1={m['material']['macro_f1']:.3f}  matAcc={m['material']['acc']:.3f}  "
              f"condF1={m['condition']['f1']:.3f}")
        if score > best:
            best = score; best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if EARLY_STOP_PATIENCE and phase == "finetune" and no_improve >= EARLY_STOP_PATIENCE:
                print(f"  [{tag}] early stop at ep{ep} (no val gain for {EARLY_STOP_PATIENCE} epochs)")
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist)

def train_vit(name, df_tr, df_va, df_te=None, layer_decay=0.75, lp_epochs=3,
              freeze_blocks=0, drop_path=0.1, img_size=None):
    """Optimised ViT fine-tune (SC23-SC25). Returns (model, history) exactly like train_backbone, so
    the caller saves logits/metrics/weights identically. Knobs: layer_decay (LLRD factor, 0.6-0.85),
    lp_epochs (linear-probe warmup epochs; 0 disables LP-FT), freeze_blocks (permanently frozen early
    blocks), drop_path (stochastic depth), img_size (override IMG_SIZE for higher-res ViTs)."""
    print(f"\n=== Training ViT (optimised): {name} | LLRD={layer_decay} lp_epochs={lp_epochs} "
          f"freeze_blocks={freeze_blocks} drop_path={drop_path} img={img_size or IMG_SIZE} ===")
    model = MultiTaskViT(name, drop_path=drop_path, img_size=img_size).to(DEVICE)
    return _fit_vit(model, df_tr, df_va, df_te, tag=name,
                    layer_decay=layer_decay, lp_epochs=lp_epochs, freeze_blocks=freeze_blocks)

# =====================================================================================
# 11. LOGITS CACHE I/O
# =====================================================================================
_HEADS = ["material","condition","route","threat"]
def save_logits(path, val, test):
    (ov, yv, iv), (ot, yt, it) = val, test
    np.savez_compressed(path,
        **{f"val_log_{k}": ov[k] for k in _HEADS},
        **{f"val_y_{k}":   yv[k] for k in _HEADS}, val_items=np.array(iv).astype(str),
        **{f"test_log_{k}": ot[k] for k in _HEADS},
        **{f"test_y_{k}":   yt[k] for k in _HEADS}, test_items=np.array(it).astype(str))
    return path

def load_logits(path):
    z = np.load(path, allow_pickle=True)
    val  = ({k: z[f"val_log_{k}"]  for k in _HEADS}, {k: z[f"val_y_{k}"]  for k in _HEADS}, z["val_items"])
    test = ({k: z[f"test_log_{k}"] for k in _HEADS}, {k: z[f"test_y_{k}"] for k in _HEADS}, z["test_items"])
    return val, test

# =====================================================================================
# 12. CALIBRATION — per-head temperature scaling
# =====================================================================================
def fit_T_softmax(logits, labels, steps=300):
    z = torch.tensor(logits, dtype=torch.float32); y = torch.tensor(labels, dtype=torch.long)
    logT = torch.zeros(1, requires_grad=True); opt = torch.optim.Adam([logT], lr=0.05)
    for _ in range(steps): opt.zero_grad(); F.cross_entropy(z/logT.exp(), y).backward(); opt.step()
    return float(logT.exp().detach())

def fit_T_binary(logit, labels, steps=300):
    z = torch.tensor(logit, dtype=torch.float32); y = torch.tensor(labels, dtype=torch.float32)
    logT = torch.zeros(1, requires_grad=True); opt = torch.optim.Adam([logT], lr=0.05)
    for _ in range(steps): opt.zero_grad(); F.binary_cross_entropy_with_logits(z/logT.exp(), y).backward(); opt.step()
    return float(logT.exp().detach())

def fit_T_corn(logits, labels, steps=300):
    # Single scalar temperature shared across the CORN cut-points (matches fit_T_softmax /
    # fit_T_binary). A per-cut-point temperature vector has 3x the free parameters and
    # over-fits a small validation split, which regresses the already well-calibrated
    # threat head; the single scalar avoids that. Returns a Python float.
    z = torch.tensor(logits, dtype=torch.float32); y = torch.tensor(labels, dtype=torch.long)
    logT = torch.zeros(1, requires_grad=True); opt = torch.optim.Adam([logT], lr=0.05)
    for _ in range(steps): opt.zero_grad(); corn_loss(z/logT.exp(), y).backward(); opt.step()
    return float(logT.exp().detach())

# =====================================================================================
# 13. POLICY / OVERRIDE ENGINE + decision cost
# =====================================================================================
def apply_policy(material_probs, condition_probs, threat_probs, policy=None):
    policy = policy or POLICY
    routes, expl = [], []
    for i in range(len(material_probs)):
        mp = material_probs[i]; mat = MATERIALS[int(mp.argmax())]; mconf = float(mp.max())
        cond = "Poor" if condition_probs[i] > 0.5 else "Good"
        thr = int(threat_probs[i].argmax())
        e = []
        if mat in policy["material_force_special"]:
            routes.append("Special"); expl.append([f"material override: {mat}"]); continue
        if thr >= policy.get("threat_force_special_level", 99):
            routes.append("Special"); expl.append([f"threat backstop: predicted threat={thr} -> Special"]); continue
        if mat == "glass" and cond == "Poor":
            routes.append("Special"); expl.append(["glass + Poor -> broken glass; Special"]); continue
        if mat == "composite":
            top2 = mp.argsort()[-2:]
            if any(MATERIALS[c] in policy["material_force_special"] and mp[c] > 0.15 for c in top2):
                routes.append("Special"); expl.append(["composite hazardous sub-component -> Special"]); continue
        route = (policy["material_default_route_good"] if cond == "Good"
                 else policy["material_default_route_poor"])[mat]
        e.append(f"matrix[{mat},{cond}]={route}")
        if route == "Reuse" and (cond != "Good" or thr > policy["reuse_max_threat"]):
            route = "Recycle" if mat in policy["recyclables"] else "Landfill"
            e.append(f"reuse blocked (cond/threat) -> {route}")
        if mconf < policy["confidence_threshold_route"]:
            routes.append("Unsure"); expl.append(e + [f"low material confidence {mconf:.2f}"]); continue
        routes.append(route); expl.append(e)
    return np.array(routes, dtype=object), expl

COST = np.array([[0,1,1,20],[1,0,1,20],[2,2,0,15],[1,1,1,0]], dtype=float)
R2I = {r:i for i,r in enumerate(ROUTES)}

def decision_report(pred_routes, true_idx):
    cost, covered, hz_total, hz_caught = 0.0, 0, 0, 0
    for p, t in zip(pred_routes, true_idx):
        tr = ROUTES[int(t)]
        if tr == "Special": hz_total += 1
        if p == "Unsure":
            if tr == "Special": hz_caught += 1
            continue
        covered += 1; cost += COST[R2I[p], R2I[tr]]
        if tr == "Special" and p == "Special": hz_caught += 1
    return {"total_cost":cost, "coverage":covered/len(pred_routes),
            "hazard_recall":hz_caught/max(hz_total,1), "n":len(pred_routes)}

# =====================================================================================
# 13b. CONFIDENCE-AWARE RECONCILIATION LAYER  (cross-head consistency + interdependency)
#   Uses ALL FOUR calibrated heads. (1) material+condition imply an EXPECTED threat (prior);
#   reconcile with the threat head conservatively. (2) the learned ROUTE head gives a 2nd route
#   estimate; reconcile it with the policy-rule route. (3) calibrated CONFIDENCE gates abstention.
# =====================================================================================
SAFETY_ORDER = {"Reuse":0, "Recycle":1, "Landfill":2, "Special":3, "Unsure":-1}  # higher = more conservative

def _rule_route(mat, cond, thr, mp, policy):
    """Deterministic policy route (no confidence gate). Returns (route, reasons)."""
    if mat in policy["material_force_special"]: return "Special", [f"force_special:{mat}"]
    if thr >= policy.get("threat_force_special_level", 99): return "Special", [f"threat={thr} backstop"]
    if mat == "glass" and cond == "Poor": return "Special", ["glass+Poor->broken"]
    if mat == "composite":
        top2 = mp.argsort()[-2:]
        if any(MATERIALS[c] in policy["material_force_special"] and mp[c] > 0.15 for c in top2):
            return "Special", ["composite hazardous subpart"]
    route = (policy["material_default_route_good"] if cond == "Good"
             else policy["material_default_route_poor"])[mat]
    if route == "Reuse" and (cond != "Good" or thr > policy["reuse_max_threat"]):
        route = "Recycle" if mat in policy["recyclables"] else "Landfill"
    return route, [f"matrix[{mat},{cond}]={route}"]

def decide(material_probs, condition_probs, threat_probs, route_probs=None, policy=None,
           head_route_trust=0.70):
    """Confidence-aware decision. Feed CALIBRATED probs. Returns dict of per-item arrays:
       route (final), threat (reconciled), confidence, and flags/expl. route_probs = learned route head."""
    policy = policy or POLICY
    cth = policy["confidence_threshold_route"]
    R, Tf, Cf, FL, EX = [], [], [], [], []
    for i in range(len(material_probs)):
        mp = material_probs[i]; mat = MATERIALS[int(mp.argmax())]; mconf = float(mp.max())
        cprob = float(condition_probs[i]); cond = "Poor" if cprob > 0.5 else "Good"; cconf = max(cprob, 1-cprob)
        tp = threat_probs[i]; t_head = int(np.argmax(tp)); tconf = float(np.max(tp))
        # (1) expected threat from material+condition, then conservative reconciliation with the head
        t_prior = policy["material_default_threat"][mat]
        if mat == "glass" and cond == "Poor": t_prior = max(t_prior, 3)
        t_final = max(t_prior, t_head)                       # never under-call threat (safety)
        threat_conflict = (t_head != t_prior)
        # (2) policy-rule route (using the reconciled threat) vs the learned route head
        r_pol, why = _rule_route(mat, cond, t_final, mp, policy)
        if route_probs is not None:
            rp = route_probs[i]; r_head = ROUTES[int(np.argmax(rp))]; rconf = float(np.max(rp))
        else:
            r_head, rconf = r_pol, 1.0
        route_conflict = (route_probs is not None and r_head != r_pol)
        # (3) reconcile routes: safety first, then agreement, then confidence
        if r_pol == "Special" or r_head == "Special":
            route = "Special"; why.append("safety: Special on either head/policy")
        elif r_pol == r_head:
            route = r_pol; why.append("policy==head")
        else:
            route = r_pol if (SAFETY_ORDER[r_pol] >= SAFETY_ORDER[r_head] or rconf < head_route_trust) else r_head
            why.append(f"conflict pol={r_pol}/head={r_head} (rconf={rconf:.2f})->{route}")
        # (4) abstention — only when the heads genuinely DISAGREE *and* confidence is low. Do NOT abstain a
        #     correct, agreed-on benign route just because a 10-class softmax sits below a fixed threshold
        #     (that caused spurious "Unsure" on confidently-routed items). Hazards (Special) are never abstained.
        if route != "Special" and (route_conflict or threat_conflict) and mconf < cth:
            route = "Unsure"; why.append(f"low confidence + head conflict (mconf={mconf:.2f})")
        R.append(route); Tf.append(t_final); Cf.append(round(float(min(mconf, tconf, rconf)), 3))
        FL.append({"threat_conflict":threat_conflict, "route_conflict":route_conflict})
        EX.append(why)
    return {"route":np.array(R, dtype=object), "threat":np.array(Tf), "confidence":np.array(Cf),
            "flags":FL, "expl":EX}

def decision_report_v2(dec, true_route_idx, true_threat=None):
    """decision_report + cross-head consistency stats + confidence-stratified accuracy."""
    rep = decision_report(dec["route"], true_route_idx)
    rep["route_conflicts"]  = int(sum(f["route_conflict"]  for f in dec["flags"]))
    rep["threat_conflicts"] = int(sum(f["threat_conflict"] for f in dec["flags"]))
    rep["mean_confidence"]  = round(float(np.mean(dec["confidence"])), 3)
    if true_threat is not None:
        rep["threat_recon_mae"] = round(float(np.abs(dec["threat"] - np.asarray(true_threat)).mean()), 3)
    # accuracy on high- vs low-confidence decisions (calibration should make high-conf more correct)
    conf = dec["confidence"]; routes = dec["route"]; med = float(np.median(conf))
    for band, mask in [("high_conf", conf >= med), ("low_conf", conf < med)]:
        idx = np.where(mask)[0]; cov = [j for j in idx if routes[j] != "Unsure"]
        acc = np.mean([ROUTES[int(true_route_idx[j])] == routes[j] for j in cov]) if cov else float("nan")
        rep[f"{band}_route_acc"] = round(float(acc), 3)
    return rep

# =====================================================================================
# 14. ENSEMBLE over cached logits (cheap "combine backbones" — module 6/7)
# =====================================================================================
def _align(items_ref, items_other, arr):
    """reorder `arr` (rows indexed by items_other order) to match items_ref order, by item id."""
    pos = {it:i for i,it in enumerate(list(items_other))}
    return arr[[pos[it] for it in items_ref]]

def ensemble_logits(named_logits, items_ref, mode="mean", per_head_backbone=None, weights=None):
    """named_logits: dict name -> (logit_dict, item_array). Returns one fused logit_dict aligned to items_ref.
       mode='mean' averages all; per_head_backbone={head:name} routes a head to one model's logits."""
    names = list(named_logits)
    out = {}
    for h in _HEADS:
        if per_head_backbone and h in per_head_backbone:
            nm = per_head_backbone[h]; lg, it = named_logits[nm]
            out[h] = _align(items_ref, it, lg[h]); continue
        stacks = []
        for nm in names:
            lg, it = named_logits[nm]; stacks.append(_align(items_ref, it, lg[h]))
        if weights is not None:
            w = np.array([weights[nm] for nm in names], dtype=np.float64); w = w/w.sum()
            out[h] = np.tensordot(w, np.stack(stacks), axes=(0,0)).astype(np.float32)
        else:
            out[h] = np.mean(stacks, axis=0).astype(np.float32)
    return out

# =====================================================================================
# 15. DETECTION — export YOLO dataset from the manifests; end-to-end glue (module 8)
# =====================================================================================
def export_yolo_dataset(image_root, out_dir, class_name="waste_item", include_aug=False):
    """Write a single-class YOLO detection dataset from train/val/test.csv.
    bbox is stored TOP-LEFT (x,y,w,h) normalized [0,1] -> converted to YOLO center (cx,cy,w,h).
    Returns the path to data.yaml."""
    splits = {"train":"train.csv", "val":"validation.csv", "test":"test.csv"}
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for split, csvf in splits.items():
        df = pd.read_csv(os.path.join(image_root, csvf))
        if (not include_aug) and "source" in df.columns:
            df = df[df["source"].fillna("original") == "original"]
        bidx = _basename_index(image_root, split)
        imdir = os.path.join(out_dir, "images", split); ladir = os.path.join(out_dir, "labels", split)
        os.makedirs(imdir, exist_ok=True); os.makedirs(ladir, exist_ok=True)
        n = 0
        for _, r in df.iterrows():
            ap = _resolve(image_root, split, r["item_id"], r["image_path"], bidx)
            if not ap or not os.path.exists(ap): continue
            stem = f"{r['item_id']}__{os.path.splitext(os.path.basename(ap))[0]}"
            shutil.copyfile(ap, os.path.join(imdir, stem + os.path.splitext(ap)[1]))
            try:
                x, y, w, h = [float(r[k]) for k in ["bbox_x","bbox_y","bbox_w","bbox_h"]]
            except Exception:
                continue
            cx, cy = x + w/2, y + h/2
            cx, cy, w, h = [min(max(v, 0.0), 1.0) for v in (cx, cy, w, h)]
            with open(os.path.join(ladir, stem + ".txt"), "w") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            n += 1
        counts[split] = n
    yaml = (f"path: {os.path.abspath(out_dir)}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            f"nc: 1\nnames: ['{class_name}']\n")
    with open(os.path.join(out_dir, "data.yaml"), "w") as f: f.write(yaml)
    print("YOLO dataset written:", counts, "->", out_dir)
    return os.path.join(out_dir, "data.yaml")

def crop_from_pred(img, box_xyxy_px, pad=0.2):
    """Crop using a detector's absolute-pixel xyxy box (for end-to-end detect->classify)."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box_xyxy_px
    bx, by, bw, bh = x1/W, y1/H, (x2-x1)/W, (y2-y1)/H
    return object_crop(img, (bx, by, bw, bh), pad=pad)

print("thesis_common (10-class) loaded | device:", DEVICE, "| albumentations:", USE_ALB,
      "| materials:", len(MATERIALS))
