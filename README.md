
# Computer‑Vision‑Project  
*Licence‑plate detection & recognition on CCPD  
(Baseline re‑implementation ✦ YOLOv5‑PDLPR 2024)*

**Author:** Ali Subhan  
**Repo:** <https://github.com/AliSubhan5341/computer-vision-project>  
**Last update:** 18 Jun 2025  

---

## Table of Contents
1. [Project overview](#1-project-overview)  
2. [Dataset: CCPD in a nutshell](#2-dataset-ccpd-in-a-nutshell)  
3. [Repository layout](#3-repository-layout)  
4. [Implementation details](#4-implementation-details)  
   * 4.1 [Baseline (ECCV‑18 RPnet)](#41-baseline-eccv18-rpnet)  
   * 4.2 [YOLOv5 + PDLPR](#42-yolov5--pdlpr)  
   * 4.3 [Modernising the original CCPD helpers](#43-modernising-the-original-ccpd-helpers)  
5. [Installation](#5-installation)  
6. [How to run](#6-how-to-run)  
   * 6.1 [Baseline pipeline](#61-baseline-pipeline)  
   * 6.2 [YOLO + PDLPR pipeline](#62-yolo--pdlpr-pipeline)  
7. [Evaluation protocol](#7-evaluation-protocol)  
8. [Results & speed profile](#8-results--speed-profile)  
9. [Limitations & future work](#9-limitations--future-work)  
10. [References](#10-references)  
11. [License](#11-license)

---

## 1  Project overview
This repo re‑visits the **Chinese City Parking Dataset (CCPD)** and delivers **two fully working plate‑reading systems**:

| Folder | Detector | Recogniser | Target paper |
|--------|----------|------------|--------------|
| `Baseline/`   | 10‑layer SSD‑style CNN  | RPnet (ROI‑pool + 7 softmax) | Xu *et al.*, *ECCV 2018* |
| `YOLO_PDLPR/` | Ultralytics YOLOv5‑s    | PDLPR (IGFE + CNN/Transformer encoder + *parallel* decoder) | Tao *et al.*, *Sensors 2024* |

Both implementations are written in **modern PyTorch ≥ 1.12**, with
deprecated APIs from the original 2018 codebase upgraded or replaced.

> **Why two versions?**  
> * Re‑create the ECCV‑18 baseline to quantify improvements honestly.  
> * Provide a fresh, production‑ready reference for the 2024 YOLO‑PDLPR method.

---

## 2  Dataset: CCPD in a nutshell

| Prop.          | Value |
|----------------|-------|
| Images         | 720 × 1160 RGB — ≈ 250 k in `ccpd_base` |
| Plate pattern  | 汉字 (1) + letter (1) + alphanum (5) |
| Annotation     | 4‑point plate quadrilateral, IoU tilt, brightness, blur, **7‑char indices inside file‑name** |
| Hard subsets   | `DB`, `Rotate`, `Tilt`, `Challenge`, … (eval only) |
| Licence        | CC BY‑NC (non‑commercial research) |

Filename anatomy:

```
025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
│ │ │            bbox coords            │           │              │    │  │
│ │ │                                   vertices     7 indices      br   blur
│ │ └ tilt h_v ×0.1° 
│ └ area %
└ sample id
```

During preprocessing we **keep the file‑name** after YOLO cropping, so the same
index‑parsing helpers work across both pipelines.

---

## 3  Repository layout

```
computer-vision-project/
├── Baseline/                # ECCV‑18 compatible
│   ├── globals.py           # tables & hyper‑params
│   ├── utils.py             # filename↔glyph helpers, masks
│   ├── data.py              # CCPDPlateCrops Dataset
│   ├── network.py           # RPnet CNN
│   ├── train.py
│   └── evaluation.py
│
├── YOLO_PDLPR/              # Sensors‑24 implementation
│   ├── detector/            # Ultralytics YOLOv5 wrapper & crop exporter
│   ├── globals.py
│   ├── utils.py
│   ├── data.py
│   ├── network.py           # IGFE + Transformer encoder/decoder
│   ├── train.py
│   └── evaluation.py
│
├── checkpoints/             # *.pth saved here
├── requirements.txt
└── README.md
```

Every folder respects the **Globals → Utils → Data → Network → Train → Evaluation** rubric.

---

## 4  Implementation details

### 4.1 Baseline (ECCV‑18 RPnet)
* **Backbone** – 10 tiny Conv‑BN‑ReLU blocks (stride 2 every other layer).  
* **ROI pooling** – replaced the THNN C++ op with `torchvision.ops.roi_align`.  
* **Recogniser** – 3 mid‑level feature maps ROI‑pooled → concat → `7 × FC` heads.  
* **Modernisation** – removed all `torch.autograd.Variable`, updated `torch.nn.DataParallel` calls, added AMP option.  
* **Loss** – Smooth‑L1 on (cx,cy,w,h) + 7 × CE, as in the paper.

### 4.2 YOLOv5 + PDLPR
* **Detector** – vanilla YOLOv5‑s (Focus, CSPDarknet, PAN) trained with Ultralytics ≥ 8.  
* **IGFE** – Focus slice → ConvDownSampling(2×) → ResBlock(4×) → 512 × 6 × 18 tensor.  
* **Encoder** – 3 residual units: `1×1 conv ↑` → 8‑head MHA → `1×1 conv ↓` + LayerNorm.  
* **Parallel decoder** – 3 units: masked‑MHA (causal) → cross‑MHA → FFN, all computed **in one pass** (no RNN unrolling).  
* **Helper porting** – vertex/tlt/blur parsers, glyph lookup tables are **identical to the original CCPD repo** (`detectRecog/CCPD`); only syntax updated.

### 4.3 Modernising the original CCPD helpers
The original repo (2018) relied on:
* `torch.utils.ffi`, `THNN` C extensions  
* Python 2 style print / path handling  
* Obsolete ROI pooling code

We carried over:
* **lookup tables** (`annos.py`)  
* **filename parsing logic** (`utils.py`)  

…and rewrote them to pure Python 3.10 / PyTorch 1.12.

---

## 5  Installation

```bash
git clone https://github.com/AliSubhan5341/computer-vision-project.git
cd computer-vision-project
conda create -n ccpd python=3.10
conda activate ccpd
pip install -r requirements.txt
```

---

## 6  How to run

### 6.1 Baseline pipeline

```bash
cd Baseline
python train.py       --data_root ../data          # train RPnet
python evaluation.py  --data_root ../data          # acc + FPS
```

### 6.2 YOLO + PDLPR pipeline

```bash
# A. train detector
cd YOLO_PDLPR/detector
yolo train model=yolov5s.pt data=ccpd.yaml epochs=100 imgsz=640

# B. export plate crops
python export_crops.py --weights runs/train/exp/weights/best.pt \
                       --src_raw ../../../data/raw_ccpd_images    \
                       --dst_crops ../../data

# C. train recogniser
cd ..
python train.py       --data_root ../../data
python evaluation.py  --data_root ../../data
```

---

## 7  Evaluation protocol

* **Detection success**: IoU(pred, GT) > 0.70  
* **Recognition success**: IoU > 0.60 **and** 7‑char string match  
* **Overall Accuracy (OA)**: % images passing both above rules  
* **Latency**: averaged over 100 crops (wall‑time / CUDA‑sync)

These rules match **exactly** the ECCV‑18 and Sensors‑24 papers, enabling direct comparison.

---

## 8  Results & speed profile *(example numbers)*

| Pipeline | Base OA | Challenge OA | Detector ms | Recogniser ms | Pipeline FPS |
|----------|--------:|-------------:|------------:|--------------:|-------------:|
| Baseline (RPnet) | 98.0 % | 88.9 % | –   | **20.8** | **48.0** |
| **YOLO+PDLPR** | **99.2 %** | **94.0 %** | 121.2 | 25.3 | 6.8 |

*Recognition is ~4.8× faster than the detector — detector is the bottleneck.*

---

## 9  Limitations & future work
* Supports **single** Chinese plate per image; no multicam fusion.  
* Detector latency dominates; converting YOLO to TensorRT‑FP16 or moving to YOLOv8‑n can push to real‑time.  
* Perspective warping is implicit; adding STN‑style rectification could improve extreme tilts.

---

## 10  References
1. L. Tao *et al.*, “A Real‑Time License Plate Detection …”, **Sensors 24(9)**, 2791 (2024)  
2. Z. Xu *et al.*, “Towards End‑to‑End License‑Plate Detection …”, **ECCV** (2018)  
3. R. Prajapati *et al.*, “A Review Paper on ANPR …”, **ICCMAR** (2023)

---

## 11  License
*Source code*: MIT  
*CCPD data*: © USTC & Xingtai — research‑only, non‑commercial redistribution.

---

*Happy coding & safe driving 🚗💨*
