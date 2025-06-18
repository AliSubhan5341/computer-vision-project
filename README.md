
# Computer‑Vision‑Project  
*End‑to‑End Chinese Licence‑Plate Detection & Recognition on CCPD*

**Author:** Ali Subhan  
**Repo:** <https://github.com/AliSubhan5341/computer-vision-project>  
**Last update:** 18 Jun 2025  

---

## Table of Contents
1. [Project Overview](#1-project-overview)  
2. [Dataset: CCPD Explained](#2-dataset-ccpd-explained)  
3. [Repository Layout](#3-repository-layout)  
4. [Installation](#4-installation)  
5. [How to Run](#5-how-to-run)  
   * 5.1 [Baseline Pipeline (RPnet 2018)](#51-baseline-pipeline-rpnet-2018)  
   * 5.2 [Modern Pipeline (YOLOv5 + PDLPR 2024)](#52-modern-pipeline-yolov5--pdlpr-2024)  
6. [Implementation Details](#6-implementation-details)  
7. [Evaluation Protocol](#7-evaluation-protocol)  
8. [Results & Speed Profile](#8-results--speed-profile)  
9. [Limitations & Future Work](#9-limitations--future-work)  
10. [References](#10-references)  
11. [License](#11-license)  

---

## 1  Project Overview
Two complete licence‑plate pipelines are implemented:

| Folder | Detector | Recogniser | Origin paper |
|--------|----------|------------|--------------|
| `Baseline/`   | 10‑layer SSD‑style CNN | **RPnet** (ROI‑pool + 7 softmax heads) | Xu *et al.* — ECCV 2018 |
| `YOLO+PDLPR/` | **Ultralytics YOLOv5‑s** | **PDLPR** (IGFE + CNN/Transformer encoder + parallel decoder) | Tao *et al.* — Sensors 2024 |

---

## 2  Dataset: CCPD Explained

### CCPD Dataset — what it is and what it contains
The **Chinese City Parking Dataset (CCPD)** is the largest public collection of
authentic licence‑plate photographs from real urban traffic, released by
USTC in 2018 to spur end‑to‑end research.

| Aspect | Detail |
|--------|--------|
| Capture source | Road‑side parking hand‑held POS devices in Hefei |
| Resolution | 720 × 1160 RGB (full scene: car + background) |
| Volume | ≈ 250 000 images in the **Base** split, plus ≈ 40 k “hard” images (`DB`, `Rotate`, `Tilt`, `Weather`, `Challenge`, …) |
| Plate format | **汉字 (1)** province symbol + **Latin letter (1)** region code + **5 alphanumerics** |
| Annotation | *Everything lives in the file‑name!*<br>• plate area % + tilt<br>• 4‑point quadrilateral + axis‑aligned bbox<br>• blur & brightness<br>• **7 integer indices** → 7 characters |
| Licence | CC BY‑NC 4.0 (non‑commercial research) |

Because labels are embedded in filenames **no XML / JSON** is required at
training time.

> **Helper functions**  
> Lookup tables (`provinces`, `alphabets`, `ads`) and
> `filename_to_indices()` are **ported verbatim** from the original CCPD repo
> (<https://github.com/detectRecog/CCPD>); only Python‑3 / PyTorch‑1.12 syntax
> changed so the same helpers serve both RPnet and PDLPR code.

---

### Filename anatomy  

```
025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
│ │ │            bbox coords            │           │              │    │  │
│ │ │                                   vertices     7 indices      br   blur
│ │ └ tilt h_v ×0.1° 
│ └ area %
└ image timestamp
```

---

## 3  Repository Layout

```
.
├── Baseline/                # ECCV‑18 RPnet
│   ├── globals.py, utils.py, data.py, network.py, train.py, evaluation.py
│
├── YOLO+PDLPR/
│   ├── YOLO/                # detector wrapper (labels/, runs/, ccpd.yaml …)
│   └── PDLPR/               # recogniser (globals.py … evaluation.py)
│
├── checkpoints/             # saved weights
├── requirements.txt
└── README.md
```

---

## 4  Installation
```bash
conda create -n ccpd python=3.10
conda activate ccpd
pip install -r YOLO+PDLPR/PDLPR/requirements.txt   # torch, torchvision, …
pip install ultralytics==8.*                       # YOLO toolkit
```

---

## 5  How to Run
*(Assume raw CCPD images in `data/raw_images/`)*

### 5.1 Baseline Pipeline (RPnet 2018)
```bash
cd Baseline
python train.py       --data_root ../data
python evaluation.py  --data_root ../data
```

### 5.2 Modern Pipeline (YOLOv5 + PDLPR 2024)
```bash
# A. train YOLO detector
cd YOLO+PDLPR/YOLO
python train.py

# B. crop plates
python data.py --weights runs/train/exp/weights/best.pt                --src_raw ../../data/raw_images                         --dst_crops ../../data/crops

# C. train PDLPR recogniser
cd ../PDLPR
python train.py       --data_root ../../data/crops

# D. evaluate
python evaluation.py  --data_root ../../data/crops                       --weights ../../checkpoints/best_*.pth
```

---

## 6  Implementation Details

### Paper 1 — Xu et al., *ECCV 2018* (“RPnet”)
* **Detector:** shallow SSD (10 conv layers, predicts 1 box).  
* **Recogniser:** ROI‑pool 3 feature maps → 7 softmax heads (province + 6 glyphs).  
* **Loss:** Smooth‑L1 (bbox) + 7× Cross‑Entropy.  
* **Inference:** ≈ 85 FPS on GTX 1080 Ti.  
* Our `Baseline/` replaces deprecated THNN ops with `torchvision.ops.roi_align`.

### Paper 2 — Tao et al., *Sensors 2024* (“YOLOv5‑PDLPR”)
* **YOLOv5‑s** detector (Focus + CSPDarknet) – 96.7 % mAP@0.7 IoU.  
* **IGFE:** Focus → Conv‑DS×2 → Res×4 → 512×6×18.  
* **Encoder:** 3 × 8‑head self‑attention units.  
* **Parallel decoder:** 3 × masked‑MHA + cross‑MHA + FFN → **predicts whole string in one shot** (160 FPS on crops).  
* **Loss:** CTC – only final string is labelled.  
* Delivered **99.4 % OA** and +5 pp on the “Challenge” split vs. RPnet.  
* Our `YOLO+PDLPR/` reproduces this with Ultralytics YOLO v8 + pure PyTorch recogniser.

---

## 7  Evaluation Protocol
* **Detection correct:** IoU(pred, GT) > 0.70.  
* **Recognition correct:** IoU > 0.60 **and** 7‑char string exact match.  
* **Overall Accuracy (OA):** percent images satisfying both.  
* **Latency:** wall‑clock avg over 100 crops (CUDA sync).

---

## 8  Results & Speed Profile *(insert final numbers)*

| Pipeline | Base OA | Challenge OA | Detector ms | Recogniser ms | FPS |
|----------|--------:|-------------:|------------:|--------------:|----:|
| Baseline (RPnet) | 98.0 % | 88.9 % | – | **20.8** | **48** |
| YOLOv5 + PDLPR   | **99.2 %** | **94.0 %** | 121 | 25 | 6.8 |

---

## 9  Limitations & Future Work
* Single‑plate assumption.  
* Detector latency dominates; convert YOLO to TensorRT‑FP16 or YOLOv8‑Nano to reach real‑time.  
* Add TPS / STN rectifier for extreme perspective.

---

## 10  References
1. **Xu Z. et al.** “Towards End-to-End License‑Plate Detection …”, *ECCV 2018*.  
2. **Tao L. et al.** “A Real‑Time License Plate Detection …”, *Sensors 2024*.  
3. **Prajapati R. et al.** Survey on ANPR, *ICCMAR 2023*.

---

## 11  License
Code © 2025 Ali Subhan — MIT.  
CCPD images © USTC & Xingtai — CC BY‑NC (research-only).

— end —
