
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
This repo benchmarks two complete licence‑plate pipelines on the **Chinese City
Parking Dataset (CCPD)**:

| Folder | Detector | Recogniser | Origin paper |
|--------|----------|------------|--------------|
| `Baseline/`   | 10‑layer SSD‑style CNN | **RPnet** (ROI‑pool + 7 softmax heads) | Xu *et al.* — ECCV 2018 |
| `YOLO+PDLPR/` | **Ultralytics YOLOv5‑s** | **PDLPR** (IGFE + CNN/Transformer encoder + parallel decoder) | Tao *et al.* — Sensors 2024 |

All code is modernised to **PyTorch ≥ 1.12**; deprecated C++/THNN ops from the
original CCPD repo were removed or rewritten.

---

## 2  Dataset: CCPD Explained

| Property | Value |
|----------|-------|
| Images | 720 × 1160 RGB, ≈ 250 k (*Base* split) |
| Plate pattern | 汉字 (1) + letter (1) + alphanum (5) |
| Annotation | Everything in the file‑name: area %, tilt, 4‑vertex box, blur, brightness, **7 integer indices → glyphs** |
| Hard splits | `DB`, `Rotate`, `Tilt`, `Challenge`, … |
| Licence | CC BY‑NC 4.0 (non‑commercial research) |

```
025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
│ │ │            bbox coords            │           │              │    │  │
│ │ │                                   vertices     7 idx          br   blur
│ │ └ tilt h_v ×0.1°
│ └ area %
└ sample id
```

### Index‑to‑glyph lookup  
`utils.py` holds three tables (`PROVINCES`, `ALPHABETS`, `ADS`);  
`filename_to_indices()` converts the 7 indices to characters (皖AY339S, …).

> **Note – helper code origin**  
> These lookup tables and the splitting routine are **ported verbatim** from the
> original CCPD GitHub repo (<https://github.com/detectRecog/CCPD>).  
> Only Python‑3 syntax was updated.

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
├── checkpoints/             # saved weights (*.pth)
├── requirements.txt
└── README.md
```

Each sub‑folder follows the requested **Globals → Utils → Data → Network →
Train → Evaluation** scheme.

---

## 4  Installation

```bash
git clone https://github.com/AliSubhan5341/computer-vision-project.git
cd computer-vision-project

conda create -n ccpd python=3.10
conda activate ccpd

# dependencies for recognisers + baseline
pip install -r YOLO+PDLPR/PDLPR/requirements.txt

# Ultralytics YOLO (detector)
pip install ultralytics==8.*

# (optional) optimisation toolkits
pip install onnxruntime-gpu tensorrt
```

---

## 5  How to Run

### Data prep
1. Download `ccpd_base.zip` → `data/raw_images/`.  
2. *(Optional)* put `Rotate`, `Tilt`, etc. into `data/`.

### 5.1 Baseline Pipeline (RPnet 2018)

```bash
cd Baseline
python train.py       --data_root ../data
python evaluation.py  --data_root ../data
```

### 5.2 Modern Pipeline (YOLOv5 + PDLPR 2024)

```bash
# A. train YOLO detector
cd YOLO+PDLPR/YOLO
python train.py                           # wrapper around ultralytics
# best weights → runs/train/exp/weights/best.pt

# B. crop plates for recogniser
python data.py --weights runs/train/exp/weights/best.pt                --src_raw ../../data/raw_images                         --dst_crops ../../data/crops

# C. train PDLPR recogniser
cd ../PDLPR
python train.py       --data_root ../../data/crops

# D. evaluate end‑to‑end
python evaluation.py  --data_root ../../data/crops                       --weights ../../checkpoints/best_*.pth
```

---

## 6  Implementation Details

### Baseline (ECCV‑18 RPnet)
* 10‑layer CNN detector, ROI‑align on mid‑features, 7 softmax heads.
* Loss = Smooth‑L1 (bbox) + 7× Cross‑Entropy.
* Updated from THNN to torchvision ROI‑align; fully FP32/AMP‑safe.

### YOLOv5 + PDLPR (Sensors 2024)
| Stage | Blocks |
|-------|--------|
| Detector | YOLOv5‑s (Focus, CSPDarknet, PAN, GIoU) |
| Feature extractor | **IGFE** = Focus → Conv‑DS×2 → Res×4 (512 × 6 × 18) |
| Encoder | 3 × (1×1 conv↑ → 8‑head MHA → 1×1 conv↓ + LN) |
| Decoder | 3 × (masked MHA → cross‑MHA → FFN) — **parallel**, not autoregressive |
| Loss | CTC on 7‑token sequence |

---

## 7  Evaluation Protocol

* **Detection correct** ⇔ IoU(pred, GT) > 0.70.  
* **Recognition correct** ⇔ IoU > 0.60 **and** 7‑char string exact match.  
* **Overall Accuracy (OA)** = % images passing both.  
* **Latency** = mean wall‑clock over 100 crops (CUDA‑sync).

---

## 8  Results & Speed Profile  
*(insert your final numbers here)*

| Pipeline | Base OA | Challenge OA | Detector ms | Recogniser ms | FPS |
|----------|--------:|-------------:|------------:|--------------:|----:|
| Baseline (RPnet) | 98.0 % | 88.9 % | – | **20.8** | **48** |
| YOLOv5 + PDLPR   | **99.2 %** | **94.0 %** | 121 | 25 | 6.8 |

---

## 9  Limitations & Future Work
* Single-plate images; multi-plate parsing not supported.  
* Detector dominates latency; TensorRT or YOLOv8‑Nano could push to real‑time.  
* Extreme perspective could benefit from TPS or STN rectification.

---

## 10  References

| Ref | Summary |
|-----|---------|
| **Xu et al., 2018 — ECCV** | Released CCPD and **RPnet**, a 10‑layer detector + 7‑head recogniser. |
| **Tao et al., 2024 — Sensors** | Introduced **PDLPR** (IGFE + parallel Transformer decoder) and paired it with YOLOv5 for real‑time performance (≈160 FPS on crops). |
| **Prajapati et al., 2023** | Survey of machine‑learning approaches to ANPR. |

---

## 11  License

Source code © 2025 Ali Subhan — MIT Licence.  
CCPD images © USTC & Xingtai — CC BY‑NC (research‑only).

---

*Happy training 🚗💨*
