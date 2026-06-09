# Comprehensive Road Scene Understanding for Autonomous Driving

**FAIMDL 2025/26 — Image Segmentation Project**
Politecnico di Torino

> **Report**: 3_IS_s353753_s362637_s353385_s353978_Colella_Giordana_Ferrara_Cellaura
> **Experiment tracking (W&B)**: [https://wandb.ai/danieleferrara-politecnico-di-torino/eomt?nw=nwuserdanieleferrara]

---

## Repository Structure

```

MaskArchitectureAnomaly_CourseProject/
│
├── eomt/                              # EoMT source code (original repo)
│   ├── finetuning_3_exp.ipynb         # Step 5: Fine-tuning experiments
│   └── inference_eomt_coco_ft.ipynb   # Step 4: Zero-shot inference + mapping + fine-tuned models inference
│
├── eval/
│   ├── erfnet.py                      # ERFNet model definition
│   ├── evalAnomaly.ipynb              # Step 7: ERFNet anomaly baselines
│   └── evalAnomaly_eomt.ipynb         # Step 8: EoMT anomaly evaluation
│
└── trained_models/                    # Pre-trained checkpoints
    ├── eomt_cityscapes.bin
    └── erfnet_pretrained.pth
```

---

## Setup

All notebooks run on **Google Colab (T4 GPU)**. Mount Google Drive and update `PROJECT_ROOT` at the top of each notebook.

### Required checkpoints (place in `trained_models/`)

| File | Source |
|---|---|
| `eomt_coco.bin` | [EoMT official release](https://github.com/tue-mps/eomt) |
| `eomt_cityscapes.bin` | [EoMT official release](https://github.com/tue-mps/eomt) |
| `erfnet_pretrained.pth` | [ERFNet official release](https://github.com/Eromera/erfnet_pytorch) |

### Cityscapes dataset

Place the following zip files in `datasets/cityscapes/` (no extraction needed):
- `leftImg8bit_trainvaltest.zip`
- `gtFine_trainvaltest.zip`

### Anomaly validation datasets

Download `Anomaly_Validation_Datasets.zip` from the course materials and extract into `eval/anomaly_datasets/`.

