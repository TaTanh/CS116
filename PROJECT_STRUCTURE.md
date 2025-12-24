# 🎯 Product Recommendation System

Hệ thống recommendation sử dụng LightGBM với 13 features, trained trên 168M samples.

## 📁 Cấu trúc thư mục

```
neSemi/
├── 📂 demo/                    # Demo system (Console & Web)
│   ├── demo_presentation.py    # Console demo
│   ├── demo_web.py            # Flask web server
│   ├── run_console_demo.bat   # Shortcut chạy console
│   ├── run_web_demo.bat       # Shortcut chạy web
│   └── templates/             # HTML templates
│       └── demo.html          # Interactive web UI
│
├── 📂 scripts/                 # Training & processing scripts
│   ├── train_*.py             # Model training scripts
│   ├── lightgbm_*.py          # LightGBM variants
│   ├── convert_*.py           # Submission converters
│   └── compare_*.py           # Analysis tools
│
├── 📂 docs/                    # Tài liệu đầy đủ
│   ├── README.md              # Main README
│   ├── README_DEMO.md         # Demo guide chi tiết
│   ├── QUICKSTART.txt         # Quick start
│   ├── BAO_CAO_KET_QUA.md    # Báo cáo kết quả
│   └── *.md                   # Các docs khác
│
├── 📂 src/                     # Source code chính
│   └── recommender/           # Core recommendation modules
│       ├── candidates.py      # Candidate generation
│       ├── features.py        # Feature engineering
│       ├── train.py           # Training pipeline
│       └── ...
│
├── 📂 outputs/                 # Model outputs & results
│   ├── models/                # Trained models (.pkl)
│   ├── predictions/           # Predictions (.parquet)
│   ├── features/              # Feature data
│   └── *.json                 # Metrics & submissions
│
├── 📂 notebooks/               # Jupyter notebooks
│   ├── eda_analysis.ipynb     # Exploratory analysis
│   └── analyze_results.ipynb  # Results analysis
│
├── requirements.txt            # Python dependencies
└── groundtruth.pkl            # Test groundtruth data
```

## 🚀 Quick Start

### 1. Xem Demo (Khuyến nghị!)

#### Console Demo - Terminal output
```bash
cd demo
python demo_presentation.py
```

#### Web Demo - Interactive UI
```bash
cd demo
python demo_web.py
# Mở browser: http://localhost:5000
```

**Features:**
- ⭐ Best Case customer (precision cao nhất)
- ⚠️ Worst Case customer (khó dự đoán nhất)
- 🎲 Random Case (có thể refresh!)
- 📊 Feature Importance chart
- 🔄 Interactive refresh

### 2. Training Model

```bash
cd scripts
python train_lightgbm_parameter.py
```

### 3. Xem Kết quả

```bash
cd scripts
python compare_feature_results.py
```

## 📊 Model Performance

```
Model: LightGBM (Tuned Hyperparameters)
Training Samples: 168M
Features: 13
Test Customers: 644,970

Top Features:
  1. purchase_frequency (31.6%)
  2. category_cnt_hist (19.1%)
  3. days_since_last_purchase (18.9%)
```

## 📖 Documentation

- **[Quick Start](docs/QUICKSTART.txt)** - Bắt đầu nhanh
- **[Demo Guide](docs/README_DEMO.md)** - Hướng dẫn demo đầy đủ
- **[Results Report](docs/BAO_CAO_KET_QUA.md)** - Báo cáo kết quả
- **[Presentation Materials](docs/SLIDE_DETAILS.md)** - Slide details

## 🛠️ Development

### Requirements
```bash
pip install -r requirements.txt
```

### Key Dependencies
- polars>=0.20.0
- lightgbm>=4.0.0
- flask>=3.0.0
- scikit-learn>=1.3.0

## 📝 Notes

- **Console Demo**: Tốt cho testing nhanh, copy output
- **Web Demo**: Tốt cho presentation, interactive
- **Training Scripts**: Trong `scripts/`, chạy từ root directory
- **Outputs**: Tất cả trong `outputs/`, auto-generated

## 🎯 Quick Commands

```bash
# Demo
cd demo && python demo_web.py

# Train model
cd scripts && python train_lightgbm_parameter.py

# Compare results
cd scripts && python compare_feature_results.py

# Create submission
cd scripts && python convert_to_submission.py
```

---

**Tạo bởi:** Product Recommendation Team  
**Last Updated:** December 24, 2025  
**Version:** 2.0 - Restructured ✨
