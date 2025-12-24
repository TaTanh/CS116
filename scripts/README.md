# 📜 Training & Processing Scripts

Các scripts để training models và xử lý data.

## 📁 Files

### Model Training
- **train_all_models.py** - Train tất cả models (Logistic, Random Forest, XGBoost, LightGBM)
- **train_lightgbm_3features.py** - LightGBM với 3 features
- **train_lightgbm_5features.py** - LightGBM với 5 features
- **train_lightgbm_9features.py** - LightGBM với 9 features
- **train_lightgbm_parameter.py** - LightGBM với tuned parameters
- **train_lightgbm_without_history.py** - LightGBM không dùng history features

### Groundtruth Processing
- **lightgbm_with_newgroundtruth.py** - Train với groundtruth mới
- **lightgbm_without_history_newgroundtruth.py** - Train không history với groundtruth mới

### Submission Processing
- **convert_to_submission.py** - Convert predictions thành submission format
- **convert_to_submission_without_history.py** - Convert cho model không history
- **optimize_submission.py** - Optimize submission file size

### Analysis
- **compare_feature_results.py** - So sánh kết quả giữa các feature sets

## 🚀 Cách dùng

```bash
# Training model chính
python train_lightgbm_parameter.py

# Training với feature sets khác nhau
python train_lightgbm_3features.py
python train_lightgbm_5features.py
python train_lightgbm_9features.py

# So sánh kết quả
python compare_feature_results.py

# Tạo submission
python convert_to_submission.py
python optimize_submission.py
```

## 📊 Output

Tất cả outputs được lưu trong thư mục `../outputs/`:
- Models: `outputs/models/`
- Predictions: `outputs/predictions/`
- Metrics: `outputs/metrics_*.json`
- Submissions: `outputs/submission_*.json`
