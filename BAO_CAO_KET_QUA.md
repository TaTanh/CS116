# BÁO CÁO KẾT QUẢ - PRODUCT RECOMMENDATION SYSTEM

## 📊 KẾT QUẢ CUỐI CÙNG

### Score trên hệ thống thầy: **6.89%**

### Metrics đánh giá nội bộ (trên new groundtruth):
- **Precision@10**: 4.38%
- **Recall@10**: 12.30%
- **NDCG@10**: 9.62%
- **F1@10**: 6.46%

---

## 🎯 PHƯƠNG PHÁP

### 1. Model sử dụng
- **Model**: LightGBM (Gradient Boosting)
- **Type**: Classification model (binary prediction)
- **Training**: Đã train trên data 2024 (Jan - Nov)
- **File model**: `outputs/models/model_lightgbm_tuned_20251221_103746.pkl`

### 2. Data và Time Windows
**Training data:**
- Historical period: 01/01/2024 → 01/11/2024 (11 tháng)
- Recent period: 01/11/2024 → 01/12/2024 (1 tháng)
- Transactions: Khoảng 80M records
- Customers: 644,970 customers

**Test data (Groundtruth):**
- File: `final_groundtruth.pkl` (từ thầy)
- Customers: 644,970 (tăng 253,070 so với groundtruth cũ)
- Format: Dictionary {customer_id: [item_ids]}

### 3. Features Engineering (13 features)
1. **X1_brand_cnt_hist**: Số brand đã mua trong lịch sử
2. **X2_age_group_cnt_hist**: Số age group đã mua
3. **X3_category_cnt_hist**: Số category đã mua
4. **X4_days_since_last_purchase**: Số ngày từ lần mua cuối
5. **X5_purchase_frequency**: Tần suất mua hàng
6. **X6_is_power_user**: Có phải power user không
7. **X7_avg_items_per_purchase**: Trung bình items/đơn
8. **X8_top_brand_ratio**: Tỷ lệ mua brand yêu thích
9. **X9_brand_diversity**: Độ đa dạng brand
10. **X10_category_diversity_score**: Độ đa dạng category
11. **X11_purchase_day_mode**: Ngày trong tuần hay mua
12. **X12_is_new_customer**: Khách hàng mới hay cũ
13. **X13_avg_item_popularity**: Độ phổ biến trung bình của items

### 4. Submission Strategy
- **Customers submitted**: 100,000 customers
- **Selection method**: Top customers theo average prediction score
- **Items per customer**: 10 items (top 10 predictions)
- **File size**: 14.33 MB
- **File**: `outputs/submission_lightgbm_optimized.json`

---

## 🔄 QUY TRÌNH THỰC HIỆN

### Bước 1: Chuẩn bị Data
```bash
# Convert groundtruth mới
python convert_groundtruth.py

# Convert data tháng 1/2025 (nếu cần train lại)
python convert_jan2025_data.py
```

### Bước 2: Generate Predictions
```bash
# Dùng model đã train sẵn (KHÔNG cần train lại)
python generate_predictions_new_groundtruth.py
```
**Output:**
- Predictions file: `outputs/predictions/predictions_new_groundtruth_20251221_222506.parquet`
- Metrics file: `outputs/metrics_new_groundtruth_20251221_222506.json`
- Customers with predictions: 463,340

### Bước 3: Tạo Submission
```bash
python optimize_submission.py
```
**Output:**
- Submission file: `outputs/submission_lightgbm_optimized.json` (14.33 MB)
- Top 100K customers, 10 items each

---

## 💡 INSIGHTS VÀ PHÂN TÍCH

### Điểm mạnh của approach:
1. **Không cần train lại model**
   - Tiết kiệm thời gian (5-10 phút vs 1-2 giờ)
   - Tiết kiệm RAM
   - Model cũ (train trên 11 tháng 2024) vẫn rất tốt

2. **Chọn lọc customers thông minh**
   - Chỉ submit top 100K customers có score cao nhất
   - Tăng precision (focus vào predictions tốt nhất)
   - Giảm file size (dễ upload, dễ xử lý)

3. **Feature engineering đa dạng**
   - Kết hợp features về behavior (purchase frequency, recency)
   - Features về preferences (brand, category diversity)
   - Features về popularity (item popularity)

### So sánh với baseline:
- **Groundtruth cũ**: 391,900 customers
- **Groundtruth mới**: 644,970 customers (+65% customers)
- **Coverage**: 463,340 / 644,970 = 71.8% customers có predictions

### Kết quả:
- **Public score**: 6.89%
- **Tốt hơn random baseline** (< 1%)
- **Precision@10**: 4.38% → có thể cải thiện bằng ensemble models

---

## 📁 CẤU TRÚC FILES QUAN TRỌNG

### Input Files:
- `groundtruth.pkl` - Test set (644,970 customers)
- `final_groundtruth.pkl` - Groundtruth gốc từ thầy
- `01-2025.pkl` - Data tháng 1/2025 (nếu cần)

### Model Files:
- `outputs/models/model_lightgbm_tuned_20251221_103746.pkl` - Best model

### Output Files:
- `outputs/predictions/predictions_new_groundtruth_20251221_222506.parquet` - Predictions
- `outputs/metrics_new_groundtruth_20251221_222506.json` - Metrics
- `outputs/submission_lightgbm_optimized.json` - **SUBMISSION FILE** (14.33 MB)

### Scripts:
- `convert_groundtruth.py` - Convert groundtruth format
- `generate_predictions_new_groundtruth.py` - Generate predictions
- `optimize_submission.py` - Create submission file

---

## 🎓 KẾT LUẬN

### Thành công:
- ✅ Xây dựng hệ thống recommendation hoàn chỉnh
- ✅ Đạt score 6.89% trên public test set
- ✅ Xử lý 644K+ customers, 80M+ transactions
- ✅ Approach thực tế, tối ưu (không cần train lại)

### Bài học:
1. **Feature engineering quan trọng hơn model phức tạp**
2. **Chọn lọc customers thông minh** (top score) tăng precision
3. **Model đơn giản nhưng tốt** có thể dùng lại cho test set mới

### Cải thiện có thể:
- Ensemble multiple models (LightGBM + XGBoost + Random Forest)
- Thêm features về item characteristics
- Hyperparameter tuning kỹ hơn
- Tăng số customers submit (nếu server cho phép)

---

**Ngày hoàn thành**: 21/12/2025
**Model**: LightGBM Tuned
**Final Score**: 6.89%
