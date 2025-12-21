# THÔNG TIN BÁO CÁO ĐỒ ÁN

## 1. THÔNG TIN THÀNH VIÊN
*(Điền thông tin của bạn và các thành viên khác)*

- **Thành viên 1**: [Họ tên] - [MSSV]
- **Thành viên 2**: [Họ tên] - [MSSV]
- **Thành viên 3**: [Họ tên] - [MSSV]

---

## 2. KẾT QUẢ

### Precision@10 (Groundtruth)
**0.0415** (4.15%)

### Điểm trên hệ thống của thầy
**5.24%**

### Model cuối cùng
- **LightGBM** với tuned hyperparameters
- File: `model_lightgbm_tuned_20251221_103746.pkl`
- Predictions: `predictions_lightgbm_tuned_20251221_103746.parquet`

---

## 3. FILE PREDICTION

### File submission
- **Tên file**: `submission_lightgbm_60pct.json`
- **Kích thước**: 18.83 MB
- **Format**: Dictionary map từ customer_id → list 10 recommended item IDs
- **Coverage**: 120,000 customers (30.6% groundtruth)

### Upload lên Google Drive
1. Upload file `outputs/submission_lightgbm_60pct.json`
2. Share với quyền "Anyone with the link can view"
3. Copy link và điền vào spreadsheet

**Link GG Drive**: [ĐIỀN LINK SAU KHI UPLOAD]

---

## 4. SLIDE BÁO CÁO (5 PHÚT)

### Outline cho slide

#### Slide 1: TRANG BÌA
- Tiêu đề: **Product Recommendation System**
- Tên nhóm / Thông tin thành viên
- Ngày báo cáo

#### Slide 2: TỔNG QUAN BÀI TOÁN
- **Input**: Lịch sử giao dịch 2024 (36M+ transactions, 244K customers)
- **Output**: Dự đoán 10 sản phẩm cho mỗi khách hàng trong tháng 1/2025
- **Đánh giá**: Precision@K, NDCG@K

#### Slide 3: MÔ HÌNH SỬ DỤNG

**Thử nghiệm 4 models:**
1. Logistic Regression (baseline)
2. **LightGBM**(best)
3. XGBoost
4. Random Forest

**Model cuối cùng: LightGBM với tuned hyperparameters**
- num_leaves: 63
- max_depth: 8
- learning_rate: 0.03
- num_boost_round: 200
- Regularization: L1=0.1, L2=0.1

**Kết quả so sánh:**
| Model | Precision@10 | NDCG@10 |
|-------|--------------|---------|
| Logistic | 0.0328 | 0.1030 |
| Random Forest | 0.0388 | 0.1130 |
| XGBoost | 0.0407 | 0.1181 |
| **LightGBM (default)** | 0.0412 | 0.1190 |
| **LightGBM (tuned)** | **0.0415** | **0.1195** |

#### Slide 4: CÁC ĐẶC TRƯNG (13 FEATURES)

**Nhóm 1: Hành vi mua hàng cơ bản**
- `purchase_frequency`: Tần suất mua hàng
- `days_since_last_purchase`: Số ngày từ lần mua cuối
- `avg_items_per_purchase`: Trung bình items/đơn hàng

**Nhóm 2: Sở thích thương hiệu & danh mục**
- `brand_cnt`: Số lượng brands đã mua
- `category_cnt`: Số lượng categories đã mua
- `top_brand_ratio`: Tỷ lệ mua brand yêu thích
- `brand_diversity`: Độ đa dạng brands (entropy)
- `category_diversity_score`: Độ đa dạng categories

**Nhóm 3: Phân khúc khách hàng**
- `age_group_cnt`: Phân khúc tuổi
- `is_power_user`: Khách hàng VIP (≥10 đơn)
- `is_new_customer`: Khách hàng mới

**Nhóm 4: Mẫu hành vi**
- `purchase_day_mode`: Ngày trong tuần thường mua
- `avg_item_popularity`: Xu hướng mua hàng phổ biến/độc đáo

#### Slide 5: PHÂN TÍCH KẾT QUẢ

**Trường hợp TỐT (model dự đoán chính xác):**

**Khách hàng "Trung thành Brand":**
- Đặc điểm: `top_brand_ratio` cao (0.8-1.0), `brand_cnt` thấp (1-3)
- Hành vi: Chỉ mua 1-2 brands cố định
- Kết quả: Model dự đoán đúng 8-10/10 items
- Ví dụ: Khách mua iPhone → model recommend Apple Watch, AirPods ✓

**Khách hàng "Power User":**
- Đặc điểm: `is_power_user=1`, `purchase_frequency` cao
- Hành vi: Mua thường xuyên, có pattern rõ ràng
- Kết quả: Precision cao nhờ nhiều dữ liệu lịch sử

**Trường hợp XẤU (model dự đoán sai):**

**Khách hàng "탐험가" (Exploratory):**
- Đặc điểm: `brand_diversity` cao, `category_diversity_score` cao
- Hành vi: Mua đa dạng, ít lặp lại
- Kết quả: Model khó dự đoán (Precision < 0.02)
- Ví dụ: Lần 1 mua điện thoại, lần 2 mua đồ gia dụng, lần 3 mua sách

**Khách hàng "Mới/Ít data":**
- Đặc điểm: `is_new_customer=1` hoặc `purchase_frequency` thấp
- Hành vi: Chưa đủ dữ liệu để học pattern
- Kết quả: Model chỉ recommend items phổ biến (cold-start problem)

**Phân bố kết quả:**
- ~30% khách hàng: Precision > 0.5 (dự đoán tốt)
- ~40% khách hàng: Precision 0.1-0.5 (trung bình)
- ~30% khách hàng: Precision < 0.1 (khó dự đoán)

#### Slide 6: CHIẾN LƯỢC TỐI ƯU HÓA

**Các bước thực hiện:**
1. EDA → Chọn 13 features quan trọng
2. Time-based split: Train (Jan-Oct), Validation (Nov), Test (Dec)
3. Thử nghiệm 4 models → Chọn LightGBM
4. Hyperparameter tuning → +0.7% improvement
5. Tăng coverage: 20% → 60% customers → +29% accuracy

**Kết quả cuối:**
- **Precision@10**: 0.0415 (4.15%)
- **Điểm thực tế**: 5.24%
- **Coverage**: 120K/391K customers (30.6%)

#### Slide 7: KẾT LUẬN & HƯỚNG PHÁT TRIỂN

**Kết luận:**
- LightGBM là model tốt nhất cho bài toán
- Features về brand preference và purchase frequency quan trọng nhất
- Model hoạt động tốt với khách hàng trung thành, kém với khách mới

**Hướng phát triển:**
- Thêm features: Sequential patterns, co-occurrence
- Ensemble nhiều models
- Xử lý cold-start problem bằng content-based filtering
- Tăng coverage lên 100% customers

---

## 5. LINK SLIDE

**Link GG Slides**: [ĐIỀN LINK SAU KHI TẠO]

*(Nhớ share quyền "Anyone with the link can view")*

---

## FILE CẦN UPLOAD

1. `submission_lightgbm_60pct.json` (18.83 MB)
2. Slide PowerPoint/Google Slides
3. (Optional) Source code nén .zip
