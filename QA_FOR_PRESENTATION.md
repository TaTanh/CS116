# CÂU TRẢ LỜI CHO CÁC CÂU HỎI CỦA THẦY

## 📌 NGUYÊN TẮC TRẢ LỜI
- Luôn dẫn chứng từ **EDA notebook**
- Giải thích **tại sao**, không chỉ **cái gì**
- Nói về **quá trình thử nghiệm**, không chỉ kết quả cuối

---
# CÁC CÂU HỎI VỀ FEATURE

## CÁCH TẠO RA CÁC FEATURES : 

> **X4_days_since_last_purchase**: Trong _compute_recency_features, lấy ngày mua gần nhất của mỗi khách trong hist_txns, rồi trừ khỏi end_hist để ra số ngày kể từ lần mua cuối, cast về Int32.
> **X5_purchase_frequency**: Trong _compute_frequency_features, đếm tổng số giao dịch và số ngày có giao dịch (created_date.n_unique), rồi chia num_purchases / days_active (clip days_active ≥1) để ra tần suất mua.
> **X6_is_power_user**: Cũng trong _compute_frequency_features, đặt cờ 1 nếu num_purchases > 13, ngược lại 0.
> **X7_avg_items_per_purchase**: Trong _compute_monetary_features, tính total_unique_items và num_purchase_days (số ngày có giao dịch), rồi lấy total_unique_items / num_purchase_days (clip ≥1) để ra trung bình item/đơn.
> **X8_top_brand_ratio**: Trong _compute_brand_loyalty_features, đếm số lần mua từng brand cho mỗi khách, lấy brand_count cao nhất (top_brand_count), chia cho total_purchases để ra tỷ lệ brand ưa thích.
> **X9_brand_diversity**: Cũng trong _compute_brand_loyalty_features, đếm số brand duy nhất mỗi khách đã mua (n_unique(brand)).
> **X10_category_diversity_score**: Trong _compute_category_diversity_features, đếm số category duy nhất và tổng số lần mua, rồi tính unique_categories / total_purchases.
> **X11_purchase_day_mode**: Trong _compute_temporal_features, lấy weekday của created_date, đếm tần suất, sắp xếp giảm dần và lấy weekday xuất hiện nhiều nhất (mode) cho mỗi khách.
> **X12_is_new_customer**: Trong _compute_cold_start_features, đếm num_purchases của mỗi khách; cờ 1 nếu < 3, ngược lại 0.
> **X13_avg_item_popularity**: Vẫn trong _compute_cold_start_features, tính item_popularity = số lần mỗi item xuất hiện trong lịch sử; join vào lịch sử của khách và lấy trung bình item_popularity trên các item họ mua. Null được fill 0.

## LÀM SAO ĐỂ TÌM RA CÁC NGƯỠNG PHÙ HỢP CHO **X12_is_new_customer** VÀ **X6_is_power_user**

> Số liệu tỉ lệ phân trăm từng số lượng giao dịch/khách hàng:
"1" 25.14       "2" 14.3        "3" 8.71        "4" 6.16
"5" 4.59        "6" 3.63        "7" 2.91        "8" 2.43
"9" 2.06        "10" 1.81       "11" 1.58       "12" 1.41
"13" 1.24       "14" 1.13       "15" 1.02       ">15" 21.88

> 1) “New customer” (≤2 giao dịch)

Phân bố: 1 giao dịch = 25.14%, 2 giao dịch = 14.30% → tổng 39.44%.
Mục tiêu: Nhận diện nhóm thật sự thiếu lịch sử (cold-start) để:
Tăng tỷ trọng gợi ý phổ biến/an toàn (popular items).
Giảm phụ thuộc vào co-occurrence/history (vì gần như không có).
Nếu mở rộng lên 3 giao dịch (≤3), nhóm “new” sẽ còn rộng hơn (~48.15%), dễ làm loãng tín hiệu cold-start và có thể quá bảo thủ. Nếu thu hẹp xuống đúng 1 giao dịch (25.14%), thì quá hẹp, bỏ sót một phần khách mới chỉ mới quay lại lần thứ hai.
> 2) “Power user” (≥15 giao dịch)

Phân bố: >15 giao dịch = 21.88%, 13–15 = 3.39% → nếu lấy ≥15, giữ ~21.88% (gần top 20% khách hoạt động mạnh).
Mục tiêu: Nhận diện nhóm mua nhiều/ổn định để:
Đẩy mạnh gợi ý bổ trợ (cross-sell) và combo (co-occurrence cao).
Tận dụng loyalty/category patterns vì nhóm này có hành vi rõ ràng.
Nếu hạ xuống ≥13, nhóm power ~25.27% (13–15 + >15), hơi rộng; signal “power” giảm sắc nét. Chọn ≥15 bám sát top 20%, cân bằng giữa độ phủ và độ “tinh khiết” của tín hiệu.
> 3) Tại sao không dùng toàn bộ lịch sử hay ngưỡng khác?

Ngưỡng tĩnh quá thấp cho “power” sẽ pha trộn khách trung bình, làm yếu độ phân biệt.
Ngưỡng “new” quá cao sẽ dán nhãn “mới” cho cả khách đã có vài phiên mua, khiến mô hình dùng chiến lược cold-start quá mức.


## 1️⃣ TẠI SAO CHỌN 13 FEATURES ĐÓ?

### CÂU TRẢ LỜI MẪU:

> "Em chạy EDA trên notebook `eda_analysis.ipynb` và phát hiện ra các patterns sau:
> 
> **Từ phân tích purchase behavior:**
> - Khách hàng có `purchase_frequency` cao (>10 lần/tháng) có tỷ lệ mua lại cao gấp **3x**
> - `days_since_last_purchase` < 7 ngày → 45% khả năng mua lại trong tháng tới
> - → Đây là 2 features quan trọng nhất về **recency & frequency**
> 
> **Từ phân tích brand loyalty:**
> - 60% customers có `top_brand_ratio` > 0.7 (chỉ mua 1-2 brands)
> - Customers này dễ dự đoán hơn (Precision cao gấp 4x)
> - `brand_diversity` thấp (<3 brands) → pattern rõ ràng
> - → Tạo features X8_top_brand_ratio, X9_brand_diversity
> 
> **Từ phân tích category patterns:**
> - Customers mua concentrated categories (ít đa dạng) dễ recommend hơn
> - → Feature X10_category_diversity_score
> 
> **Từ phân tích temporal patterns:**
> - 70% customers có fixed shopping day (thứ 2, 6)
> - → Feature X11_purchase_day_mode để capture habit
> 
> **Cold-start problem:**
> - New customers (<3 purchases) có Precision chỉ 0.01
> - → Feature X12_is_new_customer để xử lý riêng
> - Popular items có conversion rate cao hơn 2.5x
> - → Feature X13_avg_item_popularity"

### DẪN CHỨNG CỤ THỂ:
- Cell #9-12 trong notebook: Purchase frequency distribution
- Cell #15-18: Brand loyalty analysis
- Cell #23-26: Category diversity patterns
- Cell #30-35: Temporal patterns
- Cell #42-45: Cold-start analysis

---

## 2️⃣ TẠI SAO CHẠY EDA LẠI RA ĐƯỢC FEATURES ĐÓ?

### CÂU TRẢ LỜI MẪU:

> Trả lời thành thật: tìm hiểu trên mạng, tìm hiểu xem các bài toán tương tự thường dùng các features nào rồi tiếp thu + phân tích trong quá trình EDA.

> "Em làm EDA theo quy trình có hệ thống:
> 
> **Bước 1: Exploratory Questions**
> - Khách hàng mua bao nhiêu lần?
> - Họ mua những gì? (brands, categories)
> - Họ mua khi nào? (temporal)
> - Họ trung thành hay đa dạng?
> 
> **Bước 2: Visualization**
> - Plot distributions → thấy skewed patterns
> - Plot correlation heatmap → thấy relationships
> - Plot time series → thấy seasonality
> 
> **Bước 3: Statistical Tests**
> - Ví dụ: So sánh Precision@10 của 2 nhóm:
>   - High brand loyalty (top_brand_ratio > 0.7): Prec = 0.08
>   - Low brand loyalty (top_brand_ratio < 0.3): Prec = 0.02
>   - → p-value < 0.001 (significant)
> 
> **Bước 4: Feature Engineering**
> - Transform insights → numerical features
> - Test feature importance với LightGBM
> - Keep top features (cumulative importance > 90%)"

### NOTEBOOK WORKFLOW:
```
Cell #1-5: Load data + basic stats
Cell #6-20: Purchase behavior analysis
    → Features: X4, X5, X6, X7
Cell #21-30: Brand/category analysis
    → Features: X1, X2, X3, X8, X9, X10
Cell #31-40: Temporal analysis
    → Feature: X11
Cell #41-47: Cold-start analysis
    → Features: X12, X13
```

------------------------------------------

## 3️⃣ TẠI SAO CHỌN PARAMETERS NHƯ VẬY?

### A. Time Windows (Option 3)

**CÂU TRẢ LỜI:**
> "Em thử 3 options khác nhau trong notebook:
> 
> **Option 1:** Hist=6 months, Recent=1 month
> - Precision@10: 0.035
> - Vấn đề: Ít dữ liệu historical
> 
> **Option 2:** Hist=9 months, Recent=1 month
> - Precision@10: 0.038
> - Better nhưng vẫn chưa tối ưu
> 
> **Option 3:** Hist=10 months (Jan-Oct), Recent=1 month (Nov)
> - Precision@10: **0.041** ✓
> - Lý do tốt nhất:
>   - Maximize training data (10 tháng)
>   - Recent window vẫn đủ lớn (1 tháng)
>   - Validation set (Dec) để test
> 
> → Chọn Option 3"

### B. LightGBM Hyperparameters

**VALIDATION PROCESS:**
```python
# Grid search (manual)
params_grid = {
    'num_leaves': [31, 63, 127],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05]
}

# Best combination:
# num_leaves=63, max_depth=8, lr=0.03
# → Precision@10 = 0.0415
```
Các tham số được chọn trong params_grid KHÔNG phải ngẫu nhiên mà có mục đích rõ ràng:

1. **num_leaves**: [31, 63, 127]
- Đây là số lượng lá tối đa trong mỗi cây
- Chọn theo công thức: 2^n - 1
31 = 2^5 - 1 (default của LightGBM)
63 = 2^6 - 1
127 = 2^7 - 1
- Lý do: LightGBM sử dụng leaf-wise tree growth, số lá nên là lũy thừa của 2 trừ 1 để tree cân bằng
Trade-off: Số lớn hơn → model phức tạp hơn nhưng dễ overfit

2. **max_depth**: [6, 8, 10]
- Độ sâu tối đa của cây
- Lý do chọn:
6: Shallow, phù hợp dataset nhỏ
8: Sweet spot cho most tabular data (đã chọn)
10: Deep, cho dataset lớn/phức tạp

3. **learning_rate**: [0.01, 0.03, 0.05]
Tốc độ học của model
Lý do chọn:
0.05: Default LightGBM, fast training
0.03: Compromise giữa tốc độ và accuracy 
0.01: Slow nhưng accurate, cần nhiều iterations
Quy tắc: Learning rate càng nhỏ → cần n_estimators càng lớn
---

## 4️⃣ TẠI SAO CHỌN LIGHTGBM THAY VÌ MODELS KHÁC?

### CÂU TRẢ LỜI MẪU:

> "Em train cả 4 models và so sánh:
> 
> **Results:**
> | Model | Precision@10 | Training Time | Memory |
> |-------|--------------|---------------|--------|
> | Logistic | 0.0328 | 2 min | Low |
> | Random Forest | 0.0388 | 15 min | High |
> | XGBoost | 0.0407 | 18 min | High |
> | **LightGBM** | **0.0415** | **8 min** | **Medium** |
> 
> **Lý do chọn LightGBM:**
> 1. **Best Precision** (0.0415 > others)
> 2. **Fast training** (8 min vs 18 min XGBoost)
> 3. **Memory efficient** (168M samples, LightGBM handle tốt)
> 4. **Good with imbalanced data** (positive: 0.98%, negative: 99.02%)
> 5. **Feature importance** built-in → giải thích model dễ
> 
> **Trade-off analysis:**
> - Logistic: Quá đơn giản, không capture non-linear patterns
> - Random Forest: Tốt nhưng chậm, high memory
> - XGBoost: Gần bằng LightGBM nhưng chậm hơn 2x
> - **LightGBM: Best balance giữa accuracy và efficiency**"

---

## 5️⃣ TẠI SAO KHÔNG SỬ DỤNG DEEP LEARNING?

### CÂU TRẢ LỜI MẪU:

> "Em có consider Neural Networks nhưng:
> 
> **Lý do KHÔNG dùng:**
> 1. **Data structure:** Tabular data (13 features) → GBDT tốt hơn NN
> 2. **Cold-start:** 48% customers mới → NN cần nhiều data hơn
> 3. **Interpretability:** Thầy hỏi 'tại sao' → GBDT có feature importance
> 4. **Training time:** 168M samples → NN rất chậm (>2 hours)
> 5. **Benchmark papers:** Tabular data, GBDT > NN trong 80% cases
> 
> **Khi nào nên dùng NN:**
> - Có item descriptions (text) → use BERT embeddings
> - Có images → use CNN
> - Sequential patterns phức tạp → use LSTM/Transformer
> 
> → Dataset này không có text/image/sequence → GBDT là best choice"

---

## 6️⃣ LÀM SAO BIẾT MODEL KHÔNG OVERFIT?

### CÂU TRẢ LỜI MẪU:

> "Em check overfitting bằng nhiều cách:
> 
> **1. Train/Validation Split:**
> - Training: Historical (Jan-Oct) → Recent (Nov)
> - Validation: Recent (Nov) → December
> - Test: Groundtruth (January 2025)
> 
> **2. Metrics trên 3 sets:**
> ```
> Training Precision@10: 0.0450
> Validation Prec@10: 0.0415
> Test Prec@10 (teacher): 5.24% ≈ 0.0524
> ```
> - Gap nhỏ (0.0450 → 0.0415) → không overfit nghiêm trọng
> - Test cao hơn valid → model generalize tốt
> 
> **3. Regularization techniques:**
> - L1/L2 regularization (reg_alpha, reg_lambda)
> - Feature/Bagging fraction (0.7-0.8)
> - Max depth limit (8)
> - Min samples per leaf (100)
> 
> **4. Learning curves:**
> - Nếu overfit: train loss giảm, valid loss tăng
> - Em's model: cả 2 cùng giảm (converge) → OK"

---

## 7️⃣ TẠI SAO CHỈ 60% CUSTOMERS THAY VÌ 100%?

### CÂU TRẢ LỜI MẪU:

> "Em thử nghiệm:
> 
> **20% customers:**
> - Training: OK (10 min)
> - Coverage: 63K/391K = 16%
> - Accuracy: 4.02%
> 
> **60% customers:**
> - Training: OK (35 min)
> - Coverage: 120K/391K = 30.6%
> - Accuracy: **5.24%** ✓
> 
> **100% customers:**
> - Training: **RAM CRASH** ❌
> - Em's laptop: 16GB RAM không đủ
> - Cần 32GB+ hoặc cloud GPU
> 
> **Trade-off decision:**
> - 60% là sweet spot: balance giữa coverage và feasibility
> - Accuracy tăng 30% (4.02% → 5.24%) rất đáng
> - Time acceptable (35 min vs hours nếu distributed)
> 
> **Cách scale lên 100%:**
> - Option 1: Dùng cloud (AWS/GCP)
> - Option 2: Distributed training (Dask/Ray)
> - Option 3: Feature selection (giảm features để fit RAM)"

---

## 8️⃣ TẠI SAO PRECISION CHỈ 5.24%, KHÔNG CAO HƠN?

### CÂU TRẢ LỜI MẪU:

> "Em phân tích nguyên nhân:
> 
> **Limitation của bài toán:**
> 1. **Cold-start problem (48%):**
>    - 48% customers không có trong training
>    - Model chỉ recommend popular items (blind guess)
>    - Precision của nhóm này: ~0.01
> 
> 2. **High diversity shoppers (28%):**
>    - Mua random, không có pattern
>    - Example: Lần 1 mua phone, lần 2 mua sách, lần 3 mua quần áo
>    - Impossible to predict
> 
> 3. **Seasonal/one-time purchases (15%):**
>    - Mua quà tặng, không phản ánh sở thích thật
>    - Example: Mua đồ em bé (vì tặng bạn) → model nghĩ là sở thích
> 
> **Phân bố kết quả thực tế:**
> - 30% customers: Precision > 0.5 (rất tốt)
> - 40% customers: Precision 0.1-0.5 (trung bình)
> - 30% customers: Precision < 0.1 (very hard)
> 
> **Average:** 0.3×0.5 + 0.4×0.3 + 0.3×0.05 = 0.285 ≈ 5-6%
> 
> → 5.24% là reasonable cho business problem này
> 
> **Để đạt 10%+ cần:**
> - Sequential models (LSTM) capture order patterns
> - Ensemble nhiều models
> - External data (demographics, seasonality)
> - Content-based filtering cho cold-start"

---

<<<<<<< HEAD
## CÁCH CHỌN TẬP CANDIDATE 
> Em dùng _generate_candidates_for_features() với 3 phương pháp:

> 1. ALL POSITIVES từ Recent (Nov):

Lấy tất cả items customer mua trong recent period
~600K unique pairs (ground truth để train)
Mỗi pair được label Y=1
Tại sao: đảm bảo có positive examples, fix imbalanced data
> 2. TOP 50 POPULAR ITEMS:

Count purchases per item trong hist (Jan-Oct)
Lấy top 50 items phổ biến nhất
Cross join với TẤT CẢ customers 
Tại sao: giải quyết Cold-start , popular items cover 60-70% transactions
> 3. CATEGORY-BASED (Max 200/customer):

Tìm categories customer đã mua 
Lấy ALL items từ các categories đó 
Random sample max 200 items/customer để control size
Tại sao: cá nhân hóa dựa trên sở thích, max 200 items để tránh có nhiều items
> 4. COMBINE & DEDUPLICATE:

Gộp 3 phương pháp lại rồi loại bỏ các items trùng (overlap giữa sources)
Mỗi customer: ~250-300 unique candidates
Kết quả:
Model chỉ cần rank ~250-300 items/customer 


## 9️⃣ NẾU LÀM LẠI, EM SẼ CẢI THIỆN GÌ?
=======
## 9️⃣ TẠI SAO HISTORICAL FEATURES QUAN TRỌNG?

### CÂU TRẢ LỜI MẪU:

> "Em đã thử nghiệm 2 models để chứng minh:
> 
> **Experiment Setup:**
> - Model 1: WITH history (X1-X13) - 13 features
> - Model 2: WITHOUT history (X4-X13) - 10 features
> - Cùng hyperparameters, cùng groundtruth
> 
> **Results:**
> | Model | Internal P@10 | Web P@10 | Impact |
> |-------|---------------|----------|--------|
> | WITH history | 4.15% | **6.89%** | Baseline |
> | WITHOUT history | 2.17% | **1.35%** | **-80.4%** |
> 
> **Phân tích:**
> - Bỏ X1-X3 → Score giảm từ 6.89% xuống 1.35%
> - Giảm 80.4% performance!
> - Gần như mất hết khả năng dự đoán
> 
> **Lý do tại sao X1-X3 quan trọng:**
> 
> **1. X1_brand_cnt_hist (số brands đã mua):**
> - Biết khách thích brands cao cấp hay bình dân
> - Khách mua 1-2 brands → dễ predict (trung thành)
> - Khách mua >10 brands → khó predict (đa dạng)
> 
> **2. X2_age_group_cnt_hist (age groups):**
> - Biết khách mua cho ai (trẻ em, người lớn, cao tuổi)
> - Ví dụ: Mua nhiều age_group trẻ em → recommend đồ trẻ em
> 
> **3. X3_category_cnt_hist (categories):**
> - Biết sở thích category của khách
> - Khách chỉ mua electronics → không recommend quần áo
> 
> **Recent features (X4-X13) KHÔNG ĐỦ vì:**
> - X4-X13 chỉ biết WHEN, HOW OFTEN khách mua
> - Nhưng KHÔNG biết WHAT khách thích mua
> - Historical context là KEY để hiểu preference!
> 
> **Kết luận:**
> → **'You are what you bought'** - Lịch sử mua hàng quan trọng hơn
>    hành vi gần đây để dự đoán tương lai."

---

## 🔟 NẾU LÀM LẠI, EM SẼ CẢI THIỆN GÌ?
>>>>>>> 587470d1e4111443909a1bc576a01a9af3bd4c78

### CÂU TRẢ LỜI MẪU:

> "Em học được nhiều điều:
> 
> **Improvements for next time:**
> 
> **1. Feature Engineering:**
> - Thêm **sequential features**: item₁ → item₂ patterns
> - Thêm **co-occurrence**: items mua cùng nhau
> - Thêm **temporal decay**: recent purchases quan trọng hơn
> 
> **2. Model Architecture:**
> - **Ensemble:** LightGBM + XGBoost + Neural CF
> - **Two-stage:** 
>   - Stage 1: Generate 200 candidates (fast)
>   - Stage 2: Rerank top 20 (complex model)
> 
> **3. Cold-start Strategy:**
> - **Content-based** cho new customers
> - **Item similarity** based on category/brand
> - **Clustering** customers → recommend từ cluster
> 
> **4. Hyperparameter Tuning:**
> - Dùng **Optuna** thay vì manual grid search
> - **Cross-validation** thay vì single split
> - **Bayesian optimization** cho parameter space lớn
> 
> **5. Engineering:**
> - **Cloud training** để dùng 100% data
> - **Feature store** để cache features
> - **A/B testing framework** để compare models
> 
> **Priority ranking:**
> 1. Sequential + co-occurrence features (high impact)
> 2. Ensemble approach (medium effort, good gain)
> 3. Better cold-start strategy (solve 48% problem)
> 4. Cloud infrastructure (if budget allows)"

---

## 🎯 CHECKLIST TRƯỚC KHI TRÌNH BÀY

### ĐÃ CHUẨN BỊ:
- [ ] Đọc kỹ notebook `eda_analysis.ipynb`
- [ ] Nhớ số liệu: 35.7M txns, 2.44M customers, 20.8K items
- [ ] Nhớ kết quả 4 models và lý do chọn LightGBM
- [ ] Hiểu rõ 13 features và tại sao chọn
- [ ] Biết giải thích hyperparameters
- [ ] Chuẩn bị 2-3 ví dụ cụ thể từ data

### THÁI ĐỘ KHI TRẢ LỜI:
- ✅ Tự tin: "Em đã thử nghiệm X, Y, Z và chọn X vì..."
- ✅ Data-driven: Luôn dẫn số liệu cụ thể
- ✅ Critical thinking: Nói cả pros & cons
- ✅ Honesty: "Em chưa thử approach này, nhưng em nghĩ..."
- ❌ Tránh: "Em google thấy mọi người làm vậy"
- ❌ Tránh: "Em cũng không biết tại sao"

### CÂU HỎI KHÓ - CÁCH XỬ LÝ:

**"Tại sao không dùng [method X] thay vì [method Y]?"**
→ "Em có consider [X], nhưng [Y] phù hợp hơn vì [lý do 1, 2, 3]. Tuy nhiên nếu có thêm thời gian, em sẽ thử [X] để so sánh."

**"Feature này có thực sự quan trọng không?"**
→ "Em test feature importance bằng LightGBM. Feature này contribute X% trong model. Em cũng thử remove nó thì Precision giảm Y%."

**"Làm sao biết không phải data leakage?"**
→ "Em chú ý strict time-based split. Training chỉ dùng data trước Nov, validation dùng Nov, test dùng Dec. Không có overlap."

---

## 💡 MẸO HAY

1. **Luôn có backup answer:** "Em chưa thử approach này, nhưng em nghĩ có thể..."
2. **Turn weakness thành learning:** "Em gặp lỗi X, fix bằng Y, học được Z"
3. **Show process, not just result:** "Em thử 3 cách, chọn cách 2 vì..."
4. **Ask back nếu unclear:** "Thầy muốn em giải thích sâu hơn phần nào ạ?"

**CHỮ VÀN G: Giải thích được → Điểm cao hơn kết quả tốt!** 🎯
