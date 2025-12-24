# 🎯 DEMO PRESENTATION - Product Recommendation System

Hệ thống demo với 2 phiên bản: Console và Web Interface

## 📋 Tính năng

### ✨ Demo Console (`demo_presentation.py`)
- **Best Case**: Khách hàng có prediction tốt nhất (precision cao)
- **Worst Case**: Khách hàng có prediction thấp nhất (khó dự đoán)
- **Random Case**: Khách hàng ngẫu nhiên (đại diện trung bình)
- **Feature Importance**: Top 10 features quan trọng nhất
- **Evaluation Metrics**: Precision@10 cho từng case

### 🌐 Demo Web Interface (`demo_web.py`)
- Giao diện đẹp, tương tác trên trình duyệt
- Hiển thị 3 cases với màu sắc phân biệt
- Biểu đồ Feature Importance (interactive)
- Real-time statistics
- Responsive design

## 🚀 Cách chạy

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Chạy Console Demo
```bash
python demo_presentation.py
```

**Output mẫu:**
```
======================================================================
PRODUCT RECOMMENDATION SYSTEM - LIVE DEMO
======================================================================

[1/4] Loading model and predictions...
Model loaded successfully
Predictions loaded: 123,456 rows

[3/4] Finding best, worst, and random demo cases...
Analysis complete: Best=80.0%, Worst=0.0%, Random customer selected

======================================================================
DEMO 1: BEST CASE - Highly Predictable Customer
======================================================================

Customer ID: 12345
Profile Type: Brand Loyal / Highly Predictable

TOP 10 RECOMMENDATIONS:
   1. Item 789 (score: 0.945) ✓ HIT
   2. Item 456 (score: 0.923) ✓ HIT
   ...

EVALUATION:
  Predicted: 10 items
  Actual purchases (Jan 2025): 12 items
  Matched: 8 items
  Precision@10: 80.0%
  Status: BEST CASE ⭐

[... Worst Case, Random Case ...]

======================================================================
FEATURE IMPORTANCE ANALYSIS
======================================================================

Top 10 Most Important Features:
   1. purchase_frequency           ████████████████████ 35.2%
   2. avg_purchase_value           ██████████████ 22.8%
   3. recency                      ████████ 15.4%
   ...
```

### 3. Chạy Web Demo
```bash
python demo_web.py
```

Sau đó mở trình duyệt và truy cập:
```
http://localhost:5000
```

**Screenshot:**
```
┌─────────────────────────────────────────────────────┐
│   🎯 Product Recommendation System                  │
│   Interactive Demo - Best, Worst & Random Cases     │
└─────────────────────────────────────────────────────┘

┌──────────┬──────────┬──────────┬──────────┐
│ LightGBM │  168M    │    13    │  45,678  │
│  Model   │ Samples  │ Features │ Customers│
└──────────┴──────────┴──────────┴──────────┘

┌───────────────┬───────────────┬───────────────┐
│  BEST CASE ⭐ │ WORST CASE ⚠️  │ RANDOM CASE 🎲│
│  80.0%       │    0.0%       │   30.0%       │
│              │               │               │
│  Top 10      │  Top 10       │  Top 10       │
│  ✓ Item 789  │    Item 123   │  ✓ Item 456   │
│  ✓ Item 456  │    Item 789   │    Item 789   │
│  ...         │  ...          │  ...          │
└───────────────┴───────────────┴───────────────┘

📊 Feature Importance Analysis
[Interactive Bar Chart - Top 15 Features]
```

## 📊 Chi tiết các Cases

### Best Case ⭐
- **Mục đích**: Thể hiện khả năng dự đoán tốt nhất của model
- **Đặc điểm**: Khách hàng có pattern mua hàng rõ ràng, trung thành với brand
- **Precision**: Thường ≥ 70%

### Worst Case ⚠️
- **Mục đích**: Thể hiện khó khăn của model với khách hàng phức tạp
- **Đặc điểm**: Khách hàng mua hàng đa dạng, không có pattern rõ ràng
- **Precision**: Thường < 20%

### Random Case 🎲
- **Mục đích**: Đại diện cho khách hàng trung bình
- **Đặc điểm**: Random selection từ test set
- **Precision**: Thường 30-50%

## 🎨 Customization

### Thay đổi số lượng customers phân tích:
Trong file `demo_presentation.py` hoặc `demo_web.py`, sửa dòng:
```python
.head(1000)  # Thay 1000 thành số lượng mong muốn
```

### Thay đổi số features hiển thị:
Trong `demo.html`, sửa dòng:
```javascript
const top15 = data.slice(0, 15);  // Thay 15 thành số lượng mong muốn
```

### Thay đổi port web server:
Trong `demo_web.py`, sửa dòng:
```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Thay 5000
```

## 🔧 Troubleshooting

### Lỗi: Model not found
Kiểm tra đường dẫn model trong code:
```python
model_path = "outputs/models/model_lightgbm_tuned_20251221_103746.pkl"
```

### Lỗi: Flask không cài đặt
```bash
pip install flask
```

### Port 5000 đã được sử dụng
Thay đổi port trong `demo_web.py` hoặc tắt ứng dụng đang dùng port 5000

## 📝 Notes

- Console demo chạy nhanh hơn, phù hợp cho demo nhanh
- Web demo đẹp hơn, phù hợp cho presentation
- Cả 2 demo đều sử dụng cùng một model và data
- Feature importance dựa trên gain (information gain) của LightGBM

## 🎯 Use Cases

1. **Presentation/Meeting**: Dùng Web demo (visual, professional)
2. **Quick Testing**: Dùng Console demo (fast, simple)
3. **Documentation**: Dùng Console demo (easy to copy output)
4. **Client Demo**: Dùng Web demo (impressive, interactive)

---
Tạo bởi: Product Recommendation Team
Model: LightGBM (Tuned Hyperparameters)
Data: 168M training samples, 13 features
