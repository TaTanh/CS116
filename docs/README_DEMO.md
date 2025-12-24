# 🎯 Product Recommendation System - DEMO

## ✨ Tính năng Demo mới

Demo system hiện có **2 phiên bản** với đầy đủ tính năng:

### 📊 Nội dung Demo
1. **Best Case** ⭐ - Khách hàng dễ dự đoán nhất (precision cao nhất)
2. **Worst Case** ⚠️ - Khách hàng khó dự đoán nhất (precision thấp nhất)
3. **Random Case** 🎲 - Khách hàng ngẫu nhiên (đại diện trung bình)
4. **Feature Importance** 📈 - Top features quan trọng của model
5. **Overall Statistics** 📊 - Tổng quan performance

---

## 🚀 Cách chạy Demo

### Option 1: Console Demo (Khuyến nghị cho Testing nhanh)

**Cách 1: Double-click file BAT**
```
Chạy file: run_console_demo.bat
```

**Cách 2: Chạy từ terminal**
```bash
E:\Nam_3_HK1\PythonMayHoc\neSemi\.venv\Scripts\python.exe demo_presentation.py
```

**Ưu điểm:**
- ⚡ Nhanh, đơn giản
- 📝 Dễ copy output
- 🎯 Tập trung vào kết quả

**Thời gian:** ~30 giây

---

### Option 2: Web Demo (Khuyến nghị cho Presentation)

**Cách 1: Double-click file BAT**
```
Chạy file: run_web_demo.bat
```

**Cách 2: Chạy từ terminal**
```bash
E:\Nam_3_HK1\PythonMayHoc\neSemi\.venv\Scripts\python.exe demo_web.py
```

Sau đó mở trình duyệt và truy cập:
```
http://localhost:5000
```

**Ưu điểm:**
- 🎨 Giao diện đẹp, professional
- 📊 Biểu đồ interactive
- 🎯 Màu sắc phân biệt rõ ràng
- 📱 Responsive design

**Thời gian:** ~30 giây loading + web interface

---

## 📋 Kết quả Demo

### Console Demo Output
```
======================================================================
DEMO 1: BEST CASE - Highly Predictable Customer
======================================================================
Customer ID: 5862045
Profile Type: Brand Loyal / Highly Predictable

TOP 10 RECOMMENDATIONS:
   1. Item 1371000000004 (score: 0.278)  
   2. Item 1371000000002 (score: 0.278)  
   3. Item 5420000000003 (score: 0.278) ✓ HIT
   ...
   
EVALUATION:
  Predicted: 10 items
  Actual purchases: 8 items
  Matched: 6 items
  Precision@10: 60.0%
  Status: BEST CASE ⭐

[... WORST CASE, RANDOM CASE ...]

FEATURE IMPORTANCE ANALYSIS:
   1. X5_purchase_frequency    ████████████ 31.6%
   2. X3_category_cnt_hist     ████████ 19.1%
   3. X4_days_since_last       ████████ 18.9%
   ...
```

### Web Demo Features
- **Interactive Charts**: Feature importance bar chart
- **Color Coding**: Green (Best), Red (Worst), Orange (Random)
- **Statistics Cards**: Model info, training size, features count
- **Real-time Data**: Auto-load từ API endpoints
- **Responsive**: Hoạt động tốt trên mọi màn hình

---

## 🎯 Chi tiết các Cases

### Best Case ⭐
- **Precision**: Thường 50-80%
- **Đặc điểm**: Pattern mua hàng rõ ràng, loyal customer
- **Use case**: Thể hiện điểm mạnh của model

### Worst Case ⚠️
- **Precision**: Thường 0-20%
- **Đặc điểm**: Mua hàng đa dạng, không có pattern
- **Use case**: Thể hiện giới hạn của model

### Random Case 🎲
- **Precision**: Thường 20-50%
- **Đặc điểm**: Khách hàng trung bình
- **Use case**: Đại diện cho average performance

---

## 📊 Feature Importance

Top 5 Features quan trọng nhất:

1. **X5_purchase_frequency** (31.6%)
   - Tần suất mua hàng của khách hàng
   
2. **X3_category_cnt_hist** (19.1%)
   - Số lượng categories đã mua
   
3. **X4_days_since_last_purchase** (18.9%)
   - Số ngày kể từ lần mua cuối
   
4. **X1_brand_cnt_hist** (9.0%)
   - Số lượng brands đã mua
   
5. **X9_brand_diversity** (8.8%)
   - Mức độ đa dạng trong việc chọn brand

**Total:** 13 features được sử dụng

---

## 🔧 Troubleshooting

### Model không tìm thấy
```
❌ Error: Model not found
```
**Giải pháp:** Kiểm tra file tồn tại:
```
outputs/models/model_lightgbm_tuned_20251221_103746.pkl
```

### Flask không cài đặt
```
❌ Error: No module named 'flask'
```
**Giải pháp:** Đã được cài đặt sẵn trong virtual environment

### Port 5000 bị chiếm
```
❌ Error: Address already in use
```
**Giải pháp:** 
1. Tắt ứng dụng đang dùng port 5000
2. Hoặc đổi port trong [demo_web.py](demo_web.py#L152): `app.run(port=5001)`

---

## 📦 Files được tạo

### Demo Scripts
- `demo_presentation.py` - Console demo script
- `demo_web.py` - Web server Flask app

### Batch Files (Windows)
- `run_console_demo.bat` - Chạy console demo
- `run_web_demo.bat` - Chạy web demo

### Templates
- `templates/demo.html` - Web UI template

### Documentation
- `DEMO_GUIDE.md` - Hướng dẫn chi tiết
- `QUICKSTART.txt` - Hướng dẫn nhanh
- `README_DEMO.md` - File này

---

## 💡 Tips & Best Practices

### Cho Presentation
1. ✅ Dùng **Web Demo** - giao diện đẹp, professional
2. ✅ Mở browser trước khi start
3. ✅ Test trước 1 lần để đảm bảo không có lỗi
4. ✅ Screenshot web demo để backup

### Cho Testing
1. ✅ Dùng **Console Demo** - nhanh hơn, dễ debug
2. ✅ Copy output để tài liệu
3. ✅ So sánh kết quả giữa các lần chạy

### Cho Development
1. ✅ Modify code trong `demo_presentation.py` hoặc `demo_web.py`
2. ✅ Adjust số lượng sample customers (default: 1000)
3. ✅ Thay đổi số features hiển thị
4. ✅ Custom colors/themes trong HTML template

---

## 📈 Model Performance Summary

```
Model: LightGBM (Tuned Hyperparameters)
Training Data: 168M samples
Features: 13 features
Test Set: 644,970 customers

Sample Results:
  • Best Case:    60% precision
  • Worst Case:    0% precision  
  • Random Case:   0-40% precision
  
Top Feature: purchase_frequency (31.6%)
```

---

## 🎓 Use Cases

| Scenario | Recommended Demo | Reason |
|----------|-----------------|--------|
| Client Meeting | Web Demo 🌐 | Professional, visual |
| Quick Testing | Console Demo 💻 | Fast, simple |
| Documentation | Console Demo 💻 | Easy to copy |
| Presentation | Web Demo 🌐 | Interactive, impressive |
| Development | Console Demo 💻 | Quick iteration |
| Live Demo | Web Demo 🌐 | Real-time, engaging |

---

## ⚙️ Technical Details

### Console Demo
- **Language:** Python 3.11
- **Dependencies:** polars, pickle, numpy
- **Output:** Terminal text with colors/emojis
- **Time:** ~30 seconds

### Web Demo
- **Framework:** Flask 3.0+
- **Frontend:** HTML5, CSS3, Chart.js
- **API:** REST JSON endpoints
- **Port:** 5000 (configurable)
- **Time:** ~30 seconds + web interface

---

## 📞 Support

Nếu có vấn đề:
1. Kiểm tra QUICKSTART.txt
2. Đọc DEMO_GUIDE.md
3. Xem Troubleshooting section ở trên
4. Check terminal output for errors

---

**Created by:** Product Recommendation Team  
**Last Updated:** December 24, 2025  
**Version:** 1.0

🎯 **Ready to demo!** Chọn phiên bản phù hợp và bắt đầu presentation! ✨
