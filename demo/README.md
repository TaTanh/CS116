# 🎯 Demo Files

Demo system cho Product Recommendation với 2 phiên bản: Console và Web Interface.

## 📁 Files

### Scripts
- **demo_presentation.py** - Console demo (terminal output)
- **demo_web.py** - Flask web server
- **run_console_demo.bat** - Shortcut chạy console demo
- **run_web_demo.bat** - Shortcut chạy web demo

### Templates
- **templates/demo.html** - Web UI với interactive features

## 🚀 Cách chạy

### Console Demo
```bash
# Double-click:
run_console_demo.bat

# Hoặc:
cd demo
python demo_presentation.py
```

### Web Demo
```bash
# Double-click:
run_web_demo.bat

# Hoặc:
cd demo
python demo_web.py
# Mở: http://localhost:5000
```

## ✨ Features

- ⭐ **Best Case** - Khách hàng dễ dự đoán nhất
- ⚠️ **Worst Case** - Khách hàng khó dự đoán nhất
- 🎲 **Random Case** - Khách hàng ngẫu nhiên (có thể refresh!)
- 📊 **Feature Importance** - Chart interactive
- 🔄 **Interactive Refresh** - Click để lấy random customer mới

Xem [../docs/README_DEMO.md](../docs/README_DEMO.md) để biết thêm chi tiết.
