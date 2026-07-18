# House Price MLP App

Mini app Streamlit minh họa bài lab dự đoán giá nhà bằng MLP Regression với PyTorch.

## Nội dung app

- Pipeline xử lý dữ liệu
- Khám phá dataset California Housing
- Huấn luyện MLP PyTorch
- Đánh giá bằng MSE, MAE, R²
- Biểu đồ Training Loss
- Biểu đồ Actual vs Predicted
- So sánh với Linear Regression, Decision Tree, Random Forest
- Form nhập input để dự đoán giá nhà

## Cách chạy

```bash
cd house_price_mlp_app
pip install -r requirements.txt
python -m streamlit run app.py
```

## Kiến trúc MLP

```text
Input 8 features → Linear(8, 32) → ReLU → Linear(32, 16) → ReLU → Linear(16, 1)
```

## Ghi chú

Target của California Housing có đơn vị xấp xỉ 100,000 USD. App chỉ dùng để minh họa cho bài lab, không dùng để định giá bất động sản thực tế.
