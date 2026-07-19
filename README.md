# Lab 4 — Dự đoán giá nhà bằng MLP Regression (PyTorch)

Bài lab xây dựng mô hình **Multi-Layer Perceptron (MLP)** bằng PyTorch để dự đoán giá nhà trên bộ dữ liệu **California Housing**, so sánh với các mô hình hồi quy truyền thống, và đóng gói thành một mini app Streamlit để trực quan hóa.

## Cấu trúc thư mục

```
lab 4/
├── note copy.ipynb          # Notebook chính (có chú thích) — nguồn tóm tắt pipeline bên dưới
├── note.ipynb                # Bản không chú thích, nội dung tương tự note copy
├── example.py                 # Code tham khảo cũ (Boston dataset + Keras), không dùng trong pipeline chính
└── house_price_mlp_app/       # Mini app Streamlit dựng lại pipeline từ notebook
    ├── app.py                 # Giao diện Streamlit (dashboard nhiều tab)
    ├── model_utils.py          # Load data, train MLP, dự đoán 1 mẫu
    ├── train_model.py          # Script train nhanh ngoài terminal (không cần Streamlit)
    ├── requirements.txt
    └── README.md               # README riêng của app (chi tiết hơn cho phần chạy app)
```

> **Lưu ý:** `note copy.ipynb` là file chính vì có đầy đủ chú thích (comment + nhận xét markdown), nội dung phân tích dưới đây dựa theo file này. `note.ipynb` có cùng logic nhưng không có chú thích.

## Tóm tắt pipeline (theo `note copy.ipynb`)

1. **Import thư viện**: numpy, pandas, matplotlib, scikit-learn (preprocessing, metrics, các model hồi quy) và PyTorch (`torch`, `torch.nn`, `DataLoader`).
2. **Load dữ liệu**: dùng `fetch_california_housing()` từ scikit-learn, gồm 8 đặc trưng đầu vào (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude) và 1 biến mục tiêu (giá nhà). Dữ liệu được gộp vào một DataFrame để quan sát.
3. **Khám phá dữ liệu (EDA sơ bộ)**: xem shape, `df.info()`, `df.describe()` để kiểm tra kiểu dữ liệu, thang đo và giá trị thiếu.
4. **Chia tập train/test**: `train_test_split` với `test_size=0.2`, `random_state=42`.
5. **Chuẩn hóa dữ liệu**: dùng `StandardScaler`, fit trên tập train rồi transform cả train lẫn test — cần thiết vì các cột có thang đo rất khác nhau (ví dụ Population hàng chục nghìn so với MedInc chỉ vài đơn vị).
6. **Chuyển sang Tensor**: `X_train`, `X_test`, `y_train`, `y_test` được convert sang `torch.tensor` (float32), sau đó đóng gói bằng `TensorDataset` + `DataLoader` (batch_size=32, shuffle=True) để train theo batch.
7. **Định nghĩa mô hình MLP** (`MLPRegressor`, kế thừa `nn.Module`):
   ```
   Input(8) → Linear(8→32) → ReLU → Linear(32→16) → ReLU → Linear(16→1)
   ```
8. **Huấn luyện**: loss = `MSELoss`, optimizer = `Adam` (lr=0.001), train trong 100 epoch. Mỗi batch thực hiện đúng vòng lặp chuẩn của PyTorch: forward → tính loss → `zero_grad()` → `backward()` → `optimizer.step()`. Loss trung bình mỗi epoch được lưu lại để vẽ biểu đồ.
9. **Đánh giá trên tập test**: tính **MSE**, **MAE**, **R²**. Kết quả trong notebook: R² ≈ 0.7873, MAE ≈ 0.3471 — mô hình giải thích được ~78.7% biến thiên của giá nhà, sai số ở mức chấp nhận được nhưng vẫn dự đoán thấp ở một số nhà giá cao.
10. **Trực quan hóa**: biểu đồ Training Loss theo epoch, và biểu đồ Actual vs Predicted.
11. **So sánh với các mô hình khác**: Linear Regression, Decision Tree, Random Forest được train trên cùng dữ liệu đã chuẩn hóa, so sánh MSE/MAE/R² với MLP trong một bảng tổng hợp, sắp xếp theo R² giảm dần.
12. **Kết luận**: Random Forest thường ổn định hơn nhờ kết hợp nhiều cây (giảm overfitting so với Decision Tree đơn lẻ); MLP học được quan hệ phi tuyến nhưng hiệu quả phụ thuộc nhiều vào kiến trúc mạng, learning rate, số epoch và cách chuẩn hóa dữ liệu.

## Cách chạy

### 1. Chạy notebook (phân tích gốc)

```bash
pip install numpy pandas matplotlib scikit-learn torch jupyter
jupyter notebook "note copy.ipynb"
```
Chạy tuần tự từ cell đầu đến cuối.

### 2. Chạy mini app Streamlit (`house_price_mlp_app/`)

App này đóng gói lại đúng pipeline trên (logic nằm trong `model_utils.py`) và thêm dashboard trực quan (dataset explorer, EDA, training, evaluation, so sánh model, form dự đoán thử).

```bash
cd house_price_mlp_app
pip install -r requirements.txt
streamlit run app.py
```

- Lần đầu mở app sẽ mất vài giây để train model MLP; kết quả được Streamlit cache lại (`st.cache_resource`).
- Có thể chỉnh `epochs`, `batch_size`, `learning rate` trực tiếp ở sidebar để train lại và so sánh kết quả.
- Tab **🔮 Dự đoán thử** cho phép nhập 8 đặc trưng đầu vào để mô hình MLP dự đoán giá nhà (đơn vị gốc ~100,000 USD).

Ngoài ra có thể train nhanh mô hình không cần giao diện:

```bash
python train_model.py
```

### Kiến trúc MLP dùng chung (notebook & app)

```
Input 8 features → Linear(8, 32) → ReLU → Linear(32, 16) → ReLU → Linear(16, 1)
```

## Ghi chú

- Target của California Housing có đơn vị xấp xỉ 100,000 USD (ví dụ dự đoán ra `2.50` ≈ 250,000 USD).
- `example.py` chỉ là code tham khảo cũ (dùng `load_boston` — dataset đã bị scikit-learn loại bỏ — và Keras), không thuộc pipeline chính của bài lab, không cần chạy.
- App chỉ nhằm mục đích minh họa cho bài lab, không dùng để định giá bất động sản thực tế.
