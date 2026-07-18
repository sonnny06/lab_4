import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from model_utils import train_mlp, predict_one


# ===================== Page config =====================
st.set_page_config(
    page_title="MLP House Price Prediction",
    page_icon="🏠",
    layout="wide",
)


# ===================== CSS =====================
st.markdown(
    """
    <style>
    :root {
        --bg: #0b1120;
        --panel: #111827;
        --panel-2: #172033;
        --navy: #e5e7eb;
        --blue: #60a5fa;
        --teal: #5eead4;
        --text: #f8fafc;
        --muted: #cbd5e1;
        --border: #334155;
        --soft: #1f2937;
        --card: #111827;
        --warning-bg: #451a03;
        --warning-text: #fed7aa;
        --success-bg: #052e2b;
        --success-text: #bbf7d0;
    }

    .stApp {background: var(--bg) !important; color: var(--text) !important;}
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}

    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: inherit;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
        color: var(--text) !important;
    }

    .hero {
        padding: 28px 32px;
        border-radius: 26px;
        background: linear-gradient(135deg, #020617 0%, #1e3a8a 58%, #0f766e 100%);
        color: #ffffff !important;
        margin-bottom: 22px;
        box-shadow: 0 14px 32px rgba(0, 0, 0, 0.35);
        border: 1px solid #334155;
    }
    .hero h1 {font-size: 38px; margin: 0 0 8px 0; color: #ffffff !important;}
    .hero p {font-size: 16px; opacity: 0.96; margin: 0; color: #ffffff !important;}

    .card, .result-card, .explain-card, .step-card {
        background: var(--card) !important;
        color: var(--text) !important;
        border-radius: 18px;
        padding: 18px 20px;
        border: 1px solid var(--border);
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.28);
        margin-bottom: 16px;
        line-height: 1.65;
    }
    .card h2, .card h3, .card h4,
    .result-card h2, .result-card h3,
    .explain-card h3, .step-card h4 {
        color: #ffffff !important;
        margin-top: 0;
    }
    .card p, .result-card p, .explain-card p, .step-card p,
    .card li, .result-card li, .explain-card li, .step-card li,
    .card b, .result-card b, .explain-card b, .step-card b {
        color: var(--text) !important;
    }
    .small-note {font-size: 14px; color: var(--muted) !important; line-height: 1.65;}
    .muted {color: var(--muted) !important;}

    .tag {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: #1e3a8a;
        color: #dbeafe !important;
        font-weight: 800;
        font-size: 13px;
        margin-right: 6px;
        margin-bottom: 8px;
    }
    .tag-green {background: #064e3b; color: #bbf7d0 !important;}
    .tag-orange {background: #7c2d12; color: #fed7aa !important;}

    .pipeline-row {
        display: grid;
        grid-template-columns: repeat(5, minmax(120px, 1fr));
        gap: 12px;
        margin: 12px 0 18px 0;
    }
    .pipe-box {
        background: var(--panel-2) !important;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px 12px;
        text-align: center;
        min-height: 96px;
        color: var(--text) !important;
    }
    .pipe-box .num {
        display: inline-flex;
        width: 28px;
        height: 28px;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        background: #2563eb;
        color: #ffffff !important;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .pipe-box b {display: block; color: #ffffff !important; margin-bottom: 4px;}
    .pipe-box span {display: block; color: var(--muted) !important; font-size: 13px; line-height: 1.35;}

    .metric-explain {
        background: var(--panel-2) !important;
        border-left: 5px solid #60a5fa;
        padding: 14px 16px;
        border-radius: 14px;
        color: var(--text) !important;
        margin-bottom: 12px;
    }
    .metric-explain b {color: #ffffff !important;}

    .warning-box {
        background: var(--warning-bg) !important;
        color: var(--warning-text) !important;
        border: 1px solid #9a3412;
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }
    .warning-box b, .warning-box p {color: var(--warning-text) !important;}
    .result-card {
        background: var(--success-bg) !important;
        border-left: 8px solid #22c55e;
    }

    /* Streamlit built-in widgets */
    [data-testid="stMetric"] {
        background: var(--panel) !important;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.22);
    }
    [data-testid="stMetric"] label,
    [data-testid="stMetric"] div {
        color: var(--text) !important;
    }

    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        background: var(--panel) !important;
        color: var(--text) !important;
        border-radius: 12px 12px 0 0;
        border: 1px solid var(--border);
        padding: 8px 14px;
    }
    .stTabs [aria-selected="true"] {
        background: #1e3a8a !important;
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] {
        background: #020617 !important;
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    div[data-baseweb="input"], div[data-baseweb="select"], textarea {
        background: var(--panel) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }
    input, textarea {
        color: var(--text) !important;
    }

    @media (max-width: 900px) {
        .pipeline-row {grid-template-columns: repeat(2, minmax(120px, 1fr));}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===================== Helper functions =====================
@st.cache_resource(show_spinner="Đang train mô hình MLP bằng PyTorch...")
def cached_train(epochs, batch_size, lr):
    return train_mlp(epochs=epochs, batch_size=batch_size, lr=lr, seed=42)


@st.cache_data(show_spinner="Đang train các mô hình so sánh...")
def compare_models(X_train_scaled, X_test_scaled, y_train, y_test, mse_mlp, mae_mlp, r2_mlp):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    results = []
    fitted_models = {}

    for name, reg_model in models.items():
        reg_model.fit(X_train_scaled, y_train)
        y_pred = reg_model.predict(X_test_scaled)
        fitted_models[name] = reg_model
        results.append({
            "Model": name,
            "MSE": mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
        })

    results.append({
        "Model": "MLP PyTorch",
        "MSE": mse_mlp,
        "MAE": mae_mlp,
        "R2": r2_mlp,
    })

    return pd.DataFrame(results).sort_values(by="R2", ascending=False), fitted_models


def render_metric_cards(metrics, df):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MSE", f"{metrics['MSE']:.4f}")
    c2.metric("MAE", f"{metrics['MAE']:.4f}")
    c3.metric("R² Score", f"{metrics['R2']:.4f}")
    c4.metric("Số dòng dữ liệu", f"{len(df):,}")


def filter_dataframe(df):
    search = st.text_input("🔍 Tìm kiếm trong bảng dữ liệu", placeholder="Ví dụ: 3.87, -119.57, 5.00001...")
    if search.strip():
        mask = df.astype(str).apply(lambda col: col.str.contains(search.strip(), case=False, na=False)).any(axis=1)
        return df[mask]
    return df


# ===================== Hero =====================
st.markdown(
    """
    <div class="hero">
        <h1>🏠 Dự đoán giá nhà bằng MLP Regression</h1>
        <p>Dashboard minh họa pipeline PyTorch: dữ liệu → tiền xử lý → tensor → MLP → huấn luyện → đánh giá → dự đoán thử.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ===================== Sidebar =====================
with st.sidebar:
    st.header("⚙️ Cấu hình mô hình")
    epochs = st.slider("Số epoch", 50, 300, 100, step=50)
    batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
    lr = st.selectbox("Learning rate", [0.0001, 0.0005, 0.001, 0.005], index=2)
    st.caption("Kiến trúc MLP giống notebook: 8 → 32 → 16 → 1.")
    st.info("Lần đầu mở app sẽ mất vài giây để train model. Sau đó Streamlit sẽ cache kết quả.")


# ===================== Load bundle =====================
bundle = cached_train(epochs, batch_size, lr)
model = bundle["model"]
scaler = bundle["scaler"]
df = bundle["df"]
feature_names = bundle["feature_names"]
y_test = bundle["y_test"]
y_pred_mlp = bundle["y_pred_mlp"]
metrics = bundle["metrics"]
train_losses = bundle["train_losses"]

results_df, fitted_models = compare_models(
    bundle["X_train_scaled"], bundle["X_test_scaled"],
    bundle["y_train"], bundle["y_test"],
    metrics["MSE"], metrics["MAE"], metrics["R2"]
)


# ===================== Overview =====================
st.markdown("### 1. Tổng quan kết quả MLP")
render_metric_cards(metrics, df)

st.markdown(
    f"""
    <div class="card">
        <span class="tag">Nhận xét nhanh</span>
        <span class="tag tag-green">Regression</span>
        <span class="tag tag-orange">PyTorch</span>
        <p>
        Mô hình MLP hiện giải thích được khoảng <b>{metrics['R2'] * 100:.2f}%</b> sự biến thiên của giá nhà trên tập test.
        MAE = <b>{metrics['MAE']:.4f}</b>, tức sai số tuyệt đối trung bình ở mức khá ổn cho một bài toán hồi quy giá nhà minh họa.
        </p>
        <p class="small-note">
        Target của California Housing có đơn vị xấp xỉ 100,000 USD. Vì vậy kết quả dự đoán 2.50 có thể hiểu gần đúng là 250,000 USD.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ===================== Tabs =====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📌 Pipeline",
    "📊 Dataset Explorer",
    "🔎 EDA",
    "🧠 Training",
    "📈 Evaluation",
    "⚖️ So sánh model",
    "🔮 Dự đoán thử",
])


# ===================== Tab 1: Pipeline =====================
with tab1:
    st.markdown("### Pipeline của bài toán")
    st.markdown(
        """
        <div class="explain-card">
            <h3>Ý tưởng chính</h3>
            <p>
            App này bám theo notebook của bạn. Mục tiêu không chỉ là dự đoán một con số, mà còn cho người xem thấy rõ
            dữ liệu đi qua những bước nào trước khi được đưa vào mô hình MLP.
            </p>
        </div>
        <div class="pipeline-row">
            <div class="pipe-box"><div class="num">1</div><b>Load Data</b><span>California Housing dataset</span></div>
            <div class="pipe-box"><div class="num">2</div><b>Split</b><span>Train/Test = 80/20</span></div>
            <div class="pipe-box"><div class="num">3</div><b>Scale</b><span>StandardScaler chống lệch thang đo</span></div>
            <div class="pipe-box"><div class="num">4</div><b>Tensor</b><span>Chuyển dữ liệu sang PyTorch Tensor</span></div>
            <div class="pipe-box"><div class="num">5</div><b>Train MLP</b><span>Linear → ReLU → Linear → ReLU → Linear</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="step-card">
                <h4>Vì sao phải chuẩn hóa?</h4>
                <p>
                Các cột như <b>Population</b>, <b>Latitude</b>, <b>MedInc</b> có thang đo rất khác nhau.
                MLP học bằng gradient, nên dữ liệu lệch thang đo có thể làm quá trình học chậm hoặc kém ổn định.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="step-card">
                <h4>Vì sao dùng PyTorch Tensor?</h4>
                <p>
                PyTorch không train trực tiếp trên DataFrame. Dữ liệu phải được chuyển sang <b>Tensor</b>, sau đó đưa vào
                <b>TensorDataset</b> và <b>DataLoader</b> để huấn luyện theo từng batch.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Kiến trúc MLP")
    st.code(
        """Input 8 features
    ↓
Linear(8 → 32)
    ↓
ReLU
    ↓
Linear(32 → 16)
    ↓
ReLU
    ↓
Linear(16 → 1)
    ↓
Predicted House Price""",
        language="text",
    )


# ===================== Tab 2: Dataset Explorer =====================
with tab2:
    st.markdown("### Dataset Explorer - Dữ liệu đầu vào")

    st.markdown(
        """
        <div class="card">
            <h3>Dataset đang dùng</h3>
            <p>
            Bộ dữ liệu California Housing gồm <b>20,640 dòng</b>, <b>8 đặc trưng đầu vào</b> và <b>1 biến mục tiêu</b> là giá nhà.
            Đây là dữ liệu dạng bảng nên cần quan sát thống kê và chuẩn hóa trước khi train MLP.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Features", len(feature_names))
    c4.metric("Target", "1")
    c5.metric("Missing", int(df.isnull().sum().sum()))

    feature_info = pd.DataFrame({
        "Cột": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude", "Target"],
        "Ý nghĩa": [
            "Thu nhập trung bình của khu vực",
            "Tuổi nhà trung bình",
            "Số phòng trung bình",
            "Số phòng ngủ trung bình",
            "Dân số khu vực",
            "Số người cư trú trung bình",
            "Vĩ độ",
            "Kinh độ",
            "Giá trị nhà trung bình cần dự đoán",
        ],
        "Vai trò": ["Input", "Input", "Input", "Input", "Input", "Input", "Input", "Input", "Target"],
    })

    st.markdown("#### Ý nghĩa các cột")
    st.dataframe(feature_info, use_container_width=True, hide_index=True)

    st.markdown("#### Xem toàn bộ dữ liệu")
    filtered_df = filter_dataframe(df)

    left, right = st.columns([1, 1])
    with left:
        show_all = st.checkbox("Hiển thị toàn bộ dữ liệu", value=False)
    with right:
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Tải dữ liệu đang xem dưới dạng CSV",
            data=csv,
            file_name="california_housing_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if show_all:
        display_df = filtered_df
    else:
        num_rows = st.slider("Chọn số dòng muốn hiển thị", min_value=10, max_value=1000, value=100, step=10)
        display_df = filtered_df.head(num_rows)

    st.dataframe(display_df, use_container_width=True, height=560)
    st.caption(f"Đang hiển thị {len(display_df):,} / {len(filtered_df):,} dòng sau khi lọc.")

    st.markdown("#### Thống kê mô tả")
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown(
        """
        <div class="metric-explain">
            <b>Cách đọc bảng describe:</b><br>
            <b>count</b> cho biết số lượng dữ liệu không bị thiếu, <b>mean</b> là trung bình, <b>std</b> là độ lệch chuẩn,
            <b>min/max</b> là giá trị nhỏ nhất/lớn nhất, còn <b>25% - 50% - 75%</b> là các mốc phân vị.
        </div>
        <div class="card">
            <p>
            Các cột đều có <b>count = 20,640</b>, nên dữ liệu không thiếu giá trị. Tuy nhiên, thang đo giữa các cột rất khác nhau:
            <b>Population</b> có thể lên tới hàng chục nghìn, trong khi <b>MedInc</b> chỉ dao động quanh vài đơn vị.
            Đây là lý do app sử dụng <b>StandardScaler</b> trước khi train MLP.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===================== Tab 3: EDA =====================
with tab3:
    st.markdown("### Exploratory Data Analysis")

    st.markdown(
        """
        <div class="card">
            <h3>Mục tiêu của EDA</h3>
            <p>
            Phần này giúp quan sát phân bố dữ liệu, kiểm tra tương quan giữa các biến và phát hiện các giá trị bất thường trước khi huấn luyện mô hình.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        selected_col = st.selectbox("Chọn cột để xem phân bố", df.columns.tolist(), index=df.columns.tolist().index("Target"))
        fig_hist, ax_hist = plt.subplots(figsize=(7, 4.2))
        ax_hist.hist(df[selected_col], bins=35)
        ax_hist.set_title(f"Distribution of {selected_col}")
        ax_hist.set_xlabel(selected_col)
        ax_hist.set_ylabel("Frequency")
        ax_hist.grid(True, alpha=0.3)
        st.pyplot(fig_hist)

    with col_b:
        scatter_col = st.selectbox("Chọn feature để so với Target", feature_names, index=0)
        fig_scatter, ax_scatter = plt.subplots(figsize=(7, 4.2))
        ax_scatter.scatter(df[scatter_col], df["Target"], alpha=0.35)
        ax_scatter.set_title(f"{scatter_col} vs Target")
        ax_scatter.set_xlabel(scatter_col)
        ax_scatter.set_ylabel("Target")
        ax_scatter.grid(True, alpha=0.3)
        st.pyplot(fig_scatter)

    st.markdown("#### Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig_corr, ax_corr = plt.subplots(figsize=(9, 6.5))
    im = ax_corr.imshow(corr)
    ax_corr.set_xticks(range(len(corr.columns)))
    ax_corr.set_yticks(range(len(corr.columns)))
    ax_corr.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax_corr.set_yticklabels(corr.columns)
    fig_corr.colorbar(im, ax=ax_corr)
    ax_corr.set_title("Correlation Matrix")
    st.pyplot(fig_corr)

    st.markdown(
        """
        <div class="metric-explain">
            Heatmap tương quan giúp xem feature nào có quan hệ mạnh/yếu với Target. Nếu một feature có tương quan cao với Target,
            nó có thể đóng vai trò quan trọng trong dự đoán. Tuy nhiên, tương quan tuyến tính không phản ánh toàn bộ quan hệ phi tuyến mà MLP có thể học được.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===================== Tab 4: Training =====================
with tab4:
    st.markdown("### Training MLP")

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown(
            f"""
            <div class="card">
                <h3>Cấu hình huấn luyện</h3>
                <p><b>Epochs:</b> {epochs}</p>
                <p><b>Batch size:</b> {batch_size}</p>
                <p><b>Learning rate:</b> {lr}</p>
                <p><b>Loss function:</b> Mean Squared Error</p>
                <p><b>Optimizer:</b> Adam</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_r:
        st.markdown(
            """
            <div class="card">
                <h3>Ý nghĩa quá trình train</h3>
                <p>
                Mỗi batch sẽ đi qua các bước: forward pass → tính loss → zero_grad → backward → optimizer.step.
                Đây là vòng lặp huấn luyện chuẩn trong PyTorch.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Training Loss")
    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(train_losses)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Over Epochs")
    ax1.grid(True)
    st.pyplot(fig1)

    st.markdown(
        """
        <div class="metric-explain">
            Nếu training loss giảm dần, mô hình đang học được quy luật từ dữ liệu train. Nếu loss đứng yên hoặc dao động mạnh,
            cần xem lại learning rate, kiến trúc mạng hoặc preprocessing.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===================== Tab 5: Evaluation =====================
with tab5:
    st.markdown("### Đánh giá mô hình MLP")
    render_metric_cards(metrics, df)

    st.markdown(
        f"""
        <div class="card">
            <h3>Ý nghĩa chỉ số đánh giá</h3>
            <p><b>MSE = {metrics['MSE']:.4f}</b>: trung bình bình phương sai số. Càng thấp càng tốt.</p>
            <p><b>MAE = {metrics['MAE']:.4f}</b>: sai số tuyệt đối trung bình. Dễ hiểu hơn MSE vì cùng đơn vị với target.</p>
            <p><b>R² = {metrics['R2']:.4f}</b>: mô hình giải thích được khoảng <b>{metrics['R2'] * 100:.2f}%</b> sự biến thiên của giá nhà.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Actual vs Predicted")
        fig2, ax2 = plt.subplots(figsize=(7, 5.5))
        ax2.scatter(y_test, y_pred_mlp, alpha=0.6)
        ax2.set_xlabel("Actual House Prices")
        ax2.set_ylabel("Predicted House Prices")
        ax2.set_title("MLP: Actual vs Predicted")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    with right:
        st.markdown("#### Residual Plot")
        residuals = y_test - y_pred_mlp
        fig_res, ax_res = plt.subplots(figsize=(7, 5.5))
        ax_res.scatter(y_pred_mlp, residuals, alpha=0.6)
        ax_res.axhline(0, linestyle="--")
        ax_res.set_xlabel("Predicted House Prices")
        ax_res.set_ylabel("Residuals")
        ax_res.set_title("Residual Plot")
        ax_res.grid(True, alpha=0.3)
        st.pyplot(fig_res)

    st.markdown(
        """
        <div class="card">
            <h3>Nhận xét biểu đồ</h3>
            <p>
            Biểu đồ Actual vs Predicted cho thấy các điểm có xu hướng đi lên từ trái sang phải, nghĩa là mô hình đã học được xu hướng chính.
            Tuy nhiên, ở vùng giá cao, các điểm phân tán mạnh hơn và có một số mẫu bị dự đoán thấp hơn thực tế.
            </p>
            <p>
            Residual Plot giúp quan sát sai số. Nếu residual phân bố quanh đường 0 thì mô hình tương đối ổn. Nếu xuất hiện pattern rõ ràng,
            mô hình vẫn còn bỏ sót một phần quy luật trong dữ liệu.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===================== Tab 6: Model Comparison =====================
with tab6:
    st.markdown("### So sánh MLP với các mô hình hồi quy khác")

    st.dataframe(results_df, use_container_width=True, hide_index=True)

    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    ax3.bar(results_df["Model"], results_df["R2"])
    ax3.set_ylabel("R² Score")
    ax3.set_title("Model Comparison by R² Score")
    ax3.tick_params(axis="x", rotation=20)
    ax3.grid(axis="y", alpha=0.3)
    st.pyplot(fig3)

    best_model = results_df.iloc[0]["Model"]
    st.success(f"Mô hình tốt nhất theo R² Score là: {best_model}")

    if "Random Forest" in fitted_models:
        rf = fitted_models["Random Forest"]
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": rf.feature_importances_,
        }).sort_values(by="Importance", ascending=False)

        st.markdown("#### Feature Importance của Random Forest")
        st.dataframe(importance_df, use_container_width=True, hide_index=True)

        fig_imp, ax_imp = plt.subplots(figsize=(8, 4.5))
        ax_imp.bar(importance_df["Feature"], importance_df["Importance"])
        ax_imp.set_title("Random Forest Feature Importance")
        ax_imp.set_ylabel("Importance")
        ax_imp.tick_params(axis="x", rotation=20)
        ax_imp.grid(axis="y", alpha=0.3)
        st.pyplot(fig_imp)

    st.markdown(
        """
        <div class="card">
            <h3>Phân tích kết quả so sánh</h3>
            <p><b>Linear Regression</b> thường thấp hơn vì chỉ học quan hệ tuyến tính.</p>
            <p><b>Decision Tree</b> học được quan hệ phi tuyến nhưng một cây đơn lẻ dễ overfit.</p>
            <p><b>Random Forest</b> thường mạnh với dữ liệu dạng bảng vì kết hợp nhiều cây quyết định, giúp giảm overfitting và tăng độ ổn định.</p>
            <p><b>MLP PyTorch</b> có kết quả cạnh tranh tốt, chứng tỏ mạng neural học được xu hướng phi tuyến trong dữ liệu.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===================== Tab 7: Prediction =====================
with tab7:
    st.markdown("### Nhập thông tin để dự đoán giá nhà")
    st.markdown(
        """
        <div class="warning-box">
            <b>Lưu ý:</b> Target của California Housing có đơn vị xấp xỉ 100,000 USD. App này dùng để minh họa bài lab,
            không dùng để định giá bất động sản thực tế.
        </div>
        """,
        unsafe_allow_html=True,
    )

    defaults = df[feature_names].median().to_dict()

    c1, c2 = st.columns(2)
    with c1:
        MedInc = st.number_input("MedInc - Thu nhập trung bình", 0.1, 20.0, float(defaults["MedInc"]), step=0.1)
        HouseAge = st.number_input("HouseAge - Tuổi nhà trung bình", 1.0, 60.0, float(defaults["HouseAge"]), step=1.0)
        AveRooms = st.number_input("AveRooms - Số phòng trung bình", 0.5, 50.0, float(defaults["AveRooms"]), step=0.1)
        AveBedrms = st.number_input("AveBedrms - Số phòng ngủ trung bình", 0.1, 20.0, float(defaults["AveBedrms"]), step=0.1)
    with c2:
        Population = st.number_input("Population - Dân số khu vực", 1.0, 40000.0, float(defaults["Population"]), step=10.0)
        AveOccup = st.number_input("AveOccup - Mật độ cư trú", 0.1, 50.0, float(defaults["AveOccup"]), step=0.1)
        Latitude = st.number_input("Latitude - Vĩ độ", 32.0, 42.5, float(defaults["Latitude"]), step=0.01)
        Longitude = st.number_input("Longitude - Kinh độ", -125.0, -113.0, float(defaults["Longitude"]), step=0.01)

    input_values = [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]

    input_preview = pd.DataFrame([input_values], columns=feature_names)
    st.markdown("#### Input hiện tại")
    st.dataframe(input_preview, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="metric-explain">
            Sau khi bấm dự đoán, input sẽ được chuẩn hóa bằng cùng scaler đã học từ tập train, rồi đưa vào mô hình MLP để sinh ra giá nhà dự đoán.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🔮 Dự đoán giá nhà", use_container_width=True):
        pred = predict_one(model, scaler, input_values)
        pred_usd = pred * 100000

        if pred < 1.5:
            label = "Nhóm giá thấp"
        elif pred < 3.0:
            label = "Nhóm giá trung bình"
        else:
            label = "Nhóm giá cao"

        st.markdown(
            f"""
            <div class="result-card">
                <h2>🏡 Kết quả dự đoán</h2>
                <p>Giá nhà dự đoán theo target gốc: <b>{pred:.4f}</b></p>
                <p>Quy đổi gần đúng: <b>{pred_usd:,.0f} USD</b></p>
                <p>Phân nhóm tham khảo: <b>{label}</b></p>
                <p class="small-note">Đây là kết quả minh họa theo dataset California Housing, không phải giá bất động sản thực tế.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown("---")
st.caption("Built with PyTorch + Streamlit. Model: MLPRegressor 8 → 32 → 16 → 1.")
