import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# =========================================================
# CẤU HÌNH TRANG & GIAO DIỆN (UI)
# =========================================================
st.set_page_config(
    page_title="Fuzzy Cafe AI Predictor",
    page_icon="☕",
    layout="wide"
)

# Custom CSS để làm đẹp giao diện
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .prediction-box {
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# KHỞI TẠO HỆ THỐNG MỜ (CACHE ĐỂ TĂNG TỐC)
# =========================================================
@st.cache_resource
def init_fuzzy_system():
    # Antecedents
    gia = ctrl.Antecedent(np.arange(15000, 80001, 100), 'gia')
    khach = ctrl.Antecedent(np.arange(20, 351, 1), 'khach')
    vitri = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'vitri')
    doithu = ctrl.Antecedent(np.arange(0, 25.1, 0.1), 'doithu')
    dientich = ctrl.Antecedent(np.arange(10, 301, 1), 'dientich')
    nhanvien = ctrl.Antecedent(np.arange(0, 20.1, 0.1), 'nhanvien')
    delivery = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'delivery')
    chongoi = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'chongoi')
    sinhvien = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'sinhvien')

    # Consequent
    doanhthu = ctrl.Consequent(np.arange(0, 25000001, 1000), 'doanhthu')

    # Membership Functions (Sử dụng Gaussmf để đảm bảo độ nhạy theo yêu cầu trước đó)
    gia['rat_thap'] = fuzz.gaussmf(gia.universe, 15000, 7000)
    gia['thap'] = fuzz.gaussmf(gia.universe, 35000, 7000)
    gia['trung_binh'] = fuzz.gaussmf(gia.universe, 55000, 8000)
    gia['cao'] = fuzz.gaussmf(gia.universe, 80000, 10000)

    khach['rat_it'] = fuzz.gaussmf(khach.universe, 20, 50)
    khach['it'] = fuzz.gaussmf(khach.universe, 120, 50)
    khach['trung_binh'] = fuzz.gaussmf(khach.universe, 220, 60)
    khach['dong'] = fuzz.gaussmf(khach.universe, 350, 70)

    vitri['xau'] = fuzz.gaussmf(vitri.universe, 0, 3)
    vitri['trung_binh'] = fuzz.gaussmf(vitri.universe, 5, 3)
    vitri['dep'] = fuzz.gaussmf(vitri.universe, 10, 3)

    doithu['rat_it'] = fuzz.gaussmf(doithu.universe, 0, 6)
    doithu['it'] = fuzz.gaussmf(doithu.universe, 10, 6)
    doithu['trung_binh'] = fuzz.gaussmf(doithu.universe, 18, 6)
    doithu['nhieu'] = fuzz.gaussmf(doithu.universe, 25, 7)

    dientich['nho'] = fuzz.gaussmf(dientich.universe, 10, 80)
    dientich['trung_binh'] = fuzz.gaussmf(dientich.universe, 150, 80)
    dientich['lon'] = fuzz.gaussmf(dientich.universe, 300, 90)

    nhanvien['it'] = fuzz.gaussmf(nhanvien.universe, 0, 7)
    nhanvien['trung_binh'] = fuzz.gaussmf(nhanvien.universe, 10, 7)
    nhanvien['du'] = fuzz.gaussmf(nhanvien.universe, 20, 8)

    for v in [delivery, chongoi, sinhvien]:
        v['khong'] = fuzz.gaussmf(v.universe, 0, 0.4)
        v['co'] = fuzz.gaussmf(v.universe, 1, 0.4)

    doanhthu['rat_thap'] = fuzz.gaussmf(doanhthu.universe, 0, 5000000)
    doanhthu['thap'] = fuzz.gaussmf(doanhthu.universe, 8000000, 5000000)
    doanhthu['trung_binh'] = fuzz.gaussmf(doanhthu.universe, 16000000, 6000000)
    doanhthu['cao'] = fuzz.gaussmf(doanhthu.universe, 22000000, 6000000)
    doanhthu['rat_cao'] = fuzz.gaussmf(doanhthu.universe, 25000000, 7000000)

    # 50 Rules (Áp dụng bộ luật của bạn)
    rules = [
        ctrl.Rule(khach['rat_it'] & gia['rat_thap'], doanhthu['rat_thap']),
        ctrl.Rule(khach['rat_it'] & gia['thap'], doanhthu['rat_thap']),
        ctrl.Rule(khach['rat_it'] & gia['trung_binh'], doanhthu['rat_thap']),
        ctrl.Rule(khach['rat_it'] & gia['cao'], doanhthu['thap']),
        ctrl.Rule(khach['it'] & gia['rat_thap'], doanhthu['thap']),
        ctrl.Rule(khach['it'] & gia['thap'], doanhthu['thap']),
        ctrl.Rule(khach['it'] & gia['trung_binh'], doanhthu['thap']),
        ctrl.Rule(khach['it'] & gia['cao'], doanhthu['trung_binh']),
        ctrl.Rule(khach['trung_binh'] & gia['rat_thap'], doanhthu['trung_binh']),
        ctrl.Rule(khach['trung_binh'] & gia['thap'], doanhthu['trung_binh']),
        ctrl.Rule(khach['trung_binh'] & gia['trung_binh'], doanhthu['trung_binh']),
        ctrl.Rule(khach['trung_binh'] & gia['cao'], doanhthu['cao']),
        ctrl.Rule(khach['dong'] & gia['rat_thap'], doanhthu['cao']),
        ctrl.Rule(khach['dong'] & gia['thap'], doanhthu['cao']),
        ctrl.Rule(khach['dong'] & gia['trung_binh'], doanhthu['cao']),
        ctrl.Rule(khach['dong'] & gia['cao'], doanhthu['rat_cao']),
        ctrl.Rule(vitri['dep'] & doithu['rat_it'], doanhthu['rat_cao']),
        ctrl.Rule(vitri['dep'] & doithu['it'], doanhthu['cao']),
        ctrl.Rule(vitri['dep'] & doithu['trung_binh'], doanhthu['cao']),
        ctrl.Rule(vitri['dep'] & doithu['nhieu'], doanhthu['trung_binh']),
        ctrl.Rule(vitri['trung_binh'] & doithu['rat_it'], doanhthu['cao']),
        ctrl.Rule(vitri['trung_binh'] & doithu['it'], doanhthu['trung_binh']),
        ctrl.Rule(vitri['trung_binh'] & doithu['trung_binh'], doanhthu['trung_binh']),
        ctrl.Rule(vitri['trung_binh'] & doithu['nhieu'], doanhthu['thap']),
        ctrl.Rule(vitri['xau'] & doithu['rat_it'], doanhthu['trung_binh']),
        ctrl.Rule(vitri['xau'] & doithu['it'], doanhthu['thap']),
        ctrl.Rule(vitri['xau'] & doithu['trung_binh'], doanhthu['thap']),
        ctrl.Rule(vitri['xau'] & doithu['nhieu'], doanhthu['rat_thap']),
        ctrl.Rule(delivery['co'] & khach['dong'], doanhthu['rat_cao']),
        ctrl.Rule(delivery['co'] & khach['trung_binh'], doanhthu['cao']),
        ctrl.Rule(delivery['co'] & vitri['xau'], doanhthu['trung_binh']),
        ctrl.Rule(chongoi['co'] & dientich['lon'], doanhthu['rat_cao']),
        ctrl.Rule(chongoi['co'] & khach['trung_binh'], doanhthu['cao']),
        ctrl.Rule(chongoi['khong'] & khach['dong'], doanhthu['thap']),
        ctrl.Rule(nhanvien['rat_it'] & khach['dong'], doanhthu['thap']),
        ctrl.Rule(nhanvien['it'] & khach['trung_binh'], doanhthu['trung_binh']),
        ctrl.Rule(nhanvien['trung_binh'] & khach['dong'], doanhthu['cao']),
        ctrl.Rule(nhanvien['du'] & khach['dong'], doanhthu['rat_cao']),
        ctrl.Rule(nhanvien['rat_it'] & khach['rat_it'], doanhthu['rat_thap']),
        ctrl.Rule(nhanvien['du'] & khach['rat_it'], doanhthu['thap']),
        ctrl.Rule(sinhvien['co'] & gia['thap'], doanhthu['cao']),
        ctrl.Rule(sinhvien['co'] & gia['cao'], doanhthu['trung_binh']),
        ctrl.Rule(sinhvien['khong'] & gia['cao'] & vitri['dep'], doanhthu['rat_cao']),
        ctrl.Rule(sinhvien['co'] & chongoi['co'], doanhthu['cao']),
        ctrl.Rule(khach['dong'] & vitri['dep'] & delivery['co'] & doithu['rat_it'], doanhthu['rat_cao']),
        ctrl.Rule(khach['rat_it'] & vitri['xau'] & doithu['nhieu'], doanhthu['rat_thap']),
        ctrl.Rule(delivery['co'] & dientich['lon'], doanhthu['cao']),
        ctrl.Rule(khach['trung_binh'] & gia['trung_binh'] & doithu['it'], doanhthu['cao']),
        ctrl.Rule(vitri['trung_binh'] & gia['thap'], doanhthu['trung_binh']),
        ctrl.Rule(nhanvien['du'] & dientich['lon'], doanhthu['cao'])
    ]

    cafe_system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(cafe_system)

sim = init_fuzzy_system()

# =========================================================
# XÂY DỰNG SIDEBAR (NHẬP LIỆU)
# =========================================================
st.sidebar.header("📊 Thông số cửa hàng")
st.sidebar.markdown("Điều chỉnh các thanh trượt để cập nhật kết quả tức thì.")

val_gia = st.sidebar.slider("💰 Giá trung bình (VNĐ)", 15000, 80000, 45000, step=500)
val_khach = st.sidebar.slider("👥 Lượng khách/ngày", 20, 350, 180, step=1)
val_vitri = st.sidebar.slider("📍 Điểm vị trí (0-10)", 0.0, 10.0, 7.0, step=0.1)
val_doithu = st.sidebar.slider("🥊 Số đối thủ cạnh tranh", 0, 25, 10, step=1)
val_dientich = st.sidebar.slider("📏 Diện tích (m2)", 10, 300, 120, step=1)
val_nhanvien = st.sidebar.slider("👷 Số nhân viên", 0, 20, 8, step=1)

st.sidebar.subheader("🌟 Tiện ích & Đối tượng")
val_delivery = st.sidebar.selectbox("📱 Có Giao hàng (Delivery)?", options=[0.0, 1.0], format_func=lambda x: "Có" if x==1.0 else "Không")
val_chongoi = st.sidebar.selectbox("🪑 Có Chỗ ngồi tại quán?", options=[0.0, 1.0], format_func=lambda x: "Có" if x==1.0 else "Không")
val_sinhvien = st.sidebar.selectbox("📚 Có nhiều Sinh viên học lâu?", options=[0.0, 1.0], format_func=lambda x: "Có" if x==1.0 else "Không")

# =========================================================
# TÍNH TOÁN & HIỂN THỊ KẾT QUẢ
# =========================================================
st.title("☕ FUZZY CAFE AI PREDICTOR")
st.markdown("Hệ thống hỗ trợ ra quyết định kinh doanh dựa trên Logic mờ.")

# Gán giá trị vào simulation
sim.input['gia'] = val_gia
sim.input['khach'] = val_khach
sim.input['vitri'] = val_vitri
sim.input['doithu'] = val_doithu
sim.input['dientich'] = val_dientich
sim.input['nhanvien'] = val_nhanvien
sim.input['delivery'] = val_delivery
sim.input['chongoi'] = val_chongoi
sim.input['sinhvien'] = val_sinhvien

try:
    sim.compute()
    result_day = sim.output['doanhthu']
    result_month = result_day * 30

    # Xác định màu sắc và cấp độ
    if result_day < 7000000:
        color, level, label = "#ef4444", "Thấp", "⭐"
    elif result_day < 15000000:
        color, level, label = "#f59e0b", "Trung Bình", "⭐⭐"
    else:
        color, level, label = "#10b981", "Cao", "⭐⭐⭐"

    # Hiển thị kết quả chính
    st.markdown(f"""
        <div class="prediction-box" style="background-color: {color};">
            <p style="font-size: 20px; margin-bottom: 0;">DOANH THU DỰ KIẾN</p>
            <h1 style="font-size: 60px; margin: 10px 0;">{result_day:,.0f} VNĐ <small>/ ngày</small></h1>
            <h2 style="font-size: 30px;">{result_month:,.0f} VNĐ <small>/ tháng</small></h2>
            <hr style="border: 0.5px solid rgba(255,255,255,0.3)">
            <p style="font-size: 24px; font-weight: bold;">{label} Mức độ: {level}</p>
        </div>
    """, unsafe_allow_html=True)

    # Hiển thị chi tiết bằng Columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tỷ lệ lấp đầy", f"{min(val_khach/3.5, 100):.1f}%")
    with col2:
        st.metric("Doanh thu TB/Khách", f"{val_gia:,.0f} VNĐ")
    with col3:
        st.metric("Chi phí NV dự tính", f"{(val_nhanvien * 250000):,.0f} VNĐ")

except Exception as e:
    st.error(f"Lỗi tính toán: {e}. Vui lòng kiểm tra lại bộ luật mờ hoặc dải input.")

st.info("💡 **Gợi ý:** Thay đổi 'Vị trí' hoặc 'Lượng khách' ở cột bên trái để thấy sự biến động mạnh mẽ của doanh thu.")
