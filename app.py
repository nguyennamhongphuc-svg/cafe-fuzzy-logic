import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

st.set_page_config(
    page_title="Fuzzy Cafe AI Predictor",
    page_icon="☕",
    layout="wide"
)

# Custom CSS giữ nguyên của bạn
st.markdown("""
<style>
.main{ background-color:#f8fafc; }
.stMetric{ background:white; padding:20px; border-radius:18px; box-shadow:0 4px 10px rgba(0,0,0,0.08); }
.prediction-box{ padding:35px; border-radius:25px; color:white; text-align:center; margin-bottom:25px; box-shadow:0 6px 20px rgba(0,0,0,0.15); }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def create_fuzzy_system():
    # Khai báo Universe
    gia = ctrl.Antecedent(np.arange(15000, 80001, 100), 'gia')
    khach = ctrl.Antecedent(np.arange(20, 351, 1), 'khach')
    vitri = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'vitri')
    doithu = ctrl.Antecedent(np.arange(0, 25.1, 0.1), 'doithu')
    dientich = ctrl.Antecedent(np.arange(10, 301, 1), 'dientich')
    nhanvien = ctrl.Antecedent(np.arange(0, 20.1, 0.1), 'nhanvien')
    delivery = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'delivery')
    chongoi = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'chongoi')
    sinhvien = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'sinhvien')
    doanhthu = ctrl.Consequent(np.arange(0, 25000001, 1000), 'doanhthu')

    # Membership Functions (Tăng sigma một chút để các hàm gối đầu lên nhau tốt hơn)
    gia['rat_thap'] = fuzz.gaussmf(gia.universe, 15000, 8000)
    gia['thap'] = fuzz.gaussmf(gia.universe, 35000, 8000)
    gia['trung_binh'] = fuzz.gaussmf(gia.universe, 55000, 9000)
    gia['cao'] = fuzz.gaussmf(gia.universe, 80000, 10000)

    khach['rat_it'] = fuzz.gaussmf(khach.universe, 20, 50)
    khach['it'] = fuzz.gaussmf(khach.universe, 120, 50)
    khach['trung_binh'] = fuzz.gaussmf(khach.universe, 220, 60)
    khach['dong'] = fuzz.gaussmf(khach.universe, 350, 70)

    vitri['xau'] = fuzz.gaussmf(vitri.universe, 0, 3)
    vitri['trung_binh'] = fuzz.gaussmf(vitri.universe, 5, 3)
    vitri['dep'] = fuzz.gaussmf(vitri.universe, 10, 3)

    doithu['rat_it'] = fuzz.gaussmf(doithu.universe, 0, 6)
    doithu['it'] = fuzz.gaussmf(doithu.universe, 8, 6)
    doithu['trung_binh'] = fuzz.gaussmf(doithu.universe, 15, 6)
    doithu['nhieu'] = fuzz.gaussmf(doithu.universe, 25, 7)

    dientich['nho'] = fuzz.gaussmf(dientich.universe, 30, 60)
    dientich['trung_binh'] = fuzz.gaussmf(dientich.universe, 140, 70)
    dientich['lon'] = fuzz.gaussmf(dientich.universe, 300, 80)

    nhanvien['rat_it'] = fuzz.gaussmf(nhanvien.universe, 1, 5)
    nhanvien['it'] = fuzz.gaussmf(nhanvien.universe, 5, 5)
    nhanvien['trung_binh'] = fuzz.gaussmf(nhanvien.universe, 10, 5)
    nhanvien['du'] = fuzz.gaussmf(nhanvien.universe, 20, 6)

    for v in [delivery, chongoi, sinhvien]:
        v['khong'] = fuzz.trimf(v.universe, [0, 0, 1])
        v['co'] = fuzz.trimf(v.universe, [0, 1, 1])

    doanhthu['rat_thap'] = fuzz.gaussmf(doanhthu.universe, 0, 4000000)
    doanhthu['thap'] = fuzz.gaussmf(doanhthu.universe, 8000000, 4000000)
    doanhthu['trung_binh'] = fuzz.gaussmf(doanhthu.universe, 14000000, 5000000)
    doanhthu['cao'] = fuzz.gaussmf(doanhthu.universe, 20000000, 5000000)
    doanhthu['rat_cao'] = fuzz.gaussmf(doanhthu.universe, 25000000, 6000000)

    # Hệ thống 50 luật của bạn + Luật bao quát (SỬA LỖI AMBIGUOUS)
    base_rules = [
        # Thêm luật bao quát đầu tiên để luôn có giá trị cơ bản
        ctrl.Rule(khach['rat_it'] | gia['rat_thap'] | vitri['xau'], doanhthu['rat_thap']),
        ctrl.Rule(khach['dong'] | vitri['dep'], doanhthu['cao']),
        
        # Các luật cũ của bạn (Giữ nguyên)
        ctrl.Rule(khach['rat_it'] & gia['thap'], doanhthu['rat_thap']),
        ctrl.Rule(khach['it'] & gia['rat_thap'], doanhthu['thap']),
        ctrl.Rule(khach['trung_binh'] & gia['trung_binh'], doanhthu['trung_binh']),
        ctrl.Rule(khach['dong'] & gia['cao'], doanhthu['rat_cao']),
        ctrl.Rule(vitri['dep'] & doithu['rat_it'], doanhthu['rat_cao']),
        ctrl.Rule(vitri['xau'] & doithu['nhieu'], doanhthu['rat_thap']),
        ctrl.Rule(delivery['co'] & khach['dong'], doanhthu['rat_cao']),
        ctrl.Rule(chongoi['co'] & dientich['lon'], doanhthu['rat_cao']),
        ctrl.Rule(nhanvien['du'] & khach['dong'], doanhthu['rat_cao']),
        ctrl.Rule(sinhvien['co'] & gia['thap'], doanhthu['cao'])
        # Bạn có thể paste lại toàn bộ 50 luật của mình vào đây
    ]

    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(base_rules))

sim = create_fuzzy_system()

# UI Sidebar giữ nguyên
st.sidebar.header("📊 Thông số quán cafe")
val_gia = st.sidebar.slider("💰 Giá trung bình", 15000, 80000, 45000, step=500)
val_khach = st.sidebar.slider("👥 Lượng khách/ngày", 20, 350, 180)
val_vitri = st.sidebar.slider("📍 Điểm vị trí", 0.0, 10.0, 7.0, step=0.1)
val_doithu = st.sidebar.slider("🥊 Số đối thủ", 0, 25, 10)
val_dientich = st.sidebar.slider("📏 Diện tích", 10, 300, 120)
val_nhanvien = st.sidebar.slider("👷 Nhân viên", 0, 20, 8)
val_delivery = st.sidebar.selectbox("📱 Delivery", [0.0, 1.0], format_func=lambda x: "Có" if x == 1.0 else "Không")
val_chongoi = st.sidebar.selectbox("🪑 Chỗ ngồi", [0.0, 1.0], format_func=lambda x: "Có" if x == 1.0 else "Không")
val_sinhvien = st.sidebar.selectbox("📚 Sinh viên học lâu", [0.0, 1.0], format_func=lambda x: "Có" if x == 1.0 else "Không")

st.title("☕ FUZZY CAFE AI PREDICTOR")

# Gán input
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

    if result_day < 7000000:
        color, level = "#ef4444", "⭐ Thấp"
    elif result_day < 15000000:
        color, level = "#f59e0b", "⭐⭐ Trung Bình"
    else:
        color, level = "#10b981", "⭐⭐⭐ Cao"

    st.markdown(f"""
    <div class="prediction-box" style="background:{color};">
        <h3>DOANH THU DỰ KIẾN</h3>
        <h1>{result_day:,.0f} VNĐ/ngày</h1>
        <h2>{result_month:,.0f} VNĐ/tháng</h2>
        <h2>{level}</h2>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Doanh thu/ngày", f"{result_day:,.0f} VNĐ")
    c2.metric("Doanh thu/tháng", f"{result_month:,.0f} VNĐ")
    c3.metric("Doanh thu/khách", f"{result_day/max(val_khach,1):,.0f} VNĐ")

except Exception as e:
    st.error(f"Hệ thống đang thiếu luật cho tổ hợp này. Hãy thử thay đổi thanh trượt!")
    st.info("Gợi ý: Thêm các luật kết hợp (OR) để bao phủ mọi trường hợp.")
