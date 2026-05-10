import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

st.set_page_config(
    page_title="Fuzzy Cafe AI Predictor",
    page_icon="☕",
    layout="wide"
)

st.markdown("""
<style>
.main{
    background-color:#f8fafc;
}
.stMetric{
    background:white;
    padding:20px;
    border-radius:18px;
    box-shadow:0 4px 10px rgba(0,0,0,0.08);
}
.prediction-box{
    padding:35px;
    border-radius:25px;
    color:white;
    text-align:center;
    margin-bottom:25px;
    box-shadow:0 6px 20px rgba(0,0,0,0.15);
}
.sidebar .sidebar-content{
    background:#ffffff;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def create_fuzzy_system():

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

    gia['rat_thap'] = fuzz.gaussmf(gia.universe, 15000, 5000)
    gia['thap'] = fuzz.gaussmf(gia.universe, 35000, 6000)
    gia['trung_binh'] = fuzz.gaussmf(gia.universe, 55000, 7000)
    gia['cao'] = fuzz.gaussmf(gia.universe, 70000, 6000)

    khach['rat_it'] = fuzz.gaussmf(khach.universe, 20, 30)
    khach['it'] = fuzz.gaussmf(khach.universe, 120, 40)
    khach['trung_binh'] = fuzz.gaussmf(khach.universe, 220, 45)
    khach['dong'] = fuzz.gaussmf(khach.universe, 350, 50)

    vitri['xau'] = fuzz.gaussmf(vitri.universe, 0, 1.5)
    vitri['trung_binh'] = fuzz.gaussmf(vitri.universe, 5, 1.8)
    vitri['dep'] = fuzz.gaussmf(vitri.universe, 10, 1.5)

    doithu['rat_it'] = fuzz.gaussmf(doithu.universe, 0, 2)
    doithu['it'] = fuzz.gaussmf(doithu.universe, 8, 2.5)
    doithu['trung_binh'] = fuzz.gaussmf(doithu.universe, 15, 2.5)
    doithu['nhieu'] = fuzz.gaussmf(doithu.universe, 25, 3)

    dientich['nho'] = fuzz.gaussmf(dientich.universe, 30, 35)
    dientich['trung_binh'] = fuzz.gaussmf(dientich.universe, 140, 40)
    dientich['lon'] = fuzz.gaussmf(dientich.universe, 300, 50)

    nhanvien['rat_it'] = fuzz.gaussmf(nhanvien.universe, 1, 2)
    nhanvien['it'] = fuzz.gaussmf(nhanvien.universe, 5, 2)
    nhanvien['trung_binh'] = fuzz.gaussmf(nhanvien.universe, 10, 2)
    nhanvien['du'] = fuzz.gaussmf(nhanvien.universe, 18, 3)

    for v in [delivery, chongoi, sinhvien]:
        v['khong'] = fuzz.trimf(v.universe, [0, 0, 1])
        v['co'] = fuzz.trimf(v.universe, [0, 1, 1])

    doanhthu['rat_thap'] = fuzz.gaussmf(doanhthu.universe, 2000000, 2500000)
    doanhthu['thap'] = fuzz.gaussmf(doanhthu.universe, 8000000, 3000000)
    doanhthu['trung_binh'] = fuzz.gaussmf(doanhthu.universe, 14000000, 3500000)
    doanhthu['cao'] = fuzz.gaussmf(doanhthu.universe, 20000000, 3500000)
    doanhthu['rat_cao'] = fuzz.gaussmf(doanhthu.universe, 25000000, 4000000)

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
        ctrl.Rule(nhanvien['du'] & dientich['lon'], doanhthu['cao']),

        ctrl.Rule(dientich['nho'] & khach['dong'], doanhthu['trung_binh']),
        ctrl.Rule(dientich['lon'] & khach['dong'], doanhthu['rat_cao']),
        ctrl.Rule(dientich['nho'] & chongoi['khong'], doanhthu['thap'])

    ]

    system = ctrl.ControlSystem(rules)
    return system

system = create_fuzzy_system()
sim = ctrl.ControlSystemSimulation(system)

st.sidebar.header("📊 Thông số quán cafe")

val_gia = st.sidebar.slider("💰 Giá trung bình", 15000, 80000, 45000, step=500)
val_khach = st.sidebar.slider("👥 Lượng khách/ngày", 20, 350, 180)
val_vitri = st.sidebar.slider("📍 Điểm vị trí", 0.0, 10.0, 7.0, step=0.1)
val_doithu = st.sidebar.slider("🥊 Số đối thủ", 0, 25, 10)
val_dientich = st.sidebar.slider("📏 Diện tích", 10, 300, 120)
val_nhanvien = st.sidebar.slider("👷 Nhân viên", 0, 20, 8)

val_delivery = st.sidebar.selectbox(
    "📱 Delivery",
    [0.0, 1.0],
    format_func=lambda x: "Có" if x == 1.0 else "Không"
)

val_chongoi = st.sidebar.selectbox(
    "🪑 Chỗ ngồi",
    [0.0, 1.0],
    format_func=lambda x: "Có" if x == 1.0 else "Không"
)

val_sinhvien = st.sidebar.selectbox(
    "📚 Sinh viên học lâu",
    [0.0, 1.0],
    format_func=lambda x: "Có" if x == 1.0 else "Không"
)

st.title("☕ FUZZY CAFE AI PREDICTOR")
st.markdown("### Dự đoán doanh thu quán cafe bằng Fuzzy Logic")

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
        color = "#ef4444"
        level = "⭐ Thấp"
    elif result_day < 15000000:
        color = "#f59e0b"
        level = "⭐⭐ Trung Bình"
    else:
        color = "#10b981"
        level = "⭐⭐⭐ Cao"

    st.markdown(f"""
    <div class="prediction-box" style="background:{color};">
        <h3>DOANH THU DỰ KIẾN</h3>
        <h1>{result_day:,.0f} VNĐ/ngày</h1>
        <h2>{result_month:,.0f} VNĐ/tháng</h2>
        <h2>{level}</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Doanh thu/ngày",
            f"{result_day:,.0f} VNĐ"
        )

    with col2:
        st.metric(
            "Doanh thu/tháng",
            f"{result_month:,.0f} VNĐ"
        )

    with col3:
        st.metric(
            "Doanh thu/khách",
            f"{result_day/max(val_khach,1):,.0f} VNĐ"
        )

except Exception as e:
    st.error(f"Lỗi hệ thống fuzzy: {e}")
