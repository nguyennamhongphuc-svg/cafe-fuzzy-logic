import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Cấu hình trang
st.set_page_config(page_title="Fuzzy Cafe AI", page_icon="☕", layout="wide")

st.markdown("""
<style>
.main {background-color: #f8fafc;}
.stMetric {background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
.prediction-box {padding: 30px; border-radius: 20px; color: white; text-align: center;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def create_fuzzy_system():
    """Tạo hệ fuzzy hoàn chỉnh"""
    # ✅ Antecedents với universe hợp lý
    gia = ctrl.Antecedent(np.arange(15000, 80001, 500), 'gia')
    khach = ctrl.Antecedent(np.arange(20, 351, 2), 'khach')
    vitri = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'vitri')
    doithu = ctrl.Antecedent(np.arange(0, 25.1, 0.3), 'doithu')
    dientich = ctrl.Antecedent(np.arange(10, 301, 2), 'dientich')
    nhanvien = ctrl.Antecedent(np.arange(0, 20.1, 0.2), 'nhanvien')
    delivery = ctrl.Antecedent(np.arange(0, 1.01, 0.02), 'delivery')
    chongoi = ctrl.Antecedent(np.arange(0, 1.01, 0.02), 'chongoi')
    sinhvien = ctrl.Antecedent(np.arange(0, 1.01, 0.02), 'sinhvien')

    doanhthu = ctrl.Consequent(np.arange(0, 25000001, 25000), 'doanhthu')

    # ✅ TRIMMF thay vì gaussmf (ổn định hơn)
    gia['rat_thap'] = fuzz.trimf(gia.universe, [15000, 18000, 31000])
    gia['thap'] = fuzz.trimf(gia.universe, [25000, 40000, 50000])
    gia['trung_binh'] = fuzz.trapmf(gia.universe, [40000, 45500, 57250, 65000])
    gia['cao'] = fuzz.trimf(gia.universe, [55000, 65000, 74000])

    khach['rat_it'] = fuzz.trimf(khach.universe, [20, 33, 91])
    khach['it'] = fuzz.trimf(khach.universe, [70, 140, 200])
    khach['trung_binh'] = fuzz.trapmf(khach.universe, [165, 220, 252, 280])
    khach['dong'] = fuzz.trimf(khach.universe, [230, 290, 318])

    vitri['xau'] = fuzz.trimf(vitri.universe, [0, 2, 4])
    vitri['trung_binh'] = fuzz.trapmf(vitri.universe, [3.5, 6, 8, 9])
    vitri['dep'] = fuzz.trimf(vitri.universe, [8, 9.5, 10])

    doithu['rat_it'] = fuzz.trimf(doithu.universe, [0, 3, 7])
    doithu['it'] = fuzz.trimf(doithu.universe, [6, 10, 13])
    doithu['trung_binh'] = fuzz.trapmf(doithu.universe, [11, 15, 17, 19])
    doithu['nhieu'] = fuzz.trimf(doithu.universe, [18, 20, 22])

    dientich['rat_nho'] = fuzz.trimf(dientich.universe, [10, 20, 60])
    dientich['nho'] = fuzz.trimf(dientich.universe, [50, 80, 120])
    dientich['trung_binh'] = fuzz.trapmf(dientich.universe, [110, 131, 170, 190])
    dientich['lon'] = fuzz.trimf(dientich.universe, [180, 220, 245])

    nhanvien['rat_it'] = fuzz.trimf(nhanvien.universe, [0, 2, 5])
    nhanvien['it'] = fuzz.trimf(nhanvien.universe, [4, 7, 9])
    nhanvien['trung_binh'] = fuzz.trapmf(nhanvien.universe, [8, 9, 11, 12])
    nhanvien['du'] = fuzz.trimf(nhanvien.universe, [11.5, 14, 15])

    for v in [delivery, chongoi, sinhvien]:
        v['khong'] = fuzz.trimf(v.universe, [0, 0, 0.4])
        v['co'] = fuzz.trimf(v.universe, [0.6, 1, 1])

    doanhthu['rat_thap'] = fuzz.trimf(doanhthu.universe, [0, 1230000, 3700000])
    doanhthu['thap'] = fuzz.trimf(doanhthu.universe, [3200000, 6020000, 8000000])
    doanhthu['trung_binh'] = fuzz.trapmf(doanhthu.universe, [7000000, 10180000, 12000000, 15000000])
    doanhthu['cao'] = fuzz.trimf(doanhthu.universe, [14000000, 18000000, 22000000])
    doanhthu['rat_cao'] = fuzz.trimf(doanhthu.universe, [20000000, 22000000, 25000000])

    # ✅ 50 luật đầy đủ
    rules = [
        ctrl.Rule(khach['rat_it'] & gia['rat_thap'], doanhthu['rat_thap']),
        ctrl.Rule(khach['rat_it'] & gia['thap'], doanhthu['rat_thap']),
        ctrl.Rule(khach['rat_it'] & gia['trung_binh'], doanhthu['rat_thap']),
        ctrl.Rule(khach['rat_it'] & gia['cao'], doanhthu['rat_thap']),
        ctrl.Rule(khach['it'] & gia['rat_thap'], doanhthu['thap']),
        ctrl.Rule(khach['it'] & gia['thap'], doanhthu['thap']),
        ctrl.Rule(khach['it'] & gia['trung_binh'], doanhthu['thap']),
        ctrl.Rule(khach['it'] & gia['cao'], doanhthu['thap']),
        ctrl.Rule(khach['trung_binh'] & gia['rat_thap'], doanhthu['trung_binh']),
        ctrl.Rule(khach['trung_binh'] & gia['thap'], doanhthu['trung_binh']),
        ctrl.Rule(khach['trung_binh'] & gia['trung_binh'], doanhthu['trung_binh']),
        ctrl.Rule(khach['trung_binh'] & gia['cao'], doanhthu['cao']),
        ctrl.Rule(khach['dong'] & gia['rat_thap'], doanhthu['cao']),
        ctrl.Rule(khach['dong'] & gia['thap'], doanhthu['cao']),
        ctrl.Rule(khach['dong'] & gia['trung_binh'], doanhthu['cao']),
        ctrl.Rule(khach['dong'] & gia['cao'], doanhthu['cao']),
        ctrl.Rule(vitri['dep'] & doithu['rat_it'], doanhthu['rat_cao']),
        ctrl.Rule(vitri['dep'] & doithu['it'], doanhthu['cao']),
        ctrl.Rule(vitri['dep'] & doithu['trung_binh'], doanhthu['cao']),
        ctrl.Rule(vitri['dep'] & doithu['nhieu'], doanhthu['trung_binh']),
        ctrl.Rule(vitri['trung_binh'] & doithu['rat_it'], doanhthu['cao']),
        ctrl.Rule(vitri['trung_binh'] & doithu['it'], doanhthu['trung_binh']),
        ctrl
