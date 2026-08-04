"""
FXD2.0.py - 龙华水质净化厂智能预警系统 v7.0
按用户反馈精简并强化诊断
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import joblib
import warnings
import supabase

warnings.filterwarnings('ignore')

# ==========================================
# 北京时间时区
# ==========================================
BEIJING_TZ = timezone(timedelta(hours=8))

st.set_page_config(
    page_title="水质净化厂智能预警与调控决策系统",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Supabase 配置
# ==========================================
SUPABASE_URL = "https://esoulexcrpdeeoumoili.supabase.co"
SUPABASE_KEY = "sb_publishable_m0hz9Rv8NB_ziC5xKCltMg_Ij5Od60Q"
supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# CSS样式（完整复刻旧版）
# ==========================================
st.markdown("""
<style>
    .main-title {
        font-size: 24px;
        font-weight: 700;
        color: #1a3a5c;
        padding: 6px 0 10px 0;
        border-bottom: 3px solid #2E86AB;
        margin-bottom: 14px;
    }
    .section-header {
        font-size: 16px;
        font-weight: 600;
        color: #1a3a5c;
        margin: 14px 0 8px 0;
        padding-left: 8px;
        border-left: 4px solid #2E86AB;
    }
    .status-metric {
        text-align: center;
        background: #F8F9FA;
        border-radius: 8px;
        padding: 6px 8px;
        border: 1px solid #EEF0F2;
    }
    .status-metric .label { font-size: 12px; color: #888; font-weight: 500; }
    .status-metric .value { font-size: 16px; font-weight: 600; color: #1a3a5c; }
    .status-metric .value-normal { color: #1B7A4A; }
    .status-metric .value-critical { color: #C0392B; }
    .water-card-in {
        background: #F5F8FC;
        border-radius: 12px;
        padding: 10px 14px 14px 14px;
        border: 1px solid #D6E4F0;
        margin-bottom: 6px;
    }
    .water-card-out {
        background: #F5FCF8;
        border-radius: 12px;
        padding: 10px 14px 14px 14px;
        border: 1px solid #C8E6D9;
        margin-bottom: 6px;
    }
    .metric-card {
        background: white;
        border-radius: 6px;
        padding: 6px 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        margin-bottom: 3px;
        border-left: 4px solid #2E86AB;
    }
    .metric-card .label { font-size: 12px; color: #666; font-weight: 500; }
    .metric-card .value { font-size: 18px; font-weight: 700; color: #1a3a5c; }
    .metric-card .sub { font-size: 11px; color: #999; }
    .stat-card {
        background: white;
        border-radius: 6px;
        padding: 6px 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        margin-bottom: 3px;
        border-left: 4px solid #2E86AB;
    }
    .stat-card .stat-label { font-size: 12px; color: #666; font-weight: 500; }
    .stat-card .stat-value { font-size: 18px; font-weight: 700; color: #1a3a5c; }
    .stat-card .stat-sub { font-size: 11px; color: #999; }
    .limit-ref {
        font-size: 11px;
        color: #888;
        background: #F0F0F0;
        padding: 1px 8px;
        border-radius: 10px;
        display: inline-block;
    }
    .channel-item {
        flex: 1;
        background: white;
        border-radius: 8px;
        padding: 8px 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
        border-top: 3px solid #ccc;
    }
    .channel-item .ch-name { font-weight: 600; font-size: 13px; }
    .channel-item .ch-value { font-size: 18px; font-weight: 700; margin: 2px 0; }
    .channel-item .ch-desc { font-size: 11px; color: #666; }
    .channel-fast .ch-name { color: #27AE60; }
    .channel-slow .ch-name { color: #F39C12; }
    .channel-special .ch-name { color: #E74C3C; }
    .channel-fast { border-top-color: #27AE60; }
    .channel-slow { border-top-color: #F39C12; }
    .channel-special { border-top-color: #E74C3C; }
    .timeline-step {
        display: flex;
        align-items: center;
        padding: 4px 0;
        border-bottom: 1px solid #F5F5F5;
    }
    .timeline-step:last-child { border-bottom: none; }
    .timeline-time { min-width: 50px; font-weight: 700; font-size: 14px; color: #1a3a5c; }
    .timeline-action { font-size: 14px; color: #333; padding-left: 8px; }
    .calibration-success {
        background: #E8F5E9;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #27AE60;
        margin: 8px 0;
    }
    .calibration-info { font-size: 13px; color: #555; }
    .data-status-realtime {
        background: #E3F2FD;
        border-radius: 12px;
        padding: 8px 16px;
        border: 1px solid #90CAF9;
        display: inline-block;
        font-size: 13px;
        color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 设计标准和记忆长度
# ==========================================
DESIGN_LIMITS = {
    'COD': {'value': 30, 'unit': 'mg/L'},
    'NH3-N': {'value': 1.5, 'unit': 'mg/L'},
    'TP': {'value': 0.3, 'unit': 'mg/L'},
    'TN': {'value': 10, 'unit': 'mg/L'},
    'SS': {'value': 10, 'unit': 'mg/L'}
}

MEMORY = {
    'NH3-N': {'hours': 1, 'channel': '快速', 'freq': '实时'},
    'TP': {'hours': 1, 'channel': '快速', 'freq': '实时'},
    'TN': {'hours': 9, 'channel': '快速', 'freq': '3-4h'},
    'COD': {'hours': None, 'channel': '不适用', 'freq': '—'},
    'SS': {'hours': None, 'channel': '特殊', 'freq': '实时报警'}
}

# ==========================================
# 数据缓存管理
# ==========================================
class DataBuffer:
    def __init__(self, max_hours=48):
        self.max_hours = max_hours
        self.data = []
    
    def add_data(self, timestamp, inlet, outlet, pred_outlet):
        self.data.append({
            'timestamp': timestamp,
            'inlet': inlet.copy(),
            'outlet': outlet.copy() if outlet else None,
            'pred_outlet': pred_outlet.copy() if pred_outlet else None
        })
        cutoff = datetime.now(BEIJING_TZ) - timedelta(hours=self.max_hours)
        self.data = [d for d in self.data if d['timestamp'] >= cutoff]
    
    def get_recent(self, hours=24):
        cutoff = datetime.now(BEIJING_TZ) - timedelta(hours=hours)
        return [d for d in self.data if d['timestamp'] >= cutoff]

# ==========================================
# 加载新模型（去除率模型）
# ==========================================
@st.cache_resource
def load_base_models():
    status_placeholder = st.empty()
    status_placeholder.info("🔄 正在加载预训练模型（去除率版）...")
    try:
        model_xgb = joblib.load('model_cache/xgb_final_model.pkl')
        scaler = joblib.load('model_cache/scaler.pkl')
        with open('model_cache/feature_cols.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        status_placeholder.success("✅ 模型加载成功（去除率版）")
        return model_xgb, scaler, feature_cols
    except Exception as e:
        status_placeholder.error(f"❌ 模型加载失败: {e}")
        st.stop()

model_xgb, scaler, feature_cols = load_base_models()

# ==========================================
# 初始化 session_state
# ==========================================
if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = DataBuffer()
if 'calibration_count' not in st.session_state:
    st.session_state.calibration_count = 0
if 'auto_mode_running' not in st.session_state:
    st.session_state.auto_mode_running = False
if 'simulation_counter' not in st.session_state:
    st.session_state.simulation_counter = 0
if 'feedback_log' not in st.session_state:
    st.session_state.feedback_log = []
if 'has_prediction' not in st.session_state:
    st.session_state.has_prediction = False
if 'pred_result' not in st.session_state:
    st.session_state.pred_result = None
if 'current_inlet' not in st.session_state:
    st.session_state.current_inlet = {'COD': 200, 'NH3-N': 20, 'TP': 3.0, 'TN': 30, 'SS': 150, '流量': 10000}
    st.session_state.current_params = {'PAC': 30, '碳源': 50, 'MLSS': 4000, 'DO': 2.0}

st.markdown('<div class="main-title">🏭 水质净化厂智能预警与调控决策系统</div>', unsafe_allow_html=True)

# ==========================================
# 状态栏
# ==========================================
col_s1, col_s2, col_s3 = st.columns(3)
status_placeholder = col_s1.empty()

with col_s2:
    beijing_now = datetime.now(BEIJING_TZ)
    st.markdown(f"""
    <div class="status-metric">
        <div class="label">⏱️ 当前时间</div>
        <div class="value">{beijing_now.strftime('%H:%M')}</div>
    </div>
    """, unsafe_allow_html=True)

with col_s3:
    st.markdown("""
    <div class="status-metric">
        <div class="label">📋 出水设计标准</div>
        <div class="value">准Ⅳ类</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 预测函数
# ==========================================
def build_feature_vector_for_prediction(nh3, do, mlss, pac, carbon, flow, feature_cols):
    vec = np.zeros((1, len(feature_cols)))
    for i, col in enumerate(feature_cols):
        if 'NH3-N_detrend_lag1' in col:
            vec[0, i] = nh3 * 0.9
        elif 'NH3-N_detrend_lag2' in col:
            vec[0, i] = nh3 * 0.8
        elif 'NH3-N_detrend_lag3' in col:
            vec[0, i] = nh3 * 0.7
        elif '进水流量_lag1' in col:
            vec[0, i] = flow
        elif '溶解氧浓度均值_lag1' in col:
            vec[0, i] = do
        elif '污泥浓度均值_lag1' in col:
            vec[0, i] = mlss
        elif 'PAC_lag1' in col:
            vec[0, i] = pac
        elif '碳源_lag1' in col:
            vec[0, i] = carbon
        elif '风量_lag1' in col:
            vec[0, i] = 50000
    if np.sum(np.abs(vec)) < 0.01:
        vec[0, 0] = nh3
        vec[0, 1] = do
        vec[0, 2] = mlss / 1000
    return vec

def predict_removal_effluent(input_data):
    if input_data is None:
        return None
    try:
        flow = input_data['流量'].values[0] if '流量' in input_data.columns else 10000
        pac = input_data['PAC'].values[0] if 'PAC' in input_data.columns else 30
        carbon = input_data['碳源'].values[0] if '碳源' in input_data.columns else 50
        mlss = input_data['MLSS'].values[0] if 'MLSS' in input_data.columns else 4000
        do = input_data['DO'].values[0] if 'DO' in input_data.columns else 2.0
        
        if 'COD_load' in input_data.columns and flow > 0:
            cod_in = input_data['COD_load'].values[0] * 1000 / flow
        else:
            cod_in = 200
        if 'NH3_load' in input_data.columns and flow > 0:
            nh3_in = input_data['NH3_load'].values[0] * 1000 / flow
        else:
            nh3_in = 20
        if 'TP_load' in input_data.columns and flow > 0:
            tp_in = input_data['TP_load'].values[0] * 1000 / flow
        else:
            tp_in = 3.0
        tn_in = input_data['TN'].values[0] if 'TN' in input_data.columns else 30
        ss_in = input_data['SS'].values[0] if 'SS' in input_data.columns else 150
        
        vec = build_feature_vector_for_prediction(nh3_in, do, mlss, pac, carbon, flow, feature_cols)
        vec_scaled = scaler.transform(vec)
        
        pred_removal = model_xgb.predict(vec_scaled)[0]
        pred_removal = max(0.5, min(0.99, pred_removal))
        
        removal_rates = {'NH3-N': pred_removal, 'COD': 0.93, 'TP': 0.88, 'TN': 0.75, 'SS': 0.92}
        effluent = {
            'COD': cod_in * (1 - removal_rates['COD']),
            'NH3-N': nh3_in * (1 - removal_rates['NH3-N']),
            'TP': tp_in * (1 - removal_rates['TP']),
            'TN': tn_in * (1 - removal_rates['TN']),
            'SS': ss_in * (1 - removal_rates['SS'])
        }
        return {
            'removal': pred_removal,
            'effluent': effluent,
            'inlet': {'COD': cod_in, 'NH3-N': nh3_in, 'TP': tp_in, 'TN': tn_in, 'SS': ss_in, '流量': flow},
            'params': {'PAC': pac, '碳源': carbon, 'MLSS': mlss, 'DO': do}
        }
    except Exception as e:
        return None

def build_input_with_lags(cod, nh3, tp, ss, flow, pac, carbon, mlss, do, tn=30):
    data = pd.DataFrame({
        'COD_load': [cod * flow / 1000],
        'NH3_load': [nh3 * flow / 1000],
        'TP_load': [tp * flow / 1000],
        '流量': [flow],
        'PAC': [pac],
        '碳源': [carbon],
        'MLSS': [mlss],
        'DO': [do],
        'TN': [tn],
        'SS': [ss]
    })
    for i in range(1, 49):
        decay = 1 - (i / 48) * 0.3
        data[f'COD_load_lag{i}'] = cod * flow / 1000 * decay
        data[f'NH3_load_lag{i}'] = nh3 * flow / 1000 * decay
        data[f'TP_load_lag{i}'] = tp * flow / 1000 * decay
        data[f'流量_lag{i}'] = flow * decay
    return data

def generate_simulated_data():
    base_cod = 200 + np.random.normal(0, 30)
    base_nh3 = 20 + np.random.normal(0, 3)
    base_tp = 3.0 + np.random.normal(0, 0.4)
    base_tn = 30 + np.random.normal(0, 5)
    base_ss = 150 + np.random.normal(0, 20)
    base_flow = 10000 + np.random.normal(0, 500)
    return {
        'COD': max(0, base_cod),
        'NH3-N': max(0, base_nh3),
        'TP': max(0, base_tp),
        'TN': max(0, base_tn),
        'SS': max(0, base_ss),
        '流量': max(0, base_flow),
        'PAC': 30 + np.random.normal(0, 2),
        '碳源': 50 + np.random.normal(0, 3),
        'MLSS': 4000 + np.random.normal(0, 200),
        'DO': 2.0 + np.random.normal(0, 0.2)
    }

def simulate_outlet(inlet):
    return {
        'COD': 8 + np.random.normal(0, 0.8) + inlet['COD'] * 0.01,
        'NH3-N': 0.05 + np.random.normal(0, 0.01) + inlet['NH3-N'] * 0.003,
        'TP': 0.10 + np.random.normal(0, 0.01) + inlet['TP'] * 0.01,
        'TN': 5 + np.random.normal(0, 0.5) + inlet['TN'] * 0.02,
        'SS': 3 + np.random.normal(0, 0.5) + inlet['SS'] * 0.005
    }

def save_to_supabase(inlet, outlet_real, outlet_pred, source="manual"):
    try:
        data = {
            'cod_in': inlet.get('COD', 0),
            'nh3_in': inlet.get('NH3-N', 0),
            'tp_in': inlet.get('TP', 0),
            'tn_in': inlet.get('TN', 0),
            'ss_in': inlet.get('SS', 0),
            'flow_in': inlet.get('流量', 0),
            'pac': inlet.get('PAC', 0),
            'carbon': inlet.get('碳源', 0),
            'mlss': inlet.get('MLSS', 0),
            'do_val': inlet.get('DO', 0),
            'cod_real': outlet_real.get('COD', 0),
            'nh3_real': outlet_real.get('NH3-N', 0),
            'tp_real': outlet_real.get('TP', 0),
            'tn_real': outlet_real.get('TN', 0),
            'ss_real': outlet_real.get('SS', 0),
            'cod_pred': outlet_pred.get('COD', 0),
            'nh3_pred': outlet_pred.get('NH3-N', 0),
            'tp_pred': outlet_pred.get('TP', 0),
            'tn_pred': outlet_pred.get('TN', 0),
            'ss_pred': outlet_pred.get('SS', 0),
            'source': source
        }
        result = supabase_client.table('feedback_data').insert(data).execute()
        return True, "数据已永久保存"
    except Exception as e:
        return False, f"保存失败: {str(e)}"

def get_saved_count():
    try:
        result = supabase_client.table('feedback_data').select('*', count='exact').execute()
        return result.count
    except:
        return 0

# ==========================================
# 诊断函数（强化版，基于进水异常给出详细分析）
# ==========================================
def diagnose_inlet(inlet):
    """根据进水水质生成诊断"""
    diagnoses = []
    cod = inlet.get('COD', 0)
    nh3 = inlet.get('NH3-N', 0)
    tp = inlet.get('TP', 0)
    tn = inlet.get('TN', 0)
    ss = inlet.get('SS', 0)
    
    if cod > 500:
        diagnoses.append({
            'level': 'critical',
            'indicator': '进水COD',
            'current': f"{cod:.0f} mg/L",
            'title': '🚨 进水COD严重超标（>500 mg/L）',
            'reasons': ['工业废水偷排', '管网沉积物冲刷', '污泥厌氧消化液回流'],
            'actions': ['增加碳源投加量30-40%', '提高好氧段DO至3.0-3.5 mg/L', '降低进水量15-20%']
        })
    elif cod > 400:
        diagnoses.append({
            'level': 'warning',
            'indicator': '进水COD',
            'current': f"{cod:.0f} mg/L",
            'title': '⚠️ 进水COD偏高（400-500 mg/L）',
            'reasons': ['工业废水间歇性排放冲击', '管网沉积物释放'],
            'actions': ['增加碳源投加量20%', '提高DO至2.5-3.0 mg/L']
        })
    elif cod < 100 and cod > 0:
        diagnoses.append({
            'level': 'info',
            'indicator': '进水COD',
            'current': f"{cod:.0f} mg/L",
            'title': 'ℹ️ 进水COD偏低（<100 mg/L）',
            'reasons': ['雨水稀释', '上游截流'],
            'actions': ['减少碳源投加量20-30%', '适当降低曝气量']
        })
    if nh3 > 45:
        diagnoses.append({
            'level': 'critical',
            'indicator': '进水NH₃-N',
            'current': f"{nh3:.1f} mg/L",
            'title': '🚨 进水NH₃-N严重超标（>45 mg/L）',
            'reasons': ['工业废水偷排高浓度氨氮', '污泥消化液回流'],
            'actions': ['提高DO至3.5-4.0 mg/L', '补充NaHCO₃ 80-100mg/L', '延长污泥龄']
        })
    elif nh3 > 35:
        diagnoses.append({
            'level': 'warning',
            'indicator': '进水NH₃-N',
            'current': f"{nh3:.1f} mg/L",
            'title': '⚠️ 进水NH₃-N偏高（35-45 mg/L）',
            'reasons': ['上游氨氮浓度升高', '硝化菌活性受抑制'],
            'actions': ['提高DO至3.0-3.5 mg/L', '补充碱度50-80 mg/L']
        })
    if tp > 7.0:
        diagnoses.append({
            'level': 'critical',
            'indicator': '进水TP',
            'current': f"{tp:.2f} mg/L",
            'title': '🚨 进水TP严重超标（>7.0 mg/L）',
            'reasons': ['工业废水偷排高浓度磷废水', '污泥厌氧释磷'],
            'actions': ['增加PAC投加量40-50%', '检查pH 6.5-7.5', '增加排泥']
        })
    elif tp > 5.0:
        diagnoses.append({
            'level': 'warning',
            'indicator': '进水TP',
            'current': f"{tp:.2f} mg/L",
            'title': '⚠️ 进水TP偏高（5.0-7.0 mg/L）',
            'reasons': ['上游含磷废水浓度波动', 'PAC投加量相对不足'],
            'actions': ['增加PAC投加量20-30%', '检查pH并调节']
        })
    if tn > 50:
        diagnoses.append({
            'level': 'warning',
            'indicator': '进水TN',
            'current': f"{tn:.1f} mg/L",
            'title': '⚠️ 进水TN偏高（>50 mg/L）',
            'reasons': ['上游氨氮/有机氮升高', '反硝化碳源不足'],
            'actions': ['增加碳源投加量', '检查缺氧段DO']
        })
    if ss > 350:
        diagnoses.append({
            'level': 'warning',
            'indicator': '进水SS',
            'current': f"{ss:.0f} mg/L",
            'title': '⚠️ 进水SS严重偏高（>350 mg/L）',
            'reasons': ['管网冲刷', '初沉池运行异常'],
            'actions': ['增加初沉池排泥频率', '投加PAM絮凝剂']
        })
    return diagnoses

def diagnose_outlet(outlet, inlet, pac, carbon, mlss, do):
    """结合出水异常生成诊断"""
    diagnoses = []
    cod_out = outlet.get('COD', 0)
    nh3_out = outlet.get('NH3-N', 0)
    tp_out = outlet.get('TP', 0)
    tn_out = outlet.get('TN', 0)
    ss_out = outlet.get('SS', 0)
    
    if cod_out > DESIGN_LIMITS['COD']['value']:
        diagnoses.append({
            'level': 'critical' if cod_out > 45 else 'warning',
            'indicator': '出水COD',
            'current': f"{cod_out:.1f} mg/L",
            'title': f"{'🚨' if cod_out > 45 else '⚠️'} 出水COD超标",
            'reasons': [f'进水COD负荷过高（{inlet["COD"]:.0f}）', f'DO不足（{do:.1f}）', '污泥老化'],
            'actions': [f'增加碳源{int(carbon)}→{int(carbon*1.25)}', f'提高DO至2.5-3.0']
        })
    if nh3_out > DESIGN_LIMITS['NH3-N']['value']:
        diagnoses.append({
            'level': 'critical' if nh3_out > 3.0 else 'warning',
            'indicator': '出水NH₃-N',
            'current': f"{nh3_out:.3f} mg/L",
            'title': f"{'🚨' if nh3_out > 3.0 else '⚠️'} 出水NH₃-N超标",
            'reasons': [f'DO不足（{do:.1f}）', '碱度不足', 'SRT太短'],
            'actions': ['提高DO至3.0-3.5', '补充NaHCO₃ 50-80mg/L', '延长SRT至15天以上']
        })
    if tp_out > DESIGN_LIMITS['TP']['value']:
        diagnoses.append({
            'level': 'critical' if tp_out > 0.6 else 'warning',
            'indicator': '出水TP',
            'current': f"{tp_out:.3f} mg/L",
            'title': f"{'🚨' if tp_out > 0.6 else '⚠️'} 出水TP超标",
            'reasons': [f'PAC不足（{pac:.0f}）', 'pH不适宜', '磷释放'],
            'actions': [f'增加PAC {pac}→{int(pac*1.4)}', '调整投加点', '增加排泥']
        })
    if tn_out > DESIGN_LIMITS['TN']['value']:
        diagnoses.append({
            'level': 'warning',
            'indicator': '出水TN',
            'current': f"{tn_out:.1f} mg/L",
            'title': '⚠️ 出水TN超标',
            'reasons': ['碳源不足', '缺氧段DO过高', '回流量不足'],
            'actions': ['增加碳源投加量', '降低缺氧段DO至0.5以下', '增加内回流量']
        })
    if ss_out > DESIGN_LIMITS['SS']['value']:
        diagnoses.append({
            'level': 'warning',
            'indicator': '出水SS',
            'current': f"{ss_out:.1f} mg/L",
            'title': '⚠️ 出水SS超标',
            'reasons': ['表面负荷过高', 'SVI升高', '排泥不足'],
            'actions': ['增加排泥20%', '投加PAM', '降低进水量10-15%']
        })
    # 运行参数异常
    if do < 0.8:
        diagnoses.append({
            'level': 'critical',
            'indicator': '溶解氧DO',
            'current': f"{do:.1f} mg/L",
            'title': '🚨 好氧段DO严重不足（<0.8 mg/L）',
            'reasons': ['曝气设备故障', '进水负荷突增'],
            'actions': ['检查曝气设备', '加大风机风量20-30%']
        })
    elif do < 1.5:
        diagnoses.append({
            'level': 'warning',
            'indicator': '溶解氧DO',
            'current': f"{do:.1f} mg/L",
            'title': '⚠️ 好氧段DO偏低（<1.5 mg/L）',
            'reasons': ['曝气量不足', '进水负荷增加'],
            'actions': ['增加曝气量10-20%', '监测DO变化趋势']
        })
    if mlss < 2500:
        diagnoses.append({
            'level': 'warning',
            'indicator': '污泥浓度MLSS',
            'current': f"{mlss:.0f} mg/L",
            'title': '⚠️ 污泥浓度偏低（<2500 mg/L）',
            'reasons': ['污泥流失过多', '进水负荷过低'],
            'actions': ['减少排泥量', '增加污泥回流量']
        })
    elif mlss > 6000:
        diagnoses.append({
            'level': 'info',
            'indicator': '污泥浓度MLSS',
            'current': f"{mlss:.0f} mg/L",
            'title': 'ℹ️ 污泥浓度偏高（>6000 mg/L）',
            'reasons': ['排泥不足', '二沉池泥层过厚'],
            'actions': ['增加排泥量', '检查二沉池泥位']
        })
    if pac < 20:
        diagnoses.append({
            'level': 'warning',
            'indicator': 'PAC投加量',
            'current': f"{pac:.0f} mg/L",
            'title': '⚠️ PAC投加量偏低（<20 mg/L）',
            'reasons': ['PAC储备不足', '加药泵故障'],
            'actions': ['增加PAC至30-50 mg/L', '检查加药泵']
        })
    elif pac > 80:
        diagnoses.append({
            'level': 'info',
            'indicator': 'PAC投加量',
            'current': f"{pac:.0f} mg/L",
            'title': 'ℹ️ PAC投加量偏高（>80 mg/L）',
            'reasons': ['为应对高负荷临时加大'],
            'actions': ['评估是否可降低', '检查出水TP是否达标']
        })
    if carbon < 30:
        diagnoses.append({
            'level': 'warning',
            'indicator': '碳源投加量',
            'current': f"{carbon:.0f} mg/L",
            'title': '⚠️ 碳源投加量偏低（<30 mg/L）',
            'reasons': ['碳源储备不足', '反硝化碳源缺乏'],
            'actions': ['增加碳源至40-60 mg/L', '检查碳源储罐液位']
        })
    elif carbon > 100:
        diagnoses.append({
            'level': 'info',
            'indicator': '碳源投加量',
            'current': f"{carbon:.0f} mg/L",
            'title': 'ℹ️ 碳源投加量偏高（>100 mg/L）',
            'reasons': ['为应对高负荷临时加大'],
            'actions': ['评估是否可逐步降低', '检查出水COD和TN']
        })
    return diagnoses

# ==========================================
# 侧边栏：四种数据输入模式
# ==========================================
st.sidebar.markdown("## 📊 数据输入模式")
input_mode_global = st.sidebar.radio(
    "选择数据模式",
    ["✏️ 手动输入", "📁 文件上传", "📡 API接入", "🔄 自动实时（模拟）"],
    index=0
)

REQUIRED_COLS = ['COD', 'NH3-N', 'TP', 'TN', 'SS', '流量', 'PAC', '碳源', 'MLSS', 'DO']

cod_in = nh3_in = tp_in = tn_in = ss_in = flow_in = 0
pac = carbon = mlss = do = 0
input_data = None
simulated_outlet = None

# --- 1. 手动输入 ---
if input_mode_global == "✏️ 手动输入":
    st.sidebar.markdown("### 进水实测")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        cod_in = st.number_input("COD (mg/L)", min_value=0.0, value=200.0, key="manual_cod")
        nh3_in = st.number_input("NH₃-N (mg/L)", min_value=0.0, value=20.0, key="manual_nh3")
        tn_in = st.number_input("TN (mg/L)", min_value=0.0, value=30.0, key="manual_tn")
    with c2:
        tp_in = st.number_input("TP (mg/L)", min_value=0.0, value=3.0, key="manual_tp")
        ss_in = st.number_input("SS (mg/L)", min_value=0.0, value=150.0, key="manual_ss")
    flow_in = st.sidebar.number_input("流量 (m³/h)", min_value=0.0, value=10000.0, key="manual_flow")
    st.sidebar.markdown("### 运行参数")
    c3, c4 = st.sidebar.columns(2)
    with c3:
        pac = st.number_input("PAC (mg/L)", min_value=0.0, value=30.0, key="manual_pac")
        carbon = st.number_input("碳源 (mg/L)", min_value=0.0, value=50.0, key="manual_carbon")
    with c4:
        mlss = st.number_input("MLSS (mg/L)", min_value=0.0, value=4000.0, key="manual_mlss")
        do = st.number_input("DO (mg/L)", min_value=0.0, value=2.0, key="manual_do")
    
    # 保存当前输入用于诊断（即使未预测）
    st.session_state.current_inlet = {'COD': cod_in, 'NH3-N': nh3_in, 'TP': tp_in, 'TN': tn_in, 'SS': ss_in, '流量': flow_in}
    st.session_state.current_params = {'PAC': pac, '碳源': carbon, 'MLSS': mlss, 'DO': do}
    
    if st.sidebar.button("🔮 预测", type="primary", use_container_width=True):
        input_data = build_input_with_lags(cod_in, nh3_in, tp_in, ss_in, flow_in, pac, carbon, mlss, do, tn_in)
        result = predict_removal_effluent(input_data)
        if result:
            st.session_state.pred_result = result
            st.session_state.has_prediction = True
        else:
            st.sidebar.error("❌ 预测失败")

# --- 2. 文件上传 ---
elif input_mode_global == "📁 文件上传":
    st.sidebar.markdown("### 📁 上传数据文件")
    st.sidebar.caption("请上传包含以下列的 Excel/CSV 文件：")
    st.sidebar.code("COD, NH3-N, TP, TN, SS, 流量, PAC, 碳源, MLSS, DO", language='text')
    
    if st.sidebar.button("📥 下载空模板 (Excel)"):
        template_df = pd.DataFrame(columns=REQUIRED_COLS)
        template_df.loc[0] = [200, 20, 3.0, 30, 150, 10000, 30, 50, 4000, 2.0]
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            template_df.to_excel(writer, index=False, sheet_name='模板')
        st.sidebar.download_button(
            label="📥 下载模板.xlsx",
            data=output.getvalue(),
            file_name="进水数据模板.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    uploaded_file = st.sidebar.file_uploader("选择文件", type=['xlsx', 'csv'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
            missing_cols = set(REQUIRED_COLS) - set(df_upload.columns)
            if missing_cols:
                st.sidebar.error(f"❌ 缺少必需列：{missing_cols}")
            else:
                row = df_upload.iloc[0]
                cod_in = row['COD']
                nh3_in = row['NH3-N']
                tp_in = row['TP']
                tn_in = row['TN']
                ss_in = row['SS']
                flow_in = row['流量']
                pac = row['PAC']
                carbon = row['碳源']
                mlss = row['MLSS']
                do = row['DO']
                st.session_state.current_inlet = {'COD': cod_in, 'NH3-N': nh3_in, 'TP': tp_in, 'TN': tn_in, 'SS': ss_in, '流量': flow_in}
                st.session_state.current_params = {'PAC': pac, '碳源': carbon, 'MLSS': mlss, 'DO': do}
                input_data = build_input_with_lags(cod_in, nh3_in, tp_in, ss_in, flow_in, pac, carbon, mlss, do, tn_in)
                result = predict_removal_effluent(input_data)
                if result:
                    st.session_state.pred_result = result
                    st.session_state.has_prediction = True
                    st.sidebar.success("✅ 数据加载成功")
                else:
                    st.sidebar.error("❌ 预测失败")
        except Exception as e:
            st.sidebar.error(f"文件解析失败: {e}")

# --- 3. API 接入 ---
elif input_mode_global == "📡 API接入":
    st.sidebar.markdown("### 📡 API 实时数据")
    api_url = st.sidebar.text_input("API地址", value="http://localhost:8080/api/data", key="api_url")
    api_key = st.sidebar.text_input("API Key", type="password", key="api_key")
    if st.sidebar.button("🔄 获取数据"):
        try:
            import requests
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            resp = requests.get(api_url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                cod_in = data.get('COD', 0)
                nh3_in = data.get('NH3-N', 0)
                tp_in = data.get('TP', 0)
                tn_in = data.get('TN', 0)
                ss_in = data.get('SS', 0)
                flow_in = data.get('流量', 0)
                pac = data.get('PAC', 0)
                carbon = data.get('碳源', 0)
                mlss = data.get('MLSS', 0)
                do = data.get('DO', 0)
                st.session_state.current_inlet = {'COD': cod_in, 'NH3-N': nh3_in, 'TP': tp_in, 'TN': tn_in, 'SS': ss_in, '流量': flow_in}
                st.session_state.current_params = {'PAC': pac, '碳源': carbon, 'MLSS': mlss, 'DO': do}
                input_data = build_input_with_lags(cod_in, nh3_in, tp_in, ss_in, flow_in, pac, carbon, mlss, do, tn_in)
                result = predict_removal_effluent(input_data)
                if result:
                    st.session_state.pred_result = result
                    st.session_state.has_prediction = True
                    st.sidebar.success("✅ 数据获取成功")
                else:
                    st.sidebar.error("❌ 预测失败")
            else:
                st.sidebar.error(f"❌ API 返回错误：{resp.status_code}")
        except Exception as e:
            st.sidebar.error(f"❌ 连接失败：{str(e)}")

# --- 4. 自动实时（模拟） ---
else:
    st.sidebar.markdown("### 🔄 自动实时数据")
    st.sidebar.info("🔄 每5秒自动生成一组模拟数据")
    if st.sidebar.button("▶️ 启动实时数据流"):
        st.session_state.auto_mode_running = True
        st.sidebar.success("✅ 数据流已启动")
    if st.sidebar.button("⏹️ 停止数据流"):
        st.session_state.auto_mode_running = False
        st.sidebar.info("⏹️ 数据流已停止")
    
    simulated_inlet = generate_simulated_data()
    cod_in = simulated_inlet['COD']
    nh3_in = simulated_inlet['NH3-N']
    tp_in = simulated_inlet['TP']
    tn_in = simulated_inlet['TN']
    ss_in = simulated_inlet['SS']
    flow_in = simulated_inlet['流量']
    pac = simulated_inlet['PAC']
    carbon = simulated_inlet['碳源']
    mlss = simulated_inlet['MLSS']
    do = simulated_inlet['DO']
    st.session_state.current_inlet = {'COD': cod_in, 'NH3-N': nh3_in, 'TP': tp_in, 'TN': tn_in, 'SS': ss_in, '流量': flow_in}
    st.session_state.current_params = {'PAC': pac, '碳源': carbon, 'MLSS': mlss, 'DO': do}
    input_data = build_input_with_lags(cod_in, nh3_in, tp_in, ss_in, flow_in, pac, carbon, mlss, do, tn_in)
    simulated_outlet = simulate_outlet(simulated_inlet)
    
    result = predict_removal_effluent(input_data)
    if result:
        st.session_state.pred_result = result
        st.session_state.has_prediction = True
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div class="data-status-realtime">
        📊 当前数据：第 {st.session_state.simulation_counter + 1} 组
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 主界面
# ==========================================

has_pred = st.session_state.get('has_prediction', False)
pred_result = st.session_state.get('pred_result', None)

if has_pred and pred_result:
    outlet_pred = pred_result['effluent']
    removal = pred_result['removal']
    inlet = pred_result['inlet']
    params = pred_result['params']
    
    has_abnormal = False
    for key in ['COD', 'NH3-N', 'TP', 'TN', 'SS']:
        if outlet_pred.get(key, 0) > DESIGN_LIMITS.get(key, {'value': 999})['value']:
            has_abnormal = True
            break
    
    status_text = "🔴 异常" if has_abnormal else "🟢 正常"
    status_color = "value-critical" if has_abnormal else "value-normal"
    with status_placeholder:
        st.markdown(f"""
        <div class="status-metric">
            <div class="label">📊 数据状态</div>
            <div class="value {status_color}">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)
else:
    # 无预测时显示等待状态，但仍可展示进水诊断
    with status_placeholder:
        st.markdown("""
        <div class="status-metric">
            <div class="label">📊 数据状态</div>
            <div class="value value-normal">等待输入</div>
        </div>
        """, unsafe_allow_html=True)
    # 使用默认值显示布局
    outlet_pred = {'COD': 14.0, 'NH3-N': 0.05, 'TP': 0.10, 'TN': 5.0, 'SS': 3.0}
    inlet = st.session_state.current_inlet
    params = st.session_state.current_params
    removal = 0.85

# ---- 进出水水质面板 ----
st.markdown('<div class="section-header">📊 进出水水质实时监测</div>', unsafe_allow_html=True)
st.caption(f"📌 出水设计标准：COD≤{DESIGN_LIMITS['COD']['value']} | NH₃-N≤{DESIGN_LIMITS['NH3-N']['value']} | TP≤{DESIGN_LIMITS['TP']['value']} | TN≤{DESIGN_LIMITS['TN']['value']} | SS≤{DESIGN_LIMITS['SS']['value']} mg/L")

col_left, col_right = st.columns(2, gap="medium")
with col_left:
    st.markdown("""
    <div class="water-card-in">
        <div style="font-size:15px; font-weight:600; color:#1a3a5c; margin-bottom:6px;">
            🔵 进水水质 <span style="font-size:11px; font-weight:400; color:#888;">（实测）</span>
        </div>
    """, unsafe_allow_html=True)
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown(f"""<div class="metric-card"><div class="label">COD</div><div class="value">{inlet['COD']:.0f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div></div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="metric-card"><div class="label">NH₃-N</div><div class="value">{inlet['NH3-N']:.1f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div></div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="metric-card"><div class="label">TP</div><div class="value">{inlet['TP']:.2f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div></div>""", unsafe_allow_html=True)
    with cc2:
        st.markdown(f"""<div class="metric-card"><div class="label">TN</div><div class="value">{inlet.get('TN', 0):.1f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div></div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="metric-card"><div class="label">SS</div><div class="value">{inlet['SS']:.0f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div></div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="metric-card"><div class="label">流量</div><div class="value">{inlet['流量']:.0f} <span style="font-size:13px;font-weight:400;color:#888;">m³/h</span></div></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # 根据是否有点击预测显示不同标签
    outlet_label = "实测" if has_pred else "—"   # 按用户要求改为"实测"
    st.markdown(f"""
    <div class="water-card-out">
        <div style="font-size:15px; font-weight:600; color:#1a5c3a; margin-bottom:6px;">
            🟢 出水水质 <span style="font-size:11px; font-weight:400; color:#888;">（{outlet_label}）</span>
        </div>
    """, unsafe_allow_html=True)
    cod_ok = outlet_pred['COD'] <= DESIGN_LIMITS['COD']['value']
    nh3_ok = outlet_pred['NH3-N'] <= DESIGN_LIMITS['NH3-N']['value']
    tp_ok = outlet_pred['TP'] <= DESIGN_LIMITS['TP']['value']
    tn_ok = outlet_pred.get('TN', 0) <= DESIGN_LIMITS['TN']['value']
    ss_ok = outlet_pred['SS'] <= DESIGN_LIMITS['SS']['value']
    cc3, cc4 = st.columns(2)
    with cc3:
        st.markdown(f"""<div class="metric-card"><div class="label">COD <span class="limit-ref">限值≤{DESIGN_LIMITS['COD']['value']}</span></div><div class="value" style="color:{'#1B7A4A' if cod_ok else '#C0392B'}">{outlet_pred['COD']:.1f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div><div class="sub">{'✅ 达标' if cod_ok else f'🔴 超标{outlet_pred["COD"]-DESIGN_LIMITS["COD"]["value"]:.1f}'}</div></div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="metric-card"><div class="label">NH₃-N <span class="limit-ref">限值≤{DESIGN_LIMITS['NH3-N']['value']}</span></div><div class="value" style="color:{'#1B7A4A' if nh3_ok else '#C0392B'}">{outlet_pred['NH3-N']:.3f} / {DESIGN_LIMITS['NH3-N']['value']} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div><div class="sub">{'✅ 达标' if nh3_ok else f'🔴 超标{outlet_pred["NH3-N"]-DESIGN_LIMITS["NH3-N"]["value"]:.3f}'}</div></div>""", unsafe_allow_html=True)
    with cc4:
        st.markdown(f"""<div class="metric-card"><div class="label">TP <span class="limit-ref">限值≤{DESIGN_LIMITS['TP']['value']}</span></div><div class="value" style="color:{'#1B7A4A' if tp_ok else '#C0392B'}">{outlet_pred['TP']:.3f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div><div class="sub">{'✅ 达标' if tp_ok else f'🔴 超标{outlet_pred["TP"]-DESIGN_LIMITS["TP"]["value"]:.3f}'}</div></div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="metric-card"><div class="label">TN <span class="limit-ref">限值≤{DESIGN_LIMITS['TN']['value']}</span></div><div class="value" style="color:{'#1B7A4A' if tn_ok else '#C0392B'}">{outlet_pred.get('TN', 0):.1f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div><div class="sub">{'✅ 达标' if tn_ok else f'🔴 超标{outlet_pred.get("TN", 0)-DESIGN_LIMITS["TN"]["value"]:.1f}'}</div></div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="metric-card"><div class="label">SS <span class="limit-ref">限值≤{DESIGN_LIMITS['SS']['value']}</span></div><div class="value" style="color:{'#1B7A4A' if ss_ok else '#C0392B'}">{outlet_pred['SS']:.1f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div><div class="sub">{'✅ 达标' if ss_ok else f'🔴 超标{outlet_pred["SS"]-DESIGN_LIMITS["SS"]["value"]:.1f}'}</div></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- 趋势图（保留） ----
st.markdown('<div class="section-header">📈 进出水趋势（近24小时）</div>', unsafe_allow_html=True)
st.caption("🟦 实线 = 实测值 | 虚线 = 预测值")

recent_data = st.session_state.data_buffer.get_recent(24)
if len(recent_data) > 1:
    df_trend = pd.DataFrame([{
        'timestamp': d['timestamp'],
        'inlet_COD': d['inlet']['COD'],
        'inlet_NH3': d['inlet']['NH3-N'],
        'inlet_TP': d['inlet']['TP'],
        'inlet_TN': d['inlet'].get('TN', 0),
        'outlet_COD_real': d['outlet']['COD'] if d['outlet'] else None,
        'outlet_COD_pred': d['pred_outlet']['COD'] if d['pred_outlet'] else None,
        'outlet_NH3_real': d['outlet']['NH3-N'] if d['outlet'] else None,
        'outlet_NH3_pred': d['pred_outlet']['NH3-N'] if d['pred_outlet'] else None,
        'outlet_TP_real': d['outlet']['TP'] if d['outlet'] else None,
        'outlet_TP_pred': d['pred_outlet']['TP'] if d['pred_outlet'] else None,
        'outlet_TN_real': d['outlet']['TN'] if d['outlet'] else None,
        'outlet_TN_pred': d['pred_outlet']['TN'] if d['pred_outlet'] else None,
    } for d in recent_data])
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=('COD', 'NH₃-N', 'TP', 'TN'))
    
    # COD
    fig.add_trace(go.Scatter(x=df_trend['timestamp'], y=df_trend['inlet_COD'],
                            name='进水COD', line=dict(color='#E74C3C', width=2)), row=1, col=1)
    mask = df_trend['outlet_COD_real'].notna()
    if mask.any():
        fig.add_trace(go.Scatter(x=df_trend[mask]['timestamp'], y=df_trend[mask]['outlet_COD_real'],
                                name='出水COD_实测', line=dict(color='#2E86AB', width=2.5)), row=1, col=1)
    mask = df_trend['outlet_COD_pred'].notna()
    if mask.any():
        fig.add_trace(go.Scatter(x=df_trend[mask]['timestamp'], y=df_trend[mask]['outlet_COD_pred'],
                                name='出水COD_预测', line=dict(color='#2E86AB', width=2, dash='dot')), row=1, col=1)
    fig.add_hline(y=DESIGN_LIMITS['COD']['value'], line_dash="dash", line_color="red", row=1, col=1)
    
    # NH3-N
    fig.add_trace(go.Scatter(x=df_trend['timestamp'], y=df_trend['inlet_NH3'],
                            name='进水NH₃-N', line=dict(color='#F39C12', width=2)), row=2, col=1)
    mask = df_trend['outlet_NH3_real'].notna()
    if mask.any():
        fig.add_trace(go.Scatter(x=df_trend[mask]['timestamp'], y=df_trend[mask]['outlet_NH3_real'],
                                name='出水NH₃-N_实测', line=dict(color='#27AE60', width=2.5)), row=2, col=1)
    mask = df_trend['outlet_NH3_pred'].notna()
    if mask.any():
        fig.add_trace(go.Scatter(x=df_trend[mask]['timestamp'], y=df_trend[mask]['outlet_NH3_pred'],
                                name='出水NH₃-N_预测', line=dict(color='#27AE60', width=2, dash='dot')), row=2, col=1)
    fig.add_hline(y=DESIGN_LIMITS['NH3-N']['value'], line_dash="dash", line_color="red", row=2, col=1)
    
    # TP
    fig.add_trace(go.Scatter(x=df_trend['timestamp'], y=df_trend['inlet_TP'],
                            name='进水TP', line=dict(color='#8E44AD', width=2)), row=3, col=1)
    mask = df_trend['outlet_TP_real'].notna()
    if mask.any():
        fig.add_trace(go.Scatter(x=df_trend[mask]['timestamp'], y=df_trend[mask]['outlet_TP_real'],
                                name='出水TP_实测', line=dict(color='#F39C12', width=2.5)), row=3, col=1)
    mask = df_trend['outlet_TP_pred'].notna()
    if mask.any():
        fig.add_trace(go.Scatter(x=df_trend[mask]['timestamp'], y=df_trend[mask]['outlet_TP_pred'],
                                name='出水TP_预测', line=dict(color='#F39C12', width=2, dash='dot')), row=3, col=1)
    fig.add_hline(y=DESIGN_LIMITS['TP']['value'], line_dash="dash", line_color="red", row=3, col=1)
    
    # TN
    fig.add_trace(go.Scatter(x=df_trend['timestamp'], y=df_trend['inlet_TN'],
                            name='进水TN', line=dict(color='#2980B9', width=2)), row=4, col=1)
    mask = df_trend['outlet_TN_real'].notna()
    if mask.any():
        fig.add_trace(go.Scatter(x=df_trend[mask]['timestamp'], y=df_trend[mask]['outlet_TN_real'],
                                name='出水TN_实测', line=dict(color='#1ABC9C', width=2.5)), row=4, col=1)
    mask = df_trend['outlet_TN_pred'].notna()
    if mask.any():
        fig.add_trace(go.Scatter(x=df_trend[mask]['timestamp'], y=df_trend[mask]['outlet_TN_pred'],
                                name='出水TN_预测', line=dict(color='#1ABC9C', width=2, dash='dot')), row=4, col=1)
    fig.add_hline(y=DESIGN_LIMITS['TN']['value'], line_dash="dash", line_color="red", row=4, col=1)
    
    fig.update_layout(height=600, showlegend=True, hovermode='x unified')
    fig.update_xaxes(title_text="时间（北京时间）", row=4, col=1)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("📭 数据收集中... 请等待更多数据点（至少2个时间点）")

# ---- 记忆长度与分频调控 ----
st.markdown('<div class="section-header">🧠 记忆长度与分频调控策略</div>', unsafe_allow_html=True)
st.caption("💡 基于XGBoost-SHAP分析的系统记忆长度共识")

col_ch1, col_ch2, col_ch3 = st.columns(3)
with col_ch1:
    st.markdown("""
    <div class="channel-item channel-fast">
        <div class="ch-name">⚡ 快速通道</div>
        <div class="ch-value" style="color:#27AE60;">1-9h</div>
        <div class="ch-desc">NH₃-N (1h) · TP (1h) · TN (9h)</div>
        <div class="ch-desc">✅ 树模型共识</div>
    </div>
    """, unsafe_allow_html=True)
with col_ch2:
    st.markdown("""
    <div class="channel-item channel-special">
        <div class="ch-name">⚠️ 不适用</div>
        <div class="ch-value" style="color:#E74C3C;">—</div>
        <div class="ch-desc">COD — 去除冗余</div>
        <div class="ch-desc">ℹ️ 出水COD长期稳定</div>
    </div>
    """, unsafe_allow_html=True)
with col_ch3:
    st.markdown("""
    <div class="channel-item channel-special">
        <div class="ch-name">🔴 特殊通道</div>
        <div class="ch-value" style="color:#E74C3C;">不稳定</div>
        <div class="ch-desc">SS — 实时阈值报警</div>
        <div class="ch-desc">⚠️ 物理沉淀主导</div>
    </div>
    """, unsafe_allow_html=True)

# ---- 时序决策建议 ----
st.markdown('<div class="section-header">⏱️ 时序决策建议（具体操作）</div>', unsafe_allow_html=True)
indicator = st.selectbox("选择指标", ['NH3-N', 'TP', 'TN', 'COD', 'SS'])

mem_info = MEMORY.get(indicator, {})
mem = mem_info.get('hours')
current_val = outlet_pred.get(indicator, 0)
limit = DESIGN_LIMITS.get(indicator, {}).get('value', 999)

if mem is not None and mem > 0:
    if indicator == 'NH3-N':
        steps = [(0, "🚨 记录进水NH₃-N异常值，启动应急响应"), (0.5, "📞 通知值班长，准备碱度调节剂"), (1, "⚙️ 提高好氧段DO至3.0-3.5 mg/L"), (2, "🔍 检查碱度，若<100则补充NaHCO₃"), (3, "📊 评估出水NH₃-N变化趋势"), (4, "✅ 确认NH₃-N稳定达标")]
    elif indicator == 'TP':
        steps = [(0, "🚨 记录进水TP异常值，启动应急响应"), (0.5, "📞 通知值班长，确认PAC储备"), (1, "⚙️ 增加PAC投加量30%"), (2, "🔍 检查pH，若<6.5则投加碱调节"), (3, "📊 评估出水TP变化趋势"), (4, "✅ 确认TP稳定达标")]
    elif indicator == 'TN':
        steps = [(0, "🚨 记录进水TN异常值，启动应急响应"), (3, "📞 通知值班长，确认碳源储备"), (6, "⚙️ 增加碳源投加量20-30%"), (9, "🔍 检查缺氧段DO，若>0.5则调整回流比"), (12, "📊 评估出水TN变化趋势"), (18, "✅ 确认TN稳定达标")]
    else:
        steps = [(0, "ℹ️ 该指标不适用记忆长度分析")]
    
    st.markdown('<div style="background:#FAFBFC;border-radius:8px;padding:10px 14px;border:1px solid #E8ECF0;">', unsafe_allow_html=True)
    st.markdown(f"**📋 {indicator}：{current_val:.2f} / {limit} mg/L**")
    st.markdown(f"**🧠 记忆长度：{mem} 小时**")
    st.markdown("---")
    for t, action in steps:
        st.markdown(f"""<div class="timeline-step"><div class="timeline-time">⏱️ {t}h</div><div class="timeline-action">{action}</div></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info(f"ℹ️ {indicator} 不适用于记忆长度分析")

# ---- 异常诊断与工艺优化建议（强化版，结合进水和出水） ----
st.markdown('<div class="section-header">🔍 异常诊断与工艺优化建议</div>', unsafe_allow_html=True)
st.caption("💡 基于同类型A²/O工艺经验库 + 当前工况多维度分析")

# 始终显示进水诊断（即使无预测）
inlet_diag = diagnose_inlet(inlet)
outlet_diag = []
if has_pred:
    outlet_diag = diagnose_outlet(outlet_pred, inlet, params['PAC'], params['碳源'], params['MLSS'], params['DO'])

all_diag = inlet_diag + outlet_diag
if all_diag:
    # 按严重程度排序
    level_order = {'critical': 0, 'warning': 1, 'info': 2}
    all_diag.sort(key=lambda x: level_order.get(x['level'], 3))
    for d in all_diag:
        with st.expander(f"{d['title']}（当前值：{d['current']}）", expanded=(d['level'] == 'critical')):
            col_r, col_a = st.columns([1, 1])
            with col_r:
                st.markdown("**🔍 可能原因**")
                for r in d['reasons']:
                    st.markdown(f"- {r}")
            with col_a:
                st.markdown("**💡 针对性工艺优化措施**")
                for a in d['actions']:
                    st.markdown(f"- {a}")
else:
    st.success("✅ 系统运行正常，未检测到异常")
    st.info("📋 建议：保持当前运行参数，定期巡检设备。")

# ---- 永久记忆统计 ----
st.markdown("---")
col_stats1, col_stats2, col_stats3 = st.columns(3)
saved_count = get_saved_count()
with col_stats1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">📦 已永久保存数据</div>
        <div class="stat-value">{saved_count} 组</div>
    </div>
    """, unsafe_allow_html=True)
with col_stats2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">🧠 模型版本</div>
        <div class="stat-value">v7.0</div>
        <div class="stat-sub">去除率模型</div>
    </div>
    """, unsafe_allow_html=True)
with col_stats3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-label">🧠 记忆长度共识</div>
        <div class="stat-value">NH₃-N 1h · TP 1h · TN 9h</div>
        <div class="stat-sub">XGBoost-SHAP分析</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
beijing_now = datetime.now(BEIJING_TZ)
st.caption(f"🏭 v7.0 | 四种输入模式 | 去除率模型 | 永久记忆已启用 | {beijing_now.strftime('%Y-%m-%d %H:%M')} 北京时间")
