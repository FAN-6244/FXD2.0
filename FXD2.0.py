# -*- coding: utf-8 -*-
"""
FXD3.0 - 水质净化厂智能预警与调控决策系统 v3.0
================================================
合并 版面.txt 的完整布局 + 去除率模型的数据逻辑
四种输入模式 · 永久记忆(Supabase) · 时序决策 · 异常诊断

预测目标：去除率(%) = (进水浓度 - 出水浓度) / 进水浓度 × 100%
记忆长度：COD=46h, NH3-N=45h, TP=46h, SS=41h（XGBoost-SHAP分析）
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
import os

warnings.filterwarnings('ignore')

# ==========================================
# 可选依赖：Supabase（永久记忆）
# ==========================================
SUPABASE_AVAILABLE = False
try:
    import supabase
    SUPABASE_AVAILABLE = True
except ImportError:
    pass

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
# CSS样式（完整版 — 来自版面.txt）
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
    .metric-card .label {
        font-size: 12px;
        color: #666;
        font-weight: 500;
    }
    .metric-card .value {
        font-size: 18px;
        font-weight: 700;
        color: #1a3a5c;
    }
    .metric-card .sub {
        font-size: 11px;
        color: #999;
    }
    .stat-card {
        background: white;
        border-radius: 6px;
        padding: 6px 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        margin-bottom: 3px;
        border-left: 4px solid #2E86AB;
    }
    .stat-card .stat-label {
        font-size: 12px;
        color: #666;
        font-weight: 500;
    }
    .stat-card .stat-value {
        font-size: 18px;
        font-weight: 700;
        color: #1a3a5c;
    }
    .stat-card .stat-sub {
        font-size: 11px;
        color: #999;
    }
    .limit-ref {
        font-size: 11px;
        color: #888;
        background: #F0F0F0;
        padding: 1px 8px;
        border-radius: 10px;
        display: inline-block;
    }
    .channel-container { display: flex; gap: 10px; margin: 6px 0; }
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
    .removal-badge {
        font-size: 11px;
        color: #1565C0;
        background: #E3F2FD;
        padding: 1px 8px;
        border-radius: 10px;
        display: inline-block;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 设计标准（含TN）
# ==========================================
DESIGN_LIMITS = {
    'COD':   {'value': 30,  'unit': 'mg/L'},
    'NH3-N': {'value': 1.5, 'unit': 'mg/L'},
    'TP':    {'value': 0.3, 'unit': 'mg/L'},
    'TN':    {'value': 10,  'unit': 'mg/L'},
    'SS':    {'value': 10,  'unit': 'mg/L'}
}

# ==========================================
# 记忆长度（去除率模型 XGBoost-SHAP 分析结果）
# SML: COD=46h, NH3-N=45h, TP=46h, SS=41h
# ==========================================
MEMORY = {
    'COD':   {'hours': 46, 'channel': '慢速', 'freq': '12-24h', 'description': '有机物降解，受泥龄与负荷影响'},
    'NH3-N': {'hours': 45, 'channel': '中速', 'freq': '8-12h',  'description': '硝化反应，DO敏感，碳源依赖'},
    'TP':    {'hours': 46, 'channel': '慢速', 'freq': '12-24h', 'description': '化学除磷+生物除磷耦合'},
    'TN':    {'hours': 45, 'channel': '中速', 'freq': '8-12h',  'description': '反硝化，碳源投加延迟'},
    'SS':    {'hours': 41, 'channel': '快速', 'freq': '3-4h',   'description': '物理沉淀，受水力负荷直接影响'}
}

# 三通道分组
CHANNELS = {
    'fast': {
        'name': '⚡ 快速通道',
        'color': '#27AE60',
        'indicators': ['SS'],
        'desc': 'SS (41h) | 更新 3-4h',
        'consensus': '⚠️ 模型R²偏低，建议实时阈值报警'
    },
    'medium': {
        'name': '🔄 中速通道',
        'color': '#2E86AB',
        'indicators': ['NH3-N', 'TN'],
        'desc': 'NH₃-N (45h) · TN (45h) | 更新 8-12h',
        'consensus': '✅ XGBoost模型可用'
    },
    'slow': {
        'name': '🐢 慢速通道',
        'color': '#F39C12',
        'indicators': ['COD', 'TP'],
        'desc': 'COD (46h) · TP (46h) | 更新 12-24h',
        'consensus': '✅ XGBoost模型可用'
    }
}

# 经验去除率（无模型文件时的后备值）
EMPIRICAL_REMOVAL = {
    'COD':   0.93,
    'NH3-N': 0.95,
    'TP':    0.88,
    'TN':    0.75,
    'SS':    0.92
}

# ==========================================
# Supabase 配置（可选）
# ==========================================
SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://esoulexcrpdeeoumoili.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', 'sb_publishable_m0hz9Rv8NB_ziC5xKCltMg_Ij5Od60Q')

supabase_client = None
if SUPABASE_AVAILABLE:
    try:
        supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        SUPABASE_AVAILABLE = False

# ==========================================
# 数据缓存管理
# ==========================================
class DataBuffer:
    def __init__(self, max_hours=48):
        self.max_hours = max_hours
        self.data = []

    def add_data(self, timestamp, inlet, outlet, pred_outlet, removal_rates):
        self.data.append({
            'timestamp': timestamp,
            'inlet': inlet.copy(),
            'outlet': outlet.copy() if outlet else None,
            'pred_outlet': pred_outlet.copy() if pred_outlet else None,
            'removal': removal_rates.copy() if removal_rates else None
        })
        cutoff = datetime.now(BEIJING_TZ) - timedelta(hours=self.max_hours)
        self.data = [d for d in self.data if d['timestamp'] >= cutoff]

    def get_recent(self, hours=24):
        cutoff = datetime.now(BEIJING_TZ) - timedelta(hours=hours)
        return [d for d in self.data if d['timestamp'] >= cutoff]

# ==========================================
# 加载去除率模型（灵活加载，无文件时用经验值）
# ==========================================
@st.cache_resource
def load_removal_rate_models():
    """
    尝试加载去除率模型文件。
    优先级：
      1. model_cache/xgb_final_model.pkl (主模型)
      2. model_cache/xgb_final_{indicator}.pkl (分指标模型)
      3. 无文件时返回None，使用经验去除率
    """
    model_dir = 'model_cache'
    models = {}
    scaler = None
    feature_cols = None

    # 加载scaler和feature_cols
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    feat_path = os.path.join(model_dir, 'feature_cols.pkl')

    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            except Exception:
                scaler = None

    if os.path.exists(feat_path):
        try:
            with open(feat_path, 'rb') as f:
                feature_cols = pickle.load(f)
        except Exception:
            feature_cols = None

    # 加载主模型
    main_model_path = os.path.join(model_dir, 'xgb_final_model.pkl')
    main_model = None
    if os.path.exists(main_model_path):
        try:
            main_model = joblib.load(main_model_path)
        except Exception:
            try:
                with open(main_model_path, 'rb') as f:
                    main_model = pickle.load(f)
            except Exception:
                main_model = None

    # 尝试加载分指标模型
    for ind in ['COD', 'NH3', 'TP', 'TN', 'SS']:
        ind_path = os.path.join(model_dir, f'xgb_final_{ind}.pkl')
        if os.path.exists(ind_path):
            try:
                models[ind] = joblib.load(ind_path)
            except Exception:
                try:
                    with open(ind_path, 'rb') as f:
                        models[ind] = pickle.load(f)
                except Exception:
                    pass

    # 如果主模型存在但没有分指标模型，把主模型分配给所有指标
    if main_model is not None and not models:
        for ind in ['COD', 'NH3', 'TP', 'TN', 'SS']:
            models[ind] = main_model

    has_model = len(models) > 0 and scaler is not None and feature_cols is not None
    return models, scaler, feature_cols, has_model

models_dict, scaler, feature_cols, HAS_MODEL = load_removal_rate_models()

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
if 'tunable_models' not in st.session_state:
    st.session_state.tunable_models = models_dict.copy() if models_dict else {}

# ==========================================
# 标题
# ==========================================
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
    model_status = "去除率模型" if HAS_MODEL else "经验值模式"
    st.markdown(f"""
    <div class="status-metric">
        <div class="label">📋 预测模式</div>
        <div class="value">{model_status}</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 特征构造（匹配去除率模型训练特征）
# ==========================================
def build_features_for_model(cod, nh3, tp, tn, ss, flow, pac, carbon, mlss, do):
    """
    构造与去除率模型训练时一致的特征矩阵。
    遍历feature_cols，按列名模式匹配赋值。
    """
    if feature_cols is None:
        return None
    data = {}
    for col in feature_cols:
        if '进水COD' in col or ('COD' in col and '出水' not in col and '去除' not in col):
            data[col] = cod
        elif 'NH3' in col and 'detrend' in col:
            data[col] = nh3 * 0.9
        elif 'NH3' in col and '出水' not in col and '去除' not in col:
            data[col] = nh3
        elif 'TP' in col and 'detrend' in col:
            data[col] = tp * 0.9
        elif 'TP' in col and '出水' not in col and '去除' not in col:
            data[col] = tp
        elif 'TN' in col and '出水' not in col and '去除' not in col:
            data[col] = tn
        elif 'SS' in col and '出水' not in col and '去除' not in col:
            data[col] = ss
        elif '流量' in col:
            data[col] = flow
        elif '降雨量' in col:
            data[col] = 0
        elif '风量' in col:
            data[col] = 50000
        elif '污泥浓度' in col or 'MLSS' in col:
            data[col] = mlss
        elif '溶解氧' in col or ('DO' in col and 'COD' not in col):
            data[col] = do
        elif '产泥量' in col:
            data[col] = 20
        elif '碳源' in col:
            data[col] = carbon
        elif '磁粉' in col:
            data[col] = 0
        elif 'PAC' in col:
            data[col] = pac
        elif '阴离子' in col:
            data[col] = 150
        elif '阳离子' in col:
            data[col] = 200
        elif '次氯酸钠' in col:
            data[col] = 0.1
        elif '生化池' in col and 'DO' in col:
            data[col] = do
        elif '_diff' in col:
            data[col] = 0
        elif '_roll_mean' in col or '_roll_std' in col:
            data[col] = 0
        elif '_lag' in col:
            data[col] = 0
        else:
            data[col] = 0
    return pd.DataFrame([data])

# ==========================================
# 预测函数（去除率 → 反推浓度）
# ==========================================
def predict_removal_rates(cod, nh3, tp, tn, ss, flow, pac, carbon, mlss, do):
    """
    预测各指标去除率，反推出水浓度。
    有模型时用模型预测，无模型时用经验值。
    """
    removal_rates = {}
    effluent = {}

    indicators_map = {
        'COD':   ('COD',   cod),
        'NH3-N': ('NH3',   nh3),
        'TP':    ('TP',    tp),
        'TN':    ('TN',    tn),
        'SS':    ('SS',    ss),
    }

    if HAS_MODEL:
        X = build_features_for_model(cod, nh3, tp, tn, ss, flow, pac, carbon, mlss, do)
        if X is not None:
            try:
                X_scaled = scaler.transform(X)
            except Exception:
                X_scaled = X.values
        else:
            X_scaled = None
    else:
        X_scaled = None

    for display_name, (model_key, inlet_val) in indicators_map.items():
        model = st.session_state.tunable_models.get(model_key)
        if model is not None and X_scaled is not None:
            try:
                pred = model.predict(X_scaled)[0]
                # 去除率裁剪到 [0, 1]
                pred = max(0.0, min(1.0, float(pred)))
                removal_rates[display_name] = pred
            except Exception:
                removal_rates[display_name] = EMPIRICAL_REMOVAL[display_name]
        else:
            removal_rates[display_name] = EMPIRICAL_REMOVAL[display_name]

        # 反推出水浓度
        effluent[display_name] = inlet_val * (1 - removal_rates[display_name])

    return removal_rates, effluent

# ==========================================
# 生成模拟实时数据
# ==========================================
def generate_simulated_data():
    base_cod = 200 + np.random.normal(0, 30)
    base_nh3 = 20 + np.random.normal(0, 3)
    base_tp = 3.0 + np.random.normal(0, 0.4)
    base_tn = 30 + np.random.normal(0, 4)
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
    """模拟实测出水（用于自动实时模式的校准对比）"""
    return {
        'COD':   8 + np.random.normal(0, 0.8) + inlet['COD'] * 0.01,
        'NH3-N': 0.05 + np.random.normal(0, 0.01) + inlet['NH3-N'] * 0.003,
        'TP':    0.10 + np.random.normal(0, 0.01) + inlet['TP'] * 0.01,
        'TN':    5 + np.random.normal(0, 0.5) + inlet['TN'] * 0.02,
        'SS':    3 + np.random.normal(0, 0.5) + inlet['SS'] * 0.005
    }

# ==========================================
# 模型微调（可选）
# ==========================================
def calibrate_model(indicator, inlet_data, real_removal_rate):
    """用实测去除率微调模型（如果模型可用）"""
    if not HAS_MODEL:
        return False, "模型未加载，无法微调"
    model_key = {'COD': 'COD', 'NH3-N': 'NH3', 'TP': 'TP', 'TN': 'TN', 'SS': 'SS'}.get(indicator, indicator)
    model = st.session_state.tunable_models.get(model_key)
    if model is None:
        return False, f"{indicator} 模型不存在"
    try:
        if hasattr(model, 'fit'):
            X = build_features_for_model(
                inlet_data['COD'], inlet_data['NH3-N'], inlet_data['TP'],
                inlet_data.get('TN', 30), inlet_data['SS'], inlet_data['流量'],
                inlet_data.get('PAC', 30), inlet_data.get('碳源', 50),
                inlet_data.get('MLSS', 4000), inlet_data.get('DO', 2.0)
            )
            if X is not None:
                X_scaled = scaler.transform(X)
                model.fit(X_scaled, [real_removal_rate])
                st.session_state.calibration_count += 1
                return True, f"✅ {indicator} 微调成功"
        return False, "模型不支持增量训练"
    except Exception as e:
        return False, f"❌ 微调失败: {str(e)}"

# ==========================================
# Supabase 永久记忆（可选）
# ==========================================
def save_to_supabase(inlet, outlet_real, outlet_pred, removal_rates, source="manual"):
    if not SUPABASE_AVAILABLE or supabase_client is None:
        return False, "Supabase未配置"
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
            'cod_real': outlet_real.get('COD', 0) if outlet_real else 0,
            'nh3_real': outlet_real.get('NH3-N', 0) if outlet_real else 0,
            'tp_real': outlet_real.get('TP', 0) if outlet_real else 0,
            'tn_real': outlet_real.get('TN', 0) if outlet_real else 0,
            'ss_real': outlet_real.get('SS', 0) if outlet_real else 0,
            'cod_pred': outlet_pred.get('COD', 0),
            'nh3_pred': outlet_pred.get('NH3-N', 0),
            'tp_pred': outlet_pred.get('TP', 0),
            'tn_pred': outlet_pred.get('TN', 0),
            'ss_pred': outlet_pred.get('SS', 0),
            'cod_removal': removal_rates.get('COD', 0),
            'nh3_removal': removal_rates.get('NH3-N', 0),
            'tp_removal': removal_rates.get('TP', 0),
            'tn_removal': removal_rates.get('TN', 0),
            'ss_removal': removal_rates.get('SS', 0),
            'source': source
        }
        result = supabase_client.table('feedback_data').insert(data).execute()
        return True, "数据已永久保存"
    except Exception as e:
        return False, f"保存失败: {str(e)}"

def get_saved_count():
    if not SUPABASE_AVAILABLE or supabase_client is None:
        return 0
    try:
        result = supabase_client.table('feedback_data').select('*', count='exact').execute()
        return result.count
    except Exception:
        return 0

# ==========================================
# 完整诊断函数
# ==========================================
def diagnose_system(inlet, outlet, pac, carbon, mlss, do):
    diagnoses = []

    # --- 进水异常 ---
    if inlet['COD'] > 500:
        diagnoses.append({
            'level': 'critical', 'indicator': '进水COD',
            'current': f"{inlet['COD']:.0f} mg/L",
            'title': '🚨 进水COD严重超标（>500 mg/L）',
            'reasons': ['工业废水偷排', '管网沉积物冲刷', '污泥厌氧消化液回流'],
            'actions': ['增加碳源投加量30-40%', '提高好氧段DO至3.0-3.5 mg/L', '降低进水量15-20%']
        })
    elif inlet['COD'] > 400:
        diagnoses.append({
            'level': 'warning', 'indicator': '进水COD',
            'current': f"{inlet['COD']:.0f} mg/L",
            'title': '⚠️ 进水COD偏高（400-500 mg/L）',
            'reasons': ['工业废水间歇性排放冲击', '管网沉积物释放'],
            'actions': ['增加碳源投加量20%', '提高DO至2.5-3.0 mg/L']
        })
    elif inlet['COD'] < 100 and inlet['COD'] > 0:
        diagnoses.append({
            'level': 'info', 'indicator': '进水COD',
            'current': f"{inlet['COD']:.0f} mg/L",
            'title': 'ℹ️ 进水COD偏低（<100 mg/L）',
            'reasons': ['雨水稀释', '上游截流'],
            'actions': ['减少碳源投加量20-30%', '适当降低曝气量']
        })

    if inlet['NH3-N'] > 45:
        diagnoses.append({
            'level': 'critical', 'indicator': '进水NH₃-N',
            'current': f"{inlet['NH3-N']:.1f} mg/L",
            'title': '🚨 进水NH₃-N严重超标（>45 mg/L）',
            'reasons': ['工业废水偷排高浓度氨氮', '污泥消化液回流'],
            'actions': ['提高DO至3.5-4.0 mg/L', '补充NaHCO₃ 80-100mg/L', '延长污泥龄']
        })
    elif inlet['NH3-N'] > 35:
        diagnoses.append({
            'level': 'warning', 'indicator': '进水NH₃-N',
            'current': f"{inlet['NH3-N']:.1f} mg/L",
            'title': '⚠️ 进水NH₃-N偏高（35-45 mg/L）',
            'reasons': ['上游氨氮浓度升高', '硝化菌活性受抑制'],
            'actions': ['提高DO至3.0-3.5 mg/L', '补充碱度50-80 mg/L']
        })

    if inlet['TP'] > 7.0:
        diagnoses.append({
            'level': 'critical', 'indicator': '进水TP',
            'current': f"{inlet['TP']:.2f} mg/L",
            'title': '🚨 进水TP严重超标（>7.0 mg/L）',
            'reasons': ['工业废水偷排高浓度磷废水', '污泥厌氧释磷'],
            'actions': ['增加PAC投加量40-50%', '检查pH 6.5-7.5', '增加排泥']
        })
    elif inlet['TP'] > 5.0:
        diagnoses.append({
            'level': 'warning', 'indicator': '进水TP',
            'current': f"{inlet['TP']:.2f} mg/L",
            'title': '⚠️ 进水TP偏高（5.0-7.0 mg/L）',
            'reasons': ['上游含磷废水浓度波动', 'PAC投加量相对不足'],
            'actions': ['增加PAC投加量20-30%', '检查pH并调节']
        })

    if inlet['SS'] > 350:
        diagnoses.append({
            'level': 'warning', 'indicator': '进水SS',
            'current': f"{inlet['SS']:.0f} mg/L",
            'title': '⚠️ 进水SS严重偏高（>350 mg/L）',
            'reasons': ['管网冲刷', '初沉池运行异常'],
            'actions': ['增加初沉池排泥频率', '投加PAM絮凝剂']
        })

    # --- 出水超标 ---
    for ind in ['COD', 'NH3-N', 'TP', 'TN', 'SS']:
        if outlet.get(ind, 0) > DESIGN_LIMITS[ind]['value']:
            limit = DESIGN_LIMITS[ind]['value']
            val = outlet[ind]
            level = 'critical' if val > limit * 1.5 else 'warning'
            icon = '🚨' if level == 'critical' else '⚠️'
            diagnoses.append({
                'level': level, 'indicator': f'出水{ind}',
                'current': f"{val:.2f} mg/L",
                'title': f'{icon} 出水{ind}超标',
                'reasons': [f'进水{ind}负荷过高（{inlet.get(ind, 0):.1f}）',
                           f'DO不足（{do:.1f}）', '去除率下降'],
                'actions': _get_outlet_actions(ind, pac, carbon, do)
            })

    # --- 运行参数 ---
    if do < 0.8:
        diagnoses.append({
            'level': 'critical', 'indicator': '溶解氧DO',
            'current': f"{do:.1f} mg/L",
            'title': '🚨 好氧段DO严重不足（<0.8 mg/L）',
            'reasons': ['曝气设备故障', '进水负荷突增'],
            'actions': ['检查曝气设备', '加大风机风量20-30%']
        })
    elif do < 1.5:
        diagnoses.append({
            'level': 'warning', 'indicator': '溶解氧DO',
            'current': f"{do:.1f} mg/L",
            'title': '⚠️ 好氧段DO偏低（<1.5 mg/L）',
            'reasons': ['曝气量不足', '进水负荷增加'],
            'actions': ['增加曝气量10-20%', '监测DO变化趋势']
        })

    if mlss < 2500:
        diagnoses.append({
            'level': 'warning', 'indicator': '污泥浓度MLSS',
            'current': f"{mlss:.0f} mg/L",
            'title': '⚠️ 污泥浓度偏低（<2500 mg/L）',
            'reasons': ['污泥流失过多', '进水负荷过低'],
            'actions': ['减少排泥量', '增加污泥回流量']
        })
    elif mlss > 6000:
        diagnoses.append({
            'level': 'info', 'indicator': '污泥浓度MLSS',
            'current': f"{mlss:.0f} mg/L",
            'title': 'ℹ️ 污泥浓度偏高（>6000 mg/L）',
            'reasons': ['排泥不足', '二沉池泥层过厚'],
            'actions': ['增加排泥量', '检查二沉池泥位']
        })

    if pac < 20:
        diagnoses.append({
            'level': 'warning', 'indicator': 'PAC投加量',
            'current': f"{pac:.0f} mg/L",
            'title': '⚠️ PAC投加量偏低（<20 mg/L）',
            'reasons': ['PAC储备不足', '加药泵故障'],
            'actions': ['增加PAC至30-50 mg/L', '检查加药泵']
        })

    if carbon < 30:
        diagnoses.append({
            'level': 'warning', 'indicator': '碳源投加量',
            'current': f"{carbon:.0f} mg/L",
            'title': '⚠️ 碳源投加量偏低（<30 mg/L）',
            'reasons': ['碳源储备不足', '反硝化碳源缺乏'],
            'actions': ['增加碳源至40-60 mg/L', '检查碳源储罐液位']
        })

    return diagnoses

def _get_outlet_actions(indicator, pac, carbon, do):
    """根据出水超标指标返回针对性措施"""
    actions = {
        'COD':   [f'增加碳源{int(carbon)}→{int(carbon*1.25)}', f'提高DO至2.5-3.0'],
        'NH3-N': ['提高DO至3.0-3.5', '补充NaHCO₃ 50-80mg/L', '延长SRT至15天以上'],
        'TP':    [f'增加PAC {pac}→{int(pac*1.4)}', '调整投加点', '增加排泥'],
        'TN':    [f'增加碳源{int(carbon)}→{int(carbon*1.3)}', '检查内外回流比', '提高缺氧段容积利用率'],
        'SS':    ['增加排泥20%', '投加PAM', '降低进水量10-15%']
    }
    return actions.get(indicator, ['检查工艺运行状态'])

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

# --- 初始化变量 ---
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
        tp_in = st.number_input("TP (mg/L)", min_value=0.0, value=3.0, key="manual_tp")
    with c2:
        tn_in = st.number_input("TN (mg/L)", min_value=0.0, value=30.0, key="manual_tn")
        ss_in = st.number_input("SS (mg/L)", min_value=0.0, value=150.0, key="manual_ss")
        flow_in = st.number_input("流量 (m³/h)", min_value=0.0, value=10000.0, key="manual_flow")
    st.sidebar.markdown("### 运行参数")
    c3, c4 = st.sidebar.columns(2)
    with c3:
        pac = st.number_input("PAC (mg/L)", min_value=0.0, value=30.0, key="manual_pac")
        carbon = st.number_input("碳源 (mg/L)", min_value=0.0, value=50.0, key="manual_carbon")
    with c4:
        mlss = st.number_input("MLSS (mg/L)", min_value=0.0, value=4000.0, key="manual_mlss")
        do = st.number_input("DO (mg/L)", min_value=0.0, value=2.0, key="manual_do")
    input_data = {
        'COD': cod_in, 'NH3-N': nh3_in, 'TP': tp_in, 'TN': tn_in, 'SS': ss_in,
        '流量': flow_in, 'PAC': pac, '碳源': carbon, 'MLSS': mlss, 'DO': do
    }
    st.sidebar.info("💡 手动模式：修改参数后自动更新预测")

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
                input_data = None
            else:
                row = df_upload.iloc[0]
                cod_in = row['COD']; nh3_in = row['NH3-N']; tp_in = row['TP']
                tn_in = row['TN']; ss_in = row['SS']; flow_in = row['流量']
                pac = row['PAC']; carbon = row['碳源']; mlss = row['MLSS']; do = row['DO']
                input_data = {
                    'COD': cod_in, 'NH3-N': nh3_in, 'TP': tp_in, 'TN': tn_in, 'SS': ss_in,
                    '流量': flow_in, 'PAC': pac, '碳源': carbon, 'MLSS': mlss, 'DO': do
                }
                st.sidebar.success(f"✅ 成功加载数据 (共 {len(df_upload)} 行)")
        except Exception as e:
            st.sidebar.error(f"❌ 文件解析失败：{str(e)}")
            input_data = None
    else:
        input_data = None

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
                cod_in = data.get('COD', 0); nh3_in = data.get('NH3-N', 0)
                tp_in = data.get('TP', 0); tn_in = data.get('TN', 30)
                ss_in = data.get('SS', 0); flow_in = data.get('流量', 0)
                pac = data.get('PAC', 0); carbon = data.get('碳源', 0)
                mlss = data.get('MLSS', 0); do = data.get('DO', 0)
                input_data = {
                    'COD': cod_in, 'NH3-N': nh3_in, 'TP': tp_in, 'TN': tn_in, 'SS': ss_in,
                    '流量': flow_in, 'PAC': pac, '碳源': carbon, 'MLSS': mlss, 'DO': do
                }
                st.sidebar.success("✅ 数据获取成功")
            else:
                st.sidebar.error(f"❌ API 返回错误：{resp.status_code}")
                input_data = None
        except Exception as e:
            st.sidebar.error(f"❌ 连接失败：{str(e)}")
            input_data = None
    else:
        input_data = None

# --- 4. 自动实时（模拟） ---
else:
    st.sidebar.markdown("### 🔄 自动实时数据")
    st.sidebar.info("🔄 每次刷新生成一组模拟数据")
    if st.sidebar.button("▶️ 启动实时数据流"):
        st.session_state.auto_mode_running = True
        st.sidebar.success("✅ 数据流已启动")
    if st.sidebar.button("⏹️ 停止数据流"):
        st.session_state.auto_mode_running = False
        st.sidebar.info("⏹️ 数据流已停止")
    simulated_inlet = generate_simulated_data()
    cod_in = simulated_inlet['COD']; nh3_in = simulated_inlet['NH3-N']
    tp_in = simulated_inlet['TP']; tn_in = simulated_inlet['TN']
    ss_in = simulated_inlet['SS']; flow_in = simulated_inlet['流量']
    pac = simulated_inlet['PAC']; carbon = simulated_inlet['碳源']
    mlss = simulated_inlet['MLSS']; do = simulated_inlet['DO']
    input_data = {
        'COD': cod_in, 'NH3-N': nh3_in, 'TP': tp_in, 'TN': tn_in, 'SS': ss_in,
        '流量': flow_in, 'PAC': pac, '碳源': carbon, 'MLSS': mlss, 'DO': do
    }
    simulated_outlet = simulate_outlet(simulated_inlet)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div class="data-status-realtime">
        📊 当前数据：第 {st.session_state.simulation_counter + 1} 组
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 主界面
# ==========================================
if input_data is not None:
    # 执行预测
    removal_rates, effluent_pred = predict_removal_rates(
        cod_in, nh3_in, tp_in, tn_in, ss_in, flow_in, pac, carbon, mlss, do
    )

    inlet = {'COD': cod_in, 'NH3-N': nh3_in, 'TP': tp_in, 'TN': tn_in, 'SS': ss_in, '流量': flow_in}

    # ---- 自动模式处理 ----
    if input_mode_global == "🔄 自动实时（模拟）" and st.session_state.auto_mode_running and simulated_outlet is not None:
        real_outlet = {
            'COD':   max(0, simulated_outlet['COD'] + np.random.normal(0, 0.3)),
            'NH3-N': max(0, simulated_outlet['NH3-N'] + np.random.normal(0, 0.005)),
            'TP':    max(0, simulated_outlet['TP'] + np.random.normal(0, 0.005)),
            'TN':    max(0, simulated_outlet['TN'] + np.random.normal(0, 0.3)),
            'SS':    max(0, simulated_outlet['SS'] + np.random.normal(0, 0.2))
        }
        # 计算实测去除率
        real_removal = {}
        for ind in ['COD', 'NH3-N', 'TP', 'TN', 'SS']:
            inlet_val = inlet.get(ind, 0)
            if inlet_val > 0.01:
                real_removal[ind] = max(0, min(1, (inlet_val - real_outlet[ind]) / inlet_val))
            else:
                real_removal[ind] = 0

        # 每5次微调一次
        if st.session_state.simulation_counter % 5 == 0 and st.session_state.simulation_counter > 0:
            for ind in ['COD', 'NH3-N', 'TP', 'TN', 'SS']:
                calibrate_model(ind, input_data, real_removal[ind])
            success, msg = save_to_supabase(inlet, real_outlet, effluent_pred, removal_rates, "auto")
            st.session_state.feedback_log.append({
                'timestamp': datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S'),
                'type': 'auto_calibration',
                'cod_pred': effluent_pred['COD'], 'cod_real': real_outlet['COD']
            })

        st.session_state.data_buffer.add_data(
            timestamp=datetime.now(BEIJING_TZ),
            inlet=inlet, outlet=real_outlet,
            pred_outlet=effluent_pred, removal_rates=removal_rates
        )
        st.session_state.simulation_counter += 1
        st.info(f"🔄 实时数据流运行中... 已接收 {st.session_state.simulation_counter} 组数据 | 已微调 {st.session_state.calibration_count} 次")
        outlet_display = real_outlet
        outlet_label = "实测"
    else:
        outlet_display = effluent_pred.copy()
        st.session_state.data_buffer.add_data(
            timestamp=datetime.now(BEIJING_TZ),
            inlet=inlet, outlet=None,
            pred_outlet=effluent_pred, removal_rates=removal_rates
        )
        outlet_label = "预测"

    # ---- 状态更新 ----
    has_abnormal = False
    for key in ['COD', 'NH3-N', 'TP', 'TN', 'SS']:
        if outlet_display.get(key, 0) > DESIGN_LIMITS[key]['value']:
            has_abnormal = True
            break
    if inlet.get('COD', 0) > 400 or inlet.get('NH3-N', 0) > 35 or inlet.get('TP', 0) > 5:
        has_abnormal = True
    status_text = "异常" if has_abnormal else "正常"
    status_color = "value-critical" if has_abnormal else "value-normal"
    with status_placeholder:
        st.markdown(f"""
        <div class="status-metric">
            <div class="label">📊 数据状态</div>
            <div class="value {status_color}">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)

    # ================================================================
    # 1. 进出水水质面板
    # ================================================================
    st.markdown('<div class="section-header">📊 进出水水质实时监测</div>', unsafe_allow_html=True)
    st.caption(f"📌 出水设计标准：COD≤{DESIGN_LIMITS['COD']['value']} | NH₃-N≤{DESIGN_LIMITS['NH3-N']['value']} | TP≤{DESIGN_LIMITS['TP']['value']} | TN≤{DESIGN_LIMITS['TN']['value']} | SS≤{DESIGN_LIMITS['SS']['value']} mg/L")

    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        st.markdown("""
        <div class="water-card-in">
            <div style="font-size:15px; font-weight:600; color:#1a3a5c; margin-bottom:6px;">
                🔵 进水水质 <span style="font-size:11px; font-weight:400; color:#888;">（实测）</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(f"""<div class="metric-card"><div class="label">COD</div><div class="value">{inlet['COD']:.0f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="metric-card"><div class="label">NH₃-N</div><div class="value">{inlet['NH3-N']:.1f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="metric-card"><div class="label">TP</div><div class="value">{inlet['TP']:.2f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div></div>""", unsafe_allow_html=True)
        with cc2:
            st.markdown(f"""<div class="metric-card"><div class="label">TN</div><div class="value">{inlet['TN']:.1f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="metric-card"><div class="label">SS</div><div class="value">{inlet['SS']:.0f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="metric-card"><div class="label">流量</div><div class="value">{inlet['流量']:.0f} <span style="font-size:13px;font-weight:400;color:#888;">m³/h</span></div></div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown(f"""
        <div class="water-card-out">
            <div style="font-size:15px; font-weight:600; color:#1a5c3a; margin-bottom:6px;">
                🟢 出水水质 <span style="font-size:11px; font-weight:400; color:#888;">（{outlet_label}）</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        indicators_display = ['COD', 'NH3-N', 'TP', 'TN', 'SS']
        cc3, cc4 = st.columns(2)
        for idx, ind in enumerate(indicators_display):
            val = outlet_display.get(ind, 0)
            limit = DESIGN_LIMITS[ind]['value']
            ok = val <= limit
            color = '#1B7A4A' if ok else '#C0392B'
            border_color = '#27AE60' if ok else '#E74C3C'
            status = '✅ 达标' if ok else f'🔴 超标{val-limit:.2f}'
            removal_pct = removal_rates.get(ind, 0) * 100
            target_col = cc3 if idx < 3 else cc4
            if idx == 4:
                target_col = cc4
            with target_col:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {border_color};">
                    <div class="label">{ind} <span class="limit-ref">限值≤{limit}</span></div>
                    <div class="value" style="color:{color};">{val:.2f} <span style="font-size:13px;font-weight:400;color:#888;">mg/L</span></div>
                    <div class="sub">{status} <span class="removal-badge">去除率 {removal_pct:.1f}%</span></div>
                </div>
                """, unsafe_allow_html=True)

    # ---- 运行参数 ----
    st.markdown("---")
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
    with col_p1:
        st.metric("PAC", f"{pac:.0f} mg/L")
    with col_p2:
        st.metric("碳源", f"{carbon:.0f} mg/L")
    with col_p3:
        st.metric("MLSS", f"{mlss:.0f} mg/L")
    with col_p4:
        st.metric("DO", f"{do:.1f} mg/L")
    with col_p5:
        avg_removal = np.mean([removal_rates.get(ind, 0) for ind in indicators_display]) * 100
        st.metric("平均去除率", f"{avg_removal:.1f}%")

    # ================================================================
    # 2. 趋势图（近24小时）
    # ================================================================
    st.markdown('<div class="section-header">📈 进出水趋势（近24小时）</div>', unsafe_allow_html=True)
    st.caption("🟦 实线 = 实测/进水 | 虚线 = 预测 | 🟥进水COD 🟧进水NH₃-N 🟪进水TP")

    recent_data = st.session_state.data_buffer.get_recent(24)
    if len(recent_data) > 1:
        df_trend = pd.DataFrame([{
            'timestamp': d['timestamp'],
            'inlet_COD': d['inlet']['COD'],
            'inlet_NH3': d['inlet']['NH3-N'],
            'inlet_TP': d['inlet']['TP'],
            'outlet_COD_real': d['outlet']['COD'] if d['outlet'] else None,
            'outlet_COD_pred': d['pred_outlet']['COD'] if d['pred_outlet'] else None,
            'outlet_NH3_real': d['outlet']['NH3-N'] if d['outlet'] else None,
            'outlet_NH3_pred': d['pred_outlet']['NH3-N'] if d['pred_outlet'] else None,
            'outlet_TP_real': d['outlet']['TP'] if d['outlet'] else None,
            'outlet_TP_pred': d['pred_outlet']['TP'] if d['pred_outlet'] else None,
        } for d in recent_data])

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=('COD', 'NH₃-N', 'TP'))

        # COD
        fig.add_trace(go.Scatter(x=df_trend['timestamp'], y=df_trend['inlet_COD'],
                                name='进水COD', line=dict(color='#E74C3C', width=2)), row=1, col=1)
        mask_real = df_trend['outlet_COD_real'].notna()
        if mask_real.any():
            fig.add_trace(go.Scatter(x=df_trend[mask_real]['timestamp'], y=df_trend[mask_real]['outlet_COD_real'],
                                    name='出水COD_实测', line=dict(color='#2E86AB', width=2.5)), row=1, col=1)
        mask_pred = df_trend['outlet_COD_pred'].notna()
        if mask_pred.any():
            fig.add_trace(go.Scatter(x=df_trend[mask_pred]['timestamp'], y=df_trend[mask_pred]['outlet_COD_pred'],
                                    name='出水COD_预测', line=dict(color='#2E86AB', width=2, dash='dot')), row=1, col=1)
        fig.add_hline(y=DESIGN_LIMITS['COD']['value'], line_dash="dash", line_color="red", row=1, col=1)

        # NH3-N
        fig.add_trace(go.Scatter(x=df_trend['timestamp'], y=df_trend['inlet_NH3'],
                                name='进水NH₃-N', line=dict(color='#F39C12', width=2)), row=2, col=1)
        mask_real = df_trend['outlet_NH3_real'].notna()
        if mask_real.any():
            fig.add_trace(go.Scatter(x=df_trend[mask_real]['timestamp'], y=df_trend[mask_real]['outlet_NH3_real'],
                                    name='出水NH₃-N_实测', line=dict(color='#27AE60', width=2.5)), row=2, col=1)
        mask_pred = df_trend['outlet_NH3_pred'].notna()
        if mask_pred.any():
            fig.add_trace(go.Scatter(x=df_trend[mask_pred]['timestamp'], y=df_trend[mask_pred]['outlet_NH3_pred'],
                                    name='出水NH₃-N_预测', line=dict(color='#27AE60', width=2, dash='dot')), row=2, col=1)
        fig.add_hline(y=DESIGN_LIMITS['NH3-N']['value'], line_dash="dash", line_color="red", row=2, col=1)

        # TP
        fig.add_trace(go.Scatter(x=df_trend['timestamp'], y=df_trend['inlet_TP'],
                                name='进水TP', line=dict(color='#8E44AD', width=2)), row=3, col=1)
        mask_real = df_trend['outlet_TP_real'].notna()
        if mask_real.any():
            fig.add_trace(go.Scatter(x=df_trend[mask_real]['timestamp'], y=df_trend[mask_real]['outlet_TP_real'],
                                    name='出水TP_实测', line=dict(color='#F39C12', width=2.5)), row=3, col=1)
        mask_pred = df_trend['outlet_TP_pred'].notna()
        if mask_pred.any():
            fig.add_trace(go.Scatter(x=df_trend[mask_pred]['timestamp'], y=df_trend[mask_pred]['outlet_TP_pred'],
                                    name='出水TP_预测', line=dict(color='#F39C12', width=2, dash='dot')), row=3, col=1)
        fig.add_hline(y=DESIGN_LIMITS['TP']['value'], line_dash="dash", line_color="red", row=3, col=1)

        fig.update_layout(height=450, showlegend=True, hovermode='x unified')
        fig.update_xaxes(title_text="时间（北京时间）", row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📭 数据收集中... 请等待更多数据点（至少2个时间点）")

    # ================================================================
    # 3. 记忆长度与分频调控
    # ================================================================
    st.markdown('<div class="section-header">🧠 记忆长度与分频调控策略</div>', unsafe_allow_html=True)
    st.caption("💡 不同污染物响应速度不同，分通道制定调控策略。基于XGBoost-SHAP去除率模型分析。")

    col_ch1, col_ch2, col_ch3 = st.columns(3)
    with col_ch1:
        st.markdown(f"""
        <div class="channel-item channel-fast">
            <div class="ch-name">{CHANNELS['fast']['name']}</div>
            <div class="ch-value" style="color:{CHANNELS['fast']['color']};">41h</div>
            <div class="ch-desc">{CHANNELS['fast']['desc']}</div>
            <div class="ch-desc">{CHANNELS['fast']['consensus']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_ch2:
        st.markdown(f"""
        <div class="channel-item channel-slow">
            <div class="ch-name">{CHANNELS['medium']['name']}</div>
            <div class="ch-value" style="color:{CHANNELS['medium']['color']};">45h</div>
            <div class="ch-desc">{CHANNELS['medium']['desc']}</div>
            <div class="ch-desc">{CHANNELS['medium']['consensus']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_ch3:
        st.markdown(f"""
        <div class="channel-item channel-special">
            <div class="ch-name">{CHANNELS['slow']['name']}</div>
            <div class="ch-value" style="color:{CHANNELS['slow']['color']};">46h</div>
            <div class="ch-desc">{CHANNELS['slow']['desc']}</div>
            <div class="ch-desc">{CHANNELS['slow']['consensus']}</div>
        </div>
        """, unsafe_allow_html=True)

    # ================================================================
    # 4. 时序决策建议
    # ================================================================
    st.markdown('<div class="section-header">⏱️ 时序决策建议（具体操作）</div>', unsafe_allow_html=True)
    indicator_select = st.selectbox("选择指标", ['COD', 'NH3-N', 'TP', 'TN', 'SS'])
    mem = MEMORY[indicator_select]['hours']
    current_val = outlet_display.get(indicator_select, 0)
    limit = DESIGN_LIMITS[indicator_select]['value']

    if mem:
        if indicator_select == 'COD':
            steps = [
                (0, "🚨 记录进水COD异常值，启动应急响应"),
                (10, "📞 通知值班长，确认碳源储备"),
                (20, "⚙️ 增加碳源投加量20%"),
                (30, "🔍 检查好氧段DO，若<2.0则增加曝气"),
                (40, "📊 评估出水COD变化趋势"),
                (46, "✅ 确认COD稳定达标，逐步回调")
            ]
        elif indicator_select == 'NH3-N':
            steps = [
                (0, "🚨 记录进水NH₃-N异常值，启动应急响应"),
                (10, "📞 通知值班长，准备碱度调节剂"),
                (20, "⚙️ 提高好氧段DO至3.0-3.5 mg/L"),
                (30, "🔍 检查碱度，若<100则补充NaHCO₃"),
                (40, "📊 评估出水NH₃-N变化趋势"),
                (45, "✅ 确认NH₃-N稳定达标")
            ]
        elif indicator_select == 'TP':
            steps = [
                (0, "🚨 记录进水TP异常值，启动应急响应"),
                (10, "📞 通知值班长，确认PAC储备"),
                (20, "⚙️ 增加PAC投加量30%"),
                (30, "🔍 检查pH，若<6.5则投加碱调节"),
                (40, "📊 评估出水TP变化趋势"),
                (46, "✅ 确认TP稳定达标，逐步回调")
            ]
        elif indicator_select == 'TN':
            steps = [
                (0, "🚨 记录进水TN异常值，启动应急响应"),
                (10, "📞 通知值班长，确认碳源储备"),
                (20, "⚙️ 增加碳源投加量25%"),
                (30, "🔍 检查内外回流比，调整至合理范围"),
                (40, "📊 评估出水TN变化趋势"),
                (45, "✅ 确认TN稳定达标")
            ]
        else:  # SS
            steps = [
                (0, "🚨 SS超标，启动应急响应"),
                (5, "📞 检查二沉池刮泥机"),
                (10, "⚙️ 增加排泥量20%"),
                (20, "🔍 检查SVI，若>150投加PAM"),
                (30, "📊 评估SS变化"),
                (41, "✅ 确认达标")
            ]

        st.markdown('<div style="background:#FAFBFC;border-radius:8px;padding:10px 14px;border:1px solid #E8ECF0;">', unsafe_allow_html=True)
        st.markdown(f"**📋 {indicator_select}：{current_val:.2f} / {limit} mg/L | 记忆长度 {mem}h | 去除率 {removal_rates.get(indicator_select, 0)*100:.1f}%**")
        st.markdown("---")
        for t, action in steps:
            st.markdown(f"""
            <div class="timeline-step">
                <div class="timeline-time">⏱️ {t}h</div>
                <div class="timeline-action">{action}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ================================================================
    # 5. 异常诊断与工艺优化
    # ================================================================
    st.markdown('<div class="section-header">🔍 异常诊断与工艺优化建议</div>', unsafe_allow_html=True)
    st.caption("💡 基于同类型A²/O工艺经验库 + 当前工况多维度分析")

    diagnoses = diagnose_system(inlet, outlet_display, pac, carbon, mlss, do)
    if diagnoses:
        level_order = {'critical': 0, 'warning': 1, 'info': 2}
        diagnoses.sort(key=lambda x: level_order.get(x['level'], 3))
        for d in diagnoses:
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

    # ================================================================
    # 6. 永久记忆统计
    # ================================================================
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
            <div class="stat-label">🔄 模型微调次数</div>
            <div class="stat-value">{st.session_state.calibration_count} 次</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stats3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">🧠 记忆长度（去除率模型）</div>
            <div class="stat-value">COD 46h · NH₃-N 45h · TP 46h · SS 41h</div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("👈 请左侧输入数据")
    st.markdown("---")
    st.markdown("#### 📌 使用说明")
    st.markdown("""
    1. 在左侧选择数据输入模式（手动/文件/API/自动实时）
    2. 输入或导入进水水质和运行参数
    3. 系统自动预测各指标去除率并反推出水浓度
    4. 查看进出水对比、趋势图、记忆长度分频调控
    5. 查看时序决策建议和异常诊断
    """)
    st.markdown("---")
    st.markdown("#### 🧠 系统记忆长度（XGBoost-SHAP 去除率模型）")
    st.markdown("""
    | 指标 | 记忆长度 | 通道 | 工艺解释 |
    | :--- | :---: | :---: | :--- |
    | **COD** | **46 小时** | 慢速 | 有机物降解，受泥龄与负荷影响 |
    | **NH₃-N** | **45 小时** | 中速 | 硝化反应，DO敏感，碳源依赖 |
    | **TP** | **46 小时** | 慢速 | 化学除磷+生物除磷耦合 |
    | **TN** | **45 小时** | 中速 | 反硝化，碳源投加延迟 |
    | **SS** | **41 小时** | 快速 | 物理沉淀，受水力负荷直接影响 |
    """)
    st.caption(f"📌 基于去除率模型 · XGBoost-SHAP 分析 · {'模型已加载' if HAS_MODEL else '使用经验值（请运行模型训练脚本生成model_cache）'}")

# ==========================================
# 页脚
# ==========================================
st.markdown("---")
beijing_now = datetime.now(BEIJING_TZ)
st.caption(f"🏭 水质净化厂智能预警与调控决策系统 v3.0 | 去除率模型 | 四种输入模式 | {'永久记忆已启用' if SUPABASE_AVAILABLE else '永久记忆未启用'} | {beijing_now.strftime('%Y-%m-%d %H:%M')} 北京时间")
