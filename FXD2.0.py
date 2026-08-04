# -*- coding: utf-8 -*-
"""
FXD3.0 - 水质净化厂智能预警与调控决策系统 v3.0
================================================
合并 版面.txt 的完整布局 + 去除率模型的数据逻辑
四种输入模式 · 永久记忆(Supabase) · 时序决策 · 异常诊断

预测目标：去除率(%) = (进水浓度 - 出水浓度) / 进水浓度 × 100%
记忆长度：COD=46h, NH3-N=45h, TP=46h, SS=41h, TN=45h（XGBoost-SHAP分析）
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
# SML: COD=46h, NH3-N=45h, TP=46h, SS=41h, TN=45h
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
# 模型微调（实时驯化 - 每次预测后立即微调）
# ==========================================
def calibrate_model(indicator, inlet_data, real_removal_rate):
    """用实测去除率实时微调模型（每次预测后立即执行）"""
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
                # 实时驯化：每次只用1个新样本做增量学习
                model.fit(X_scaled, [real_removal_rate])
                st.session_state.calibration_count += 1
                return True, f"✅ {indicator} 实时驯化成功"
        return False, "模型不支持增量训练"
    except Exception as e:
        return False, f"❌ 驯化失败: {str(e)}"

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
# 完整诊断函数（大幅细化）
# ==========================================
def diagnose_system(inlet, outlet, pac, carbon, mlss, do):
    diagnoses = []

    # ====== 1. 进水异常诊断（细化） ======

    # --- COD ---
    cod_in = inlet.get('COD', 0)
    if cod_in > 500:
        diagnoses.append({
            'level': 'critical', 'indicator': '进水COD',
            'current': f"{cod_in:.0f} mg/L",
            'title': '🚨 进水COD严重超标（>500 mg/L）',
            'reasons': [
                '工业废水偷排（电镀/印染/食品加工等含高浓度有机物废水）',
                '管网沉积物因降雨冲刷大量释放',
                '污泥厌氧消化液回流冲击',
                '上游管网破损导致渗漏污染'
            ],
            'actions': [
                '✅ 立即增加碳源投加量30-40%（当前 {:.0f}→{:.0f} mg/L）'.format(carbon, carbon*1.4),
                '✅ 提高好氧段DO至3.0-3.5 mg/L',
                '✅ 降低进水量15-20%（减轻水力负荷）',
                '✅ 加强初沉池排泥频率（防止有机物在初沉池厌氧发酵）',
                '✅ 取样送检排查上游偷排源头'
            ]
        })
    elif cod_in > 400:
        diagnoses.append({
            'level': 'warning', 'indicator': '进水COD',
            'current': f"{cod_in:.0f} mg/L",
            'title': '⚠️ 进水COD偏高（400-500 mg/L）',
            'reasons': [
                '工业废水间歇性排放冲击',
                '管网沉积物释放',
                '前段预处理效果不佳（格栅/沉砂池）'
            ],
            'actions': [
                '✅ 增加碳源投加量20%（{:.0f}→{:.0f} mg/L）'.format(carbon, carbon*1.2),
                '✅ 提高DO至2.5-3.0 mg/L',
                '✅ 检查格栅和沉砂池是否堵塞'
            ]
        })
    elif cod_in < 100 and cod_in > 0:
        diagnoses.append({
            'level': 'info', 'indicator': '进水COD',
            'current': f"{cod_in:.0f} mg/L",
            'title': 'ℹ️ 进水COD偏低（<100 mg/L）',
            'reasons': [
                '雨水稀释（降雨导致管网溢流）',
                '上游截流导致进水浓度被稀释',
                '污水处理厂进水中混入大量清水'
            ],
            'actions': [
                '✅ 减少碳源投加量20-30%（{:.0f}→{:.0f} mg/L）'.format(carbon, carbon*0.75),
                '✅ 适当降低曝气量（防止污泥自身氧化）',
                '✅ 考虑增加内回流比提高脱氮效率'
            ]
        })

    # --- NH3-N ---
    nh3_in = inlet.get('NH3-N', 0)
    if nh3_in > 45:
        diagnoses.append({
            'level': 'critical', 'indicator': '进水NH₃-N',
            'current': f"{nh3_in:.1f} mg/L",
            'title': '🚨 进水NH₃-N严重超标（>45 mg/L）',
            'reasons': [
                '工业废水偷排（化工/制药/化肥行业高氨氮废水）',
                '污泥厌氧消化液回流（消化液氨氮浓度可达500-1000 mg/L）',
                '硝化菌活性受抑制（低温/有毒物质冲击）',
                '进水pH偏低（硝化反应消耗碱度）'
            ],
            'actions': [
                '✅ 立即提高DO至3.5-4.0 mg/L',
                '✅ 补充NaHCO₃ 80-100 mg/L（维持碱度≥100 mg/L）',
                '✅ 延长污泥龄（SRT）至15天以上（保证硝化菌富集）',
                '✅ 排查是否有高氨氮工业废水偷排',
                '✅ 若硝化受抑制，可投加硝化菌种（生物增效）'
            ]
        })
    elif nh3_in > 35:
        diagnoses.append({
            'level': 'warning', 'indicator': '进水NH₃-N',
            'current': f"{nh3_in:.1f} mg/L",
            'title': '⚠️ 进水NH₃-N偏高（35-45 mg/L）',
            'reasons': [
                '上游工业废水氨氮浓度波动',
                '硝化菌活性受抑制（温度<15℃或DO不足）',
                '碱度不足（硝化消耗碱度，pH下降抑制硝化）'
            ],
            'actions': [
                '✅ 提高DO至3.0-3.5 mg/L',
                '✅ 补充碱度50-80 mg/L（NaHCO₃或石灰）',
                '✅ 检测进水pH和碱度'
            ]
        })

    # --- TP ---
    tp_in = inlet.get('TP', 0)
    if tp_in > 7.0:
        diagnoses.append({
            'level': 'critical', 'indicator': '进水TP',
            'current': f"{tp_in:.2f} mg/L",
            'title': '🚨 进水TP严重超标（>7.0 mg/L）',
            'reasons': [
                '工业废水偷排（含磷清洗剂/化肥行业废水）',
                '污泥厌氧段释磷（内回流携带高浓度磷）',
                'PAC加药泵故障导致化学除磷失效',
                'pH异常影响PAC混凝效果（最佳pH 6.5-7.5）'
            ],
            'actions': [
                '✅ 立即增加PAC投加量40-50%（{:.0f}→{:.0f} mg/L）'.format(pac, pac*1.5),
                '✅ 检查pH，若<6.5投加碱调节至7.0-7.5',
                '✅ 增加排泥量（生物除磷主要靠排泥）',
                '✅ 检查PAC加药泵和管道是否堵塞'
            ]
        })
    elif tp_in > 5.0:
        diagnoses.append({
            'level': 'warning', 'indicator': '进水TP',
            'current': f"{tp_in:.2f} mg/L",
            'title': '⚠️ 进水TP偏高（5.0-7.0 mg/L）',
            'reasons': [
                '上游含磷废水浓度波动',
                'PAC投加量相对不足',
                '生物除磷效果下降（污泥龄过长）
