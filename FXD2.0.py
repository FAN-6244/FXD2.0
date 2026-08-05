# -*- coding: utf-8 -*-
"""
FXD3.0 - 水质净化厂智能预警与调控决策系统 v3.0
================================================
合并 版面.txt 的完整布局 + 去除率模型的数据逻辑
四种输入模式 · 永久记忆(Supabase) · 时序决策 · 异常诊断

预测目标：去除率(%) = (进水浓度 - 出水浓度) / 进水浓度 × 100%
⭐ 系统记忆长度（SML）【论文结论 v7.0】：
   NH3-N = 1h (XGBoost, R²=0.8961)
   TP    = 1h (RandomForest, R²=0.9011)
   TN    = 9h (TreeConsensus, R²=0.5312)
   COD   = 不适用 (XGBoost, R²=0.4545, 去除冗余)
   SS    = 不适用 (XGBoost, R²=0.1976, 物理沉淀主导)
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
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        text-align: center;
        border-top: 3px solid #ccc;
    }
    .channel-item .ch-name { font-weight: 600; font-size: 13px; }
    .channel-item .ch-value { font-size: 18px; font-weight: 700; margin: 2px 0; }
    .channel-item .ch-desc { font-size: 11px; color: #666; }
    .channel-fast .ch-name { color: #27AE60; }
    .channel-slow .ch-name { color: #F39C12; }
    .channel-special .ch-name { color: #E74C3C; }
    .channel-na .ch-name { color: #95A5A6; }
    .channel-fast { border-top-color: #27AE60; }
    .channel-slow { border-top-color: #F39C12; }
    .channel-special { border-top-color: #E74C3C; }
    .channel-na { border-top-color: #95A5A6; }
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
# ⭐ 系统记忆长度（论文结论 v7.0）
# NH3-N=1h, TP=1h, TN=9h, COD=不适用, SS=不适用
# ==========================================
MEMORY = {
    'COD':   {'hours': None, 'channel': '不适用', 'freq': 'N/A', 'description': '去除率几乎恒定（R²=0.45），无显著记忆效应'},
    'NH3-N': {'hours': 1, 'channel': '极快', 'freq': '<2h', 'description': '硝化反应，DO敏感，响应极快（XGBoost R²=0.896）'},
    'TP':    {'hours': 1, 'channel': '极快', 'freq': '<2h', 'description': '化学除磷(PAC)瞬时反应，响应极快（RF R²=0.901）'},
    'TN':    {'hours': 9, 'channel': '中速', 'freq': '6-12h', 'description': '反硝化过程，碳源投加延迟（TreeConsensus R²=0.531）'},
    'SS':    {'hours': None, 'channel': '不适用', 'freq': 'N/A', 'description': '物理沉淀主导（R²=0.198），建议实时阈值报警'}
}

# 三通道分组（按论文结论重组）
CHANNELS = {
    'fast': {
        'name': '⚡ 极快通道 (1h)',
        'color': '#27AE60',
        'indicators': ['NH3-N', 'TP'],
        'desc': 'NH₃-N (1h) · TP (1h) | 响应 <2h',
        'consensus': '✅ 高精度模型（XGBoost/RF），适合时序预警'
    },
    'medium': {
        'name': '🔄 中速通道 (9h)',
        'color': '#2E86AB',
        'indicators': ['TN'],
        'desc': 'TN (9h) | 更新 6-12h',
        'consensus': '⚠️ 碳源管理关键指标，需结合投加策略'
    },
    'na': {
        'name': '⚪ 非记忆通道 (N/A)',
        'color': '#95A5A6',
        'indicators': ['COD', 'SS'],
        'desc': 'COD/SS 无显著记忆 | R²偏低',
        'consensus': '🔴 建议使用实时阈值报警替代时序预测'
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
# 模型微调（实时驯化 - 每次预测后立即执行）
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
# 完整诊断函数（大幅细化，含具体原因与措施）
# ==========================================
def diagnose_system(inlet, outlet, pac, carbon, mlss, do):
    diagnoses = []

    # ====== 进水异常诊断（细化） ======

    # --- COD ---
    cod_in = inlet.get('COD', 0)
    if cod_in > 500:
        diagnoses.append({
            'level': 'critical', 'indicator': '进水COD',
            'current': f"{cod_in:.0f} mg/L",
            'title': '🚨 进水COD严重超标（>500 mg/L）',
            'reasons': [
                '工业废水偷排（电镀/印染/食品加工等含高浓度有机物废水）',
                '管网沉积物因暴雨冲刷大量释放',
                '污泥厌氧消化液回流冲击（COD可达1000+）',
                '上游管网破损导致渗漏污染',
                '初沉池排泥不及时，污泥厌氧发酵产生高浓度COD'
            ],
            'actions': [
                f'✅ 立即增加碳源投加量30-40%（当前 {carbon:.0f} → {carbon*1.4:.0f} mg/L）',
                '✅ 提高好氧段DO至3.0-3.5 mg/L（当前 {:.1f}）'.format(do),
                '✅ 降低进水量15-20%（减轻水力负荷）',
                '✅ 加强初沉池排泥频率（防止有机物在初沉池厌氧发酵）',
                '✅ 取样送检排查上游偷排源头，并与环保部门联动'
            ]
        })
    elif cod_in > 400:
        diagnoses.append({
            'level': 'warning', 'indicator': '进水COD',
            'current': f"{cod_in:.0f} mg/L",
            'title': '⚠️ 进水COD偏高（400-500 mg/L）',
            'reasons': [
                '工业废水间歇性排放冲击',
                '管网沉积物释放（雨后常见）',
                '前段预处理效果不佳（格栅/沉砂池堵塞或效率低）'
            ],
            'actions': [
                f'✅ 增加碳源投加量20%（{carbon:.0f} → {carbon*1.2:.0f} mg/L）',
                f'✅ 提高DO至2.5-3.0 mg/L（当前 {do:.1f}）',
                '✅ 检查格栅和沉砂池是否堵塞，及时清理'
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
                '污水处理厂进水中混入大量清水（如冷却水）'
            ],
            'actions': [
                f'✅ 减少碳源投加量20-30%（{carbon:.0f} → {carbon*0.75:.0f} mg/L）',
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
                '硝化菌活性受抑制（低温<15℃或DO不足）',
                '进水pH偏低（硝化反应消耗碱度，pH<6.5抑制硝化）',
                '污泥龄过短（SRT<8天，硝化菌无法富集）'
            ],
            'actions': [
                f'✅ 立即提高DO至3.5-4.0 mg/L（当前 {do:.1f}）',
                '✅ 补充NaHCO₃ 80-100 mg/L（维持碱度≥100 mg/L）',
                '✅ 延长污泥龄（SRT）至15天以上（保证硝化菌富集）',
                '✅ 排查是否有高氨氮工业废水偷排，取样送检',
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
                f'✅ 提高DO至3.0-3.5 mg/L（当前 {do:.1f}）',
                '✅ 补充碱度50-80 mg/L（NaHCO₃或石灰）',
                '✅ 检测进水pH和碱度，确保pH 7.0-8.0'
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
                'PAC加药泵故障或管道堵塞导致化学除磷失效',
                'pH异常影响PAC混凝效果（最佳pH 6.5-7.5）',
                '生物除磷效果差（污泥龄过长或碳源不足）'
            ],
            'actions': [
                f'✅ 立即增加PAC投加量40-50%（{pac:.0f} → {pac*1.5:.0f} mg/L）',
                '✅ 检查pH，若<6.5投加碱调节至7.0-7.5',
                '✅ 增加排泥量（生物除磷主要靠排泥）',
                '✅ 检查PAC加药泵和管道是否堵塞，确保药剂正常投加',
                '✅ 若碳源不足，适当补充碳源以强化生物除磷'
            ]
        })
    elif tp_in > 5.0:
        diagnoses.append({
            'level': 'warning', 'indicator': '进水TP',
            'current': f"{tp_in:.2f} mg/L",
            'title': '⚠️ 进水TP偏高（5.0-7.0 mg/L）',
            'reasons': [
                '上游含磷废水浓度波动',
                'PAC投加量相对不足（未及时根据进水调整）',
                '生物除磷效果下降（污泥龄过长或DO过高抑制释磷）'
            ],
            'actions': [
                f'✅ 增加PAC投加量20-30%（{pac:.0f} → {pac*1.25:.0f} mg/L）',
                '✅ 检查pH并调节至6.5-7.5',
                '✅ 增加排泥，控制污泥龄在8-12天'
            ]
        })

    # --- SS ---
    ss_in = inlet.get('SS', 0)
    if ss_in > 350:
        diagnoses.append({
            'level': 'warning', 'indicator': '进水SS',
            'current': f"{ss_in:.0f} mg/L",
            'title': '⚠️ 进水SS严重偏高（>350 mg/L）',
            'reasons': [
                '管网冲刷（暴雨后管网沉积物冲入）',
                '初沉池运行异常（排泥不及时或刮泥机故障）',
                '上游施工或管道破损导致泥沙进入'
            ],
            'actions': [
                '✅ 增加初沉池排泥频率（至少每班排泥一次）',
                '✅ 投加PAM（聚丙烯酰胺）0.5-1.0 mg/L增强絮凝',
                '✅ 检查初沉池刮泥机和排泥泵是否正常'
            ]
        })

    # --- TN ---
    tn_in = inlet.get('TN', 0)
    if tn_in > 50:
        diagnoses.append({
            'level': 'critical', 'indicator': '进水TN',
            'current': f"{tn_in:.1f} mg/L",
            'title': '🚨 进水TN严重超标（>50 mg/L）',
            'reasons': [
                '工业废水含大量有机氮或氨氮',
                '回流污泥携带硝酸盐（内回流比过大）',
                '碳源不足导致反硝化受限',
                '缺氧段停留时间不足（HRT<2h）'
            ],
            'actions': [
                f'✅ 增加碳源投加量30-40%（{carbon:.0f} → {carbon*1.35:.0f} mg/L）',
                '✅ 检查内外回流比，调整内回流比至200-300%',
                '✅ 提高缺氧段容积利用率（搅拌器是否正常运行）',
                '✅ 若进水TN持续高，考虑增设碳源投加点（分段投加）'
            ]
        })
    elif tn_in > 35:
        diagnoses.append({
            'level': 'warning', 'indicator': '进水TN',
            'current': f"{tn_in:.1f} mg/L",
            'title': '⚠️ 进水TN偏高（35-50 mg/L）',
            'reasons': [
                '上游工业废水氮负荷波动',
                '碳源投加量不足或分配不均',
                '缺氧段DO偏高（>0.5mg/L）抑制反硝化'
            ],
            'actions': [
                f'✅ 增加碳源投加量20%（{carbon:.0f} → {carbon*1.2:.0f} mg/L）',
                '✅ 检查缺氧段DO，若>0.5mg/L则降低曝气或搅拌强度',
                '✅ 优化碳源投加点（缺氧段前端）'
            ]
        })

    # ====== 出水超标（更细的原因与措施） ======
    for ind in ['COD', 'NH3-N', 'TP', 'TN', 'SS']:
        if outlet.get(ind, 0) > DESIGN_LIMITS[ind]['value']:
            limit = DESIGN_LIMITS[ind]['value']
            val = outlet[ind]
            level = 'critical' if val > limit * 1.5 else 'warning'
            icon = '🚨' if level == 'critical' else '⚠️'
            # 根据指标定制详细原因
            reasons_map = {
                'COD': [
                    f'进水COD过高（{inlet.get("COD",0):.0f} mg/L）超出系统负荷',
                    f'好氧段DO不足（{do:.1f} mg/L），有机物降解不充分',
                    '污泥浓度（MLSS）偏低（{:.0f} mg/L）或污泥活性差'.format(mlss),
                    '碳源投加量不足（{:.0f} mg/L）导致共代谢作用弱'.format(carbon)
                ],
                'NH3-N': [
                    f'进水NH₃-N过高（{inlet.get("NH3-N",0):.1f} mg/L）',
                    f'好氧段DO不足（{do:.1f} mg/L），硝化受限',
                    f'碱度不足（硝化消耗碱度，pH可能偏低）',
                    f'污泥龄（SRT）过短（<10天），硝化菌流失'
                ],
                'TP': [
                    f'进水TP过高（{inlet.get("TP",0):.2f} mg/L）',
                    f'PAC投加量不足（{pac:.0f} mg/L）或pH不适（最佳6.5-7.5）',
                    '排泥量不足，生物除磷效果差',
                    '厌氧段释磷不充分（DO偏高或回流比不当）'
                ],
                'TN': [
                    f'进水TN过高（{inlet.get("TN",0):.1f} mg/L）',
                    f'碳源投加量不足（{carbon:.0f} mg/L），反硝化受限',
                    '缺氧段DO偏高（>0.5mg/L），抑制反硝化菌',
                    '内回流比过大（>400%）或过小（<100%）'
                ],
                'SS': [
                    f'进水SS过高（{inlet.get("SS",0):.0f} mg/L）',
                    '二沉池泥层过厚，刮泥机运行异常',
                    '污泥沉降性能差（SVI>150）',
                    '混凝剂（PAC/PAM）投加不足'
                ]
            }
            actions_map = {
                'COD': [
                    f'增加碳源投加量 {carbon:.0f}→{carbon*1.25:.0f} mg/L',
                    f'提高好氧段DO至2.5-3.0 mg/L（当前 {do:.1f}）',
                    '检查污泥活性，若MLSS<2500则减少排泥',
                    '增加曝气量，保证好氧段HRT充足'
                ],
                'NH3-N': [
                    f'提高DO至3.0-3.5 mg/L（当前 {do:.1f}）',
                    '补充NaHCO₃ 50-80 mg/L以维持碱度',
                    '延长SRT至15天以上（减少排泥）',
                    '检查温度，若<15℃需提高DO或补充硝化菌'
                ],
                'TP': [
                    f'增加PAC投加量 {pac:.0f}→{pac*1.4:.0f} mg/L',
                    '调整投加点至厌氧段末端或好氧段前端',
                    '增加排泥量，控制泥龄8-12天',
                    '检查pH，若<6.5投加石灰调节'
                ],
                'TN': [
                    f'增加碳源投加量 {carbon:.0f}→{carbon*1.3:.0f} mg/L',
                    '调整内回流比至200-300%',
                    '提高缺氧段容积利用率，检查搅拌器',
                    '控制缺氧段DO<0.5 mg/L'
                ],
                'SS': [
                    '增加排泥量20%，降低二沉池泥层高度',
                    '投加PAM 0.5-1.0 mg/L强化絮凝',
                    '降低进水量10-15%',
                    '检查二沉池刮泥机运行状态'
                ]
            }
            diagnoses.append({
                'level': level,
                'indicator': f'出水{ind}',
                'current': f"{val:.2f} mg/L",
                'title': f'{icon} 出水{ind}超标（限值≤{limit} mg/L）',
                'reasons': reasons_map[ind],
                'actions': actions_map[ind]
            })

    # ====== 运行参数异常（细化） ======
    if do < 0.8:
        diagnoses.append({
            'level': 'critical', 'indicator': '溶解氧DO',
            'current': f"{do:.1f} mg/L",
            'title': '🚨 好氧段DO严重不足（<0.8 mg/L）',
            'reasons': [
                '曝气设备故障（风机跳闸/曝气头堵塞）',
                '进水负荷突增（COD/氨氮大幅上升）',
                'DO仪探头校准偏差或结垢'
            ],
            'actions': [
                '✅ 立即检查曝气设备，确认风机运行状态',
                '✅ 加大风机风量20-30%或启动备用风机',
                '✅ 清理DO仪探头并重新校准',
                '✅ 若负荷过高，临时降低进水量'
            ]
        })
    elif do < 1.5:
        diagnoses.append({
            'level': 'warning', 'indicator': '溶解氧DO',
            'current': f"{do:.1f} mg/L",
            'title': '⚠️ 好氧段DO偏低（<1.5 mg/L）',
            'reasons': [
                '曝气量不足（风机频率偏低）',
                '进水负荷增加导致耗氧量上升',
                '水温升高导致饱和DO下降'
            ],
            'actions': [
                '✅ 增加曝气量10-20%',
                '✅ 监测DO变化趋势，每半小时记录一次',
                '✅ 检查是否有机负荷冲击'
            ]
        })

    if mlss < 2500:
        diagnoses.append({
            'level': 'warning', 'indicator': '污泥浓度MLSS',
            'current': f"{mlss:.0f} mg/L",
            'title': '⚠️ 污泥浓度偏低（<2500 mg/L）',
            'reasons': [
                '排泥过量或排泥频率过高',
                '进水负荷过低导致污泥自身氧化',
                '污泥沉降性能差导致二沉池跑泥',
                '回流泵故障导致污泥回流不足'
            ],
            'actions': [
                '✅ 减少排泥量或延长排泥间隔',
                '✅ 增加污泥回流量20-30%',
                '✅ 检查二沉池泥位，防止跑泥',
                '✅ 检测污泥沉降比（SV30），若>80%则可能有膨胀'
            ]
        })
    elif mlss > 6000:
        diagnoses.append({
            'level': 'info', 'indicator': '污泥浓度MLSS',
            'current': f"{mlss:.0f} mg/L",
            'title': 'ℹ️ 污泥浓度偏高（>6000 mg/L）',
            'reasons': [
                '排泥不足导致污泥积累',
                '二沉池泥层过厚，污泥停留时间过长',
                '进水SS过高导致污泥增量'
            ],
            'actions': [
                '✅ 增加排泥量20-30%',
                '✅ 检查二沉池泥位，必要时加强刮泥',
                '✅ 调整污泥回流量，避免污泥在系统内过度累积'
            ]
        })

    if pac < 20:
        diagnoses.append({
            'level': 'warning', 'indicator': 'PAC投加量',
            'current': f"{pac:.0f} mg/L",
            'title': '⚠️ PAC投加量偏低（<20 mg/L）',
            'reasons': [
                'PAC储备不足或药液浓度偏低',
                '加药泵故障或管道堵塞',
                '进水TP浓度上升但未及时调整投加量'
            ],
            'actions': [
                f'✅ 增加PAC至30-50 mg/L（当前 {pac:.0f}）',
                '✅ 检查加药泵运行状态和管道通畅性',
                '✅ 检测PAC药液有效浓度，必要时更换药剂'
            ]
        })

    if carbon < 30:
        diagnoses.append({
            'level': 'warning', 'indicator': '碳源投加量',
            'current': f"{carbon:.0f} mg/L",
            'title': '⚠️ 碳源投加量偏低（<30 mg/L）',
            'reasons': [
                '碳源储备不足或输送系统故障',
                '进水C/N比偏低（COD/TN<5）',
                '反硝化碳源缺乏，TN去除率下降'
            ],
            'actions': [
                f'✅ 增加碳源至40-60 mg/L（当前 {carbon:.0f}）',
                '✅ 检查碳源储罐液位和计量泵',
                '✅ 评估进水C/N比，若<5则需补充碳源至C/N≥5'
            ]
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

# 核心预测必需的列（用于文件上传解析）
CORE_REQUIRED_COLS = ['COD', 'NH3-N', 'TP', 'TN', 'SS', '流量', 'MLSS']
# 模板表头（包含序号、时间）
TEMPLATE_COLS = ['序号', '时间', 'COD', 'NH3-N', 'TP', 'TN', 'SS', '流量', 'MLSS']

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

# --- 2. 文件上传（模板已更新） ---
elif input_mode_global == "📁 文件上传":
    st.sidebar.markdown("### 📁 上传数据文件")
    st.sidebar.caption("请上传包含以下列的 Excel/CSV 文件：")
    st.sidebar.code("序号, 时间, COD, NH3-N, TP, TN, SS, 流量, MLSS", language='text')

    if st.sidebar.button("📥 下载空模板 (Excel)"):
        template_df = pd.DataFrame(columns=TEMPLATE_COLS)
        template_df.loc[0] = [1, "2026-08-01 00:00", 200, 20, 3.0, 30, 150, 10000, 4000]
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
            
            # 检查核心列是否存在
            missing_cols = set(CORE_REQUIRED_COLS) - set(df_upload.columns)
            if missing_cols:
                st.sidebar.error(f"❌ 缺少必需列：{missing_cols}")
                input_data = None
            else:
                row = df_upload.iloc[0]
                cod_in = row['COD']
                nh3_in = row['NH3-N']
                tp_in = row['TP']
                tn_in = row.get('TN', 30)
                ss_in = row['SS']
                flow_in = row['流量']
                mlss = row['MLSS']
                
                # 可选运行参数：若文件中无此列则使用默认值
                if 'PAC' in df_upload.columns:
                    pac = row['PAC']
                else:
                    pac = 30.0
                    st.sidebar.info("ℹ️ 未检测到'PAC'列，使用默认值 30 mg/L")
                
                if '碳源' in df_upload.columns:
                    carbon = row['碳源']
                else:
                    carbon = 50.0
                    st.sidebar.info("ℹ️ 未检测到'碳源'列，使用默认值 50 mg/L")
                
                if 'DO' in df_upload.columns:
                    do = row['DO']
                else:
                    do = 2.0
                    st.sidebar.info("ℹ️ 未检测到'DO'列，使用默认值 2.0 mg/L")
                
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

    # ---- 自动模式处理（实时驯化） ----
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

        # 实时驯化：每次预测后立即微调所有指标
        for ind in ['COD', 'NH3-N', 'TP', 'TN', 'SS']:
            calibrate_model(ind, input_data, real_removal[ind])
        # 保存到Supabase（可选）
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
        st.info(f"🔄 实时数据流运行中... 已接收 {st.session_state.simulation_counter} 组数据 | 已实时驯化 {st.session_state.calibration_count} 次")
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

    # ================================================================
    # 2. 趋势图（近24小时）—— 每条曲线注释精确对齐Y轴
    # ================================================================
    st.markdown('<div class="section-header">📈 进出水趋势（近24小时）</div>', unsafe_allow_html=True)
    st.caption("🟦 实线 = 实测/进水 | 虚线 = 预测 | 各指标颜色区分，标签位于曲线末端右侧，Y轴精确对齐")

    recent_data = st.session_state.data_buffer.get_recent(24)
    if len(recent_data) > 1:
        df_trend = pd.DataFrame([{
            'timestamp': d['timestamp'],
            'inlet_COD': d['inlet']['COD'],
            'inlet_NH3': d['inlet']['NH3-N'],
            'inlet_TP': d['inlet']['TP'],
            'inlet_TN': d['inlet']['TN'],
            'inlet_SS': d['inlet']['SS'],
            'outlet_COD_real': d['outlet']['COD'] if d['outlet'] else None,
            'outlet_COD_pred': d['pred_outlet']['COD'] if d['pred_outlet'] else None,
            'outlet_NH3_real': d['outlet']['NH3-N'] if d['outlet'] else None,
            'outlet_NH3_pred': d['pred_outlet']['NH3-N'] if d['pred_outlet'] else None,
            'outlet_TP_real': d['outlet']['TP'] if d['outlet'] else None,
            'outlet_TP_pred': d['pred_outlet']['TP'] if d['pred_outlet'] else None,
            'outlet_TN_real': d['outlet']['TN'] if d['outlet'] else None,
            'outlet_TN_pred': d['pred_outlet']['TN'] if d['pred_outlet'] else None,
            'outlet_SS_real': d['outlet']['SS'] if d['outlet'] else None,
            'outlet_SS_pred': d['pred_outlet']['SS'] if d['pred_outlet'] else None,
        } for d in recent_data])

        subplot_titles = ('COD', 'NH₃-N', 'TP', 'TN', 'SS')
        fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            subplot_titles=subplot_titles)

        indicators_config = [
            ('COD', '#E74C3C', '#2E86AB', '#2E86AB', DESIGN_LIMITS['COD']['value']),
            ('NH3', '#F39C12', '#27AE60', '#27AE60', DESIGN_LIMITS['NH3-N']['value']),
            ('TP', '#8E44AD', '#F39C12', '#F39C12', DESIGN_LIMITS['TP']['value']),
            ('TN', '#1F77B4', '#FF7F0E', '#FF7F0E', DESIGN_LIMITS['TN']['value']),
            ('SS', '#9467BD', '#D62728', '#D62728', DESIGN_LIMITS['SS']['value'])
        ]

        # 添加所有曲线，同时记录每条曲线所属子图行号及其末端点
        trace_data = []
        for row, (prefix, in_color, real_color, pred_color, limit) in enumerate(indicators_config, start=1):
            # 进水
            fig.add_trace(
                go.Scatter(x=df_trend['timestamp'], y=df_trend[f'inlet_{prefix}'],
                           name=f'进水 {prefix}', line=dict(color=in_color, width=2)),
                row=row, col=1
            )
            _last = df_trend[f'inlet_{prefix}'].last_valid_index()
            if _last is not None:
                trace_data.append({
                    'name': f'进水 {prefix}', 'row': row,
                    'x': df_trend.loc[_last, 'timestamp'], 'y': df_trend.loc[_last, f'inlet_{prefix}'],
                    'color': in_color
                })
            # 出水实测
            mask_real = df_trend[f'outlet_{prefix}_real'].notna()
            if mask_real.any():
                fig.add_trace(
                    go.Scatter(x=df_trend[mask_real]['timestamp'], y=df_trend[mask_real][f'outlet_{prefix}_real'],
                               name=f'出水 {prefix} 实测', line=dict(color=real_color, width=2.5)),
                    row=row, col=1
                )
                _last = df_trend.loc[mask_real, f'outlet_{prefix}_real'].last_valid_index()
                if _last is not None:
                    trace_data.append({
                        'name': f'出水 {prefix} 实测', 'row': row,
                        'x': df_trend.loc[_last, 'timestamp'], 'y': df_trend.loc[_last, f'outlet_{prefix}_real'],
                        'color': real_color
                    })
            # 出水预测
            mask_pred = df_trend[f'outlet_{prefix}_pred'].notna()
            if mask_pred.any():
                fig.add_trace(
                    go.Scatter(x=df_trend[mask_pred]['timestamp'], y=df_trend[mask_pred][f'outlet_{prefix}_pred'],
                               name=f'出水 {prefix} 预测', line=dict(color=pred_color, width=2, dash='dot')),
                    row=row, col=1
                )
                _last = df_trend.loc[mask_pred, f'outlet_{prefix}_pred'].last_valid_index()
                if _last is not None:
                    trace_data.append({
                        'name': f'出水 {prefix} 预测', 'row': row,
                        'x': df_trend.loc[_last, 'timestamp'], 'y': df_trend.loc[_last, f'outlet_{prefix}_pred'],
                        'color': pred_color
                    })
            fig.add_hline(y=limit, line_dash="dash", line_color="red", row=row, col=1)

        # ----- 为每条曲线添加末端标签 -----
        # 关键修复：通过 xref/yref 将每个标签绑定到其曲线所属子图的坐标轴。
        # 原代码未指定 yref，所有标签都被画到第1个子图(COD)的Y轴上，
        # 导致 NH3-N/TP/TN/SS 的标签 y 值被按 COD 量程(0~300)解析，与曲线严重错位。
        # 现按 trace 所属 row 绑定 y1..y5，标签 y 与曲线末端精确对齐。
        # 同一子图内多条曲线末端 y 过近时(如出水实测与预测)，给标签少量垂直像素偏移避免重叠。
        if trace_data:
            _t_min = df_trend['timestamp'].min()
            _t_max = df_trend['timestamp'].max()
            _span = _t_max - _t_min
            if hasattr(_span, 'total_seconds') and _span.total_seconds() > 0:
                _offset_delta = timedelta(seconds=_span.total_seconds() * 0.02)
            else:
                _offset_delta = timedelta(minutes=2)
            # 适当扩展X轴范围，确保末端标签不被右侧裁剪
            _pad = _span * 0.10 if (hasattr(_span, 'total_seconds') and _span.total_seconds() > 0) else timedelta(minutes=5)
            try:
                fig.update_xaxes(range=[_t_min, _t_max + _pad])
            except Exception:
                pass
            # 按子图分组，组内按添加顺序分配少量垂直偏移，避免实测/预测标签重叠
            _by_row = {}
            for _idx, _p in enumerate(trace_data):
                _by_row.setdefault(_p['row'], []).append(_idx)
            _shifts = [0, 12, -12, 24, -24]
            _yshift_map = {}
            for _r, _idxs in _by_row.items():
                for _pos, _i in enumerate(_idxs):
                    _yshift_map[_i] = _shifts[_pos] if _pos < len(_shifts) else 0
            for _i, _p in enumerate(trace_data):
                _xref = 'x' if _p['row'] == 1 else f'x{_p["row"]}'
                _yref = 'y' if _p['row'] == 1 else f'y{_p["row"]}'
                _x_pos = _p['x']
                if hasattr(_x_pos, 'to_pydatetime'):
                    _x_pos = _x_pos.to_pydatetime()
                fig.add_annotation(
                    x=_x_pos + _offset_delta,
                    y=_p['y'],
                    xref=_xref,
                    yref=_yref,
                    text=_p['name'],
                    showarrow=False,
                    yshift=_yshift_map.get(_i, 0),
                    font=dict(size=9, color=_p['color']),
                    xanchor='left'
                )

        fig.update_layout(height=650, showlegend=False, hovermode='x unified')
        fig.update_xaxes(title_text="时间（北京时间）", row=5, col=1)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📭 数据收集中... 请等待更多数据点（至少2个时间点）")

    # ================================================================
    # 3. 记忆长度与分频调控
    # ================================================================
    st.markdown('<div class="section-header">🧠 系统记忆长度（SML）与分频调控策略</div>', unsafe_allow_html=True)
    st.caption("💡 基于去除率模型 XGBoost-SHAP 分析（论文结论 v7.0）：NH₃-N=1h · TP=1h · TN=9h · COD/SS=不适用")

    col_ch1, col_ch2, col_ch3 = st.columns(3)
    with col_ch1:
        st.markdown(f"""
        <div class="channel-item channel-fast">
            <div class="ch-name">{CHANNELS['fast']['name']}</div>
            <div class="ch-value" style="color:{CHANNELS['fast']['color']};">1h</div>
            <div class="ch-desc">{CHANNELS['fast']['desc']}</div>
            <div class="ch-desc">{CHANNELS['fast']['consensus']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_ch2:
        st.markdown(f"""
        <div class="channel-item channel-slow">
            <div class="ch-name">{CHANNELS['medium']['name']}</div>
            <div class="ch-value" style="color:{CHANNELS['medium']['color']};">9h</div>
            <div class="ch-desc">{CHANNELS['medium']['desc']}</div>
            <div class="ch-desc">{CHANNELS['medium']['consensus']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_ch3:
        st.markdown(f"""
        <div class="channel-item channel-na">
            <div class="ch-name">{CHANNELS['na']['name']}</div>
            <div class="ch-value" style="color:{CHANNELS['na']['color']};">N/A</div>
            <div class="ch-desc">{CHANNELS['na']['desc']}</div>
            <div class="ch-desc">{CHANNELS['na']['consensus']}</div>
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

    if mem is not None and mem > 0:
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
                (0, "🚨 记录进水NH₃-N异常值，启动应急响应（记忆长度1h，响应极快）"),
                (0.5, "⚙️ 立即提高好氧段DO至3.0-3.5 mg/L"),
                (1, "📊 评估出水NH₃-N变化趋势（1h内可见效）"),
                (2, "✅ 确认NH₃-N稳定达标，逐步回调")
            ]
        elif indicator_select == 'TP':
            steps = [
                (0, "🚨 记录进水TP异常值，启动应急响应（记忆长度1h，PAC瞬时反应）"),
                (0.5, "⚙️ 立即增加PAC投加量30-40%"),
                (1, "📊 评估出水TP变化趋势（1h内可见效）"),
                (2, "✅ 确认TP稳定达标，逐步回调")
            ]
        elif indicator_select == 'TN':
            steps = [
                (0, "🚨 记录进水TN异常值，启动应急响应（记忆长度9h，碳源延迟）"),
                (2, "📞 通知值班长，确认碳源储备"),
                (4, "⚙️ 增加碳源投加量25-30%，检查内回流比"),
                (6, "🔍 检查缺氧段DO（<0.5mg/L）及搅拌器"),
                (9, "📊 评估出水TN变化趋势（9h达到峰值响应）"),
                (12, "✅ 确认TN稳定达标，逐步回调碳源")
            ]
        else:
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
    else:
        st.info(f"⚠️ **{indicator_select} 无显著记忆效应**（R²偏低，去除率几乎恒定/物理沉淀主导）")
        st.markdown(f"""
        <div style="background:#FFF3E0;border-radius:8px;padding:12px 16px;border-left:4px solid #F39C12;">
            <b>💡 推荐策略：实时阈值报警 + 即时干预</b><br>
            • 当前 {indicator_select} 出水值为 <b>{current_val:.2f} mg/L</b>（限值 {limit} mg/L）<br>
            • 去除率 {removal_rates.get(indicator_select, 0)*100:.1f}%<br>
            • 建议每 <b>1-2 小时</b> 取样检测一次<br>
            • 超标时立即触发声光报警并推送至值班手机<br>
            • 不依赖历史时序预测，以实测数据驱动调控
        </div>
        """, unsafe_allow_html=True)

    # ================================================================
    # 5. 异常诊断与工艺优化
    # ================================================================
    st.markdown('<div class="section-header">🔍 异常诊断与工艺优化建议</div>', unsafe_allow_html=True)
    st.caption("💡 基于同类型A²/O工艺经验库 + 当前工况多维度分析（细化版）")

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
            <div class="stat-label">🔄 模型实时驯化次数</div>
            <div class="stat-value">{st.session_state.calibration_count} 次</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stats3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">🧠 系统记忆长度（SML）</div>
            <div class="stat-value">NH₃-N=1h · TP=1h · TN=9h</div>
            <div class="stat-sub">COD/SS = 不适用（低R²）</div>
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
    5. 查看时序决策建议和细化异常诊断
    """)
    st.markdown("---")
    st.markdown("#### 🧠 系统记忆长度（SML）- 论文结论 v7.0")
    st.markdown("""
    | 指标 | 记忆长度 | 最佳模型 | R² | 工艺解释 |
    | :--- | :---: | :---: | :---: | :--- |
    | **NH₃-N** | **1 小时** | XGBoost | 0.896 | 硝化反应，DO敏感，响应极快 |
    | **TP** | **1 小时** | RandomForest | 0.901 | 化学除磷(PAC)瞬时反应 |
    | **TN** | **9 小时** | TreeConsensus | 0.531 | 反硝化，碳源投加延迟 |
    | **COD** | **不适用** | XGBoost | 0.455 | 去除率几乎恒定（去除冗余） |
    | **SS** | **不适用** | XGBoost | 0.198 | 物理沉淀主导（建议阈值报警） |
    """)
    st.caption(f"📌 基于去除率模型 · XGBoost-SHAP 分析 · {'模型已加载' if HAS_MODEL else '使用经验值（请运行模型训练脚本生成model_cache）'}")

# ==========================================
# 页脚
# ==========================================
st.markdown("---")
beijing_now = datetime.now(BEIJING_TZ)
st.caption(f"🏭 水质净化厂智能预警与调控决策系统 v3.0 | 去除率模型 | 四种输入模式 | {'永久记忆已启用' if SUPABASE_AVAILABLE else '永久记忆未启用'} | {beijing_now.strftime('%Y-%m-%d %H:%M')} 北京时间")
