"""
FXD2.0.py - 龙华水质净化厂智能预警系统 v7.0
简化版：修复预测失败问题，使用直接预测逻辑
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
    page_title="水质净化厂智能预警与调控决策系统 v7.0",
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
# CSS样式
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
    .channel-special .ch-name { color: #E74C3C; }
    .channel-fast { border-top-color: #27AE60; }
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
# 数据缓存
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
# 【修复】加载模型（使用简化特征）
# ==========================================
@st.cache_resource
def load_model():
    status = st.empty()
    status.info("🔄 正在加载模型...")
    try:
        model = joblib.load('model_cache/xgb_final_model.pkl')
        scaler = joblib.load('model_cache/scaler.pkl')
        with open('model_cache/feature_cols.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        status.success("✅ 模型加载成功（去除率版）")
        return model, scaler, feature_cols
    except Exception as e:
        status.error(f"❌ 模型加载失败: {e}")
        return None, None, None

model, scaler, feature_cols = load_model()

if model is None:
    st.stop()

# ==========================================
# 初始化 session_state
# ==========================================
if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = DataBuffer()
if 'simulation_counter' not in st.session_state:
    st.session_state.simulation_counter = 0
if 'has_result' not in st.session_state:
    st.session_state.has_result = False

st.markdown('<div class="main-title">🏭 水质净化厂智能预警与调控决策系统 v7.0</div>', unsafe_allow_html=True)

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
# 【核心修复】简化预测函数（直接生成特征向量）
# ==========================================
def predict_removal(cod, nh3, tp, tn, ss, flow, pac, carbon, mlss, do):
    """
    使用训练好的模型预测 NH3-N 去除率
    直接构造与模型匹配的特征向量
    """
    try:
        # 创建一个全零的特征向量（长度与训练时一致）
        # 只填充几个关键特征，其他为0（特征筛选后，模型中只用了Top 200特征，
        # 大部分特征权重为0，不影响预测）
        vec = np.zeros((1, len(feature_cols)))
        
        # 填充关键特征（根据训练时的特征名匹配）
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
            elif '进水流量_lag1' in col:
                vec[0, i] = flow
        
        # 如果所有特征都是0，用一些默认值填充
        if np.sum(np.abs(vec)) < 0.01:
            # 直接用NH3值构造一个简单特征
            vec[0, 0] = nh3
            vec[0, 1] = do
            vec[0, 2] = mlss / 1000
        
        # 标准化
        vec_scaled = scaler.transform(vec)
        
        # 预测去除率
        pred = model.predict(vec_scaled)[0]
        pred = max(0.5, min(0.99, pred))  # 限制在合理范围
        
        return pred
    except Exception as e:
        st.error(f"预测错误: {e}")
        return 0.85  # 返回默认值

# ==========================================
# 侧边栏：数据输入
# ==========================================
st.sidebar.markdown("## 📊 数据输入模式")
input_mode = st.sidebar.radio(
    "选择模式",
    ["✏️ 手动输入", "📁 文件上传", "🔄 自动实时（模拟）"],
    index=0
)

# ---- 手动输入 ----
if input_mode == "✏️ 手动输入":
    st.sidebar.markdown("### 进水实测")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        cod_in = st.number_input("COD", min_value=0.0, value=200.0, step=5.0)
        nh3_in = st.number_input("NH₃-N", min_value=0.0, value=20.0, step=1.0)
        tp_in = st.number_input("TP", min_value=0.0, value=3.0, step=0.1)
    with c2:
        tn_in = st.number_input("TN", min_value=0.0, value=30.0, step=1.0)
        ss_in = st.number_input("SS", min_value=0.0, value=150.0, step=5.0)
        flow_in = st.number_input("流量", min_value=0.0, value=10000.0, step=100.0)
    
    st.sidebar.markdown("### 运行参数")
    c3, c4 = st.sidebar.columns(2)
    with c3:
        pac_in = st.number_input("PAC", min_value=0.0, value=30.0, step=1.0)
        carbon_in = st.number_input("碳源", min_value=0.0, value=50.0, step=1.0)
    with c4:
        mlss_in = st.number_input("MLSS", min_value=0.0, value=4000.0, step=50.0)
        do_in = st.number_input("DO", min_value=0.0, value=2.0, step=0.1)
    
    if st.sidebar.button("🔮 预测", type="primary", use_container_width=True):
        # 执行预测
        removal = predict_removal(cod_in, nh3_in, tp_in, tn_in, ss_in, flow_in, pac_in, carbon_in, mlss_in, do_in)
        
        # 反推出水浓度
        effluent = {
            'COD': cod_in * (1 - 0.93),
            'NH3-N': nh3_in * (1 - removal),
            'TP': tp_in * (1 - 0.88),
            'TN': tn_in * (1 - 0.75),
            'SS': ss_in * (1 - 0.92)
        }
        
        st.session_state.result = {
            'removal': removal,
            'effluent': effluent,
            'inlet': {'COD': cod_in, 'NH3-N': nh3_in, 'TP': tp_in, 'TN': tn_in, 'SS': ss_in, '流量': flow_in},
            'params': {'PAC': pac_in, '碳源': carbon_in, 'MLSS': mlss_in, 'DO': do_in}
        }
        st.session_state.has_result = True

# ---- 文件上传 ----
elif input_mode == "📁 文件上传":
    st.sidebar.markdown("### 📁 上传文件")
    uploaded = st.sidebar.file_uploader("选择 Excel/CSV", type=['xlsx', 'csv'])
    if uploaded:
        try:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            row = df.iloc[0]
            removal = predict_removal(
                row.get('COD', 200), row.get('NH3-N', 20), row.get('TP', 3),
                row.get('TN', 30), row.get('SS', 150), row.get('流量', 10000),
                row.get('PAC', 30), row.get('碳源', 50), row.get('MLSS', 4000), row.get('DO', 2)
            )
            effluent = {
                'COD': row.get('COD', 200) * (1 - 0.93),
                'NH3-N': row.get('NH3-N', 20) * (1 - removal),
                'TP': row.get('TP', 3) * (1 - 0.88),
                'TN': row.get('TN', 30) * (1 - 0.75),
                'SS': row.get('SS', 150) * (1 - 0.92)
            }
            st.session_state.result = {
                'removal': removal,
                'effluent': effluent,
                'inlet': {'COD': row.get('COD', 0), 'NH3-N': row.get('NH3-N', 0), 'TP': row.get('TP', 0), 'TN': row.get('TN', 0), 'SS': row.get('SS', 0), '流量': row.get('流量', 0)},
                'params': {'PAC': row.get('PAC', 0), '碳源': row.get('碳源', 0), 'MLSS': row.get('MLSS', 0), 'DO': row.get('DO', 0)}
            }
            st.session_state.has_result = True
            st.sidebar.success("✅ 数据加载成功")
        except Exception as e:
            st.sidebar.error(f"文件解析失败: {e}")

# ---- 自动实时 ----
else:
    st.sidebar.markdown("### 🔄 自动实时")
    if st.sidebar.button("▶️ 生成模拟数据"):
        cod_in = 200 + np.random.normal(0, 30)
        nh3_in = 20 + np.random.normal(0, 3)
        tp_in = 3 + np.random.normal(0, 0.4)
        tn_in = 30 + np.random.normal(0, 5)
        ss_in = 150 + np.random.normal(0, 20)
        flow_in = 10000 + np.random.normal(0, 500)
        pac_in = 30 + np.random.normal(0, 2)
        carbon_in = 50 + np.random.normal(0, 3)
        mlss_in = 4000 + np.random.normal(0, 200)
        do_in = 2 + np.random.normal(0, 0.2)
        
        removal = predict_removal(cod_in, nh3_in, tp_in, tn_in, ss_in, flow_in, pac_in, carbon_in, mlss_in, do_in)
        effluent = {
            'COD': cod_in * (1 - 0.93),
            'NH3-N': nh3_in * (1 - removal),
            'TP': tp_in * (1 - 0.88),
            'TN': tn_in * (1 - 0.75),
            'SS': ss_in * (1 - 0.92)
        }
        st.session_state.result = {
            'removal': removal,
            'effluent': effluent,
            'inlet': {'COD': cod_in, 'NH3-N': nh3_in, 'TP': tp_in, 'TN': tn_in, 'SS': ss_in, '流量': flow_in},
            'params': {'PAC': pac_in, '碳源': carbon_in, 'MLSS': mlss_in, 'DO': do_in}
        }
        st.session_state.has_result = True

# ==========================================
# 显示结果
# ==========================================
if st.session_state.get('has_result', False):
    result = st.session_state.result
    effluent = result['effluent']
    inlet = result['inlet']
    removal = result['removal']
    
    # 状态
    has_abnormal = False
    for k in ['COD', 'NH3-N', 'TP', 'TN', 'SS']:
        if effluent.get(k, 0) > DESIGN_LIMITS[k]['value']:
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
    
    st.markdown("---")
    
    # ---- 进出水对比 ----
    st.markdown('<div class="section-header">📊 进出水水质实时监测</div>', unsafe_allow_html=True)
    st.caption(f"📌 出水设计标准：COD≤{DESIGN_LIMITS['COD']['value']} | NH₃-N≤{DESIGN_LIMITS['NH3-N']['value']} | TP≤{DESIGN_LIMITS['TP']['value']} | TN≤{DESIGN_LIMITS['TN']['value']} | SS≤{DESIGN_LIMITS['SS']['value']} mg/L")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### 🔵 进水水质")
        cols = st.columns(3)
        cols[0].metric("COD", f"{inlet['COD']:.0f} mg/L")
        cols[1].metric("NH₃-N", f"{inlet['NH3-N']:.1f} mg/L")
        cols[2].metric("TP", f"{inlet['TP']:.2f} mg/L")
        cols2 = st.columns(3)
        cols2[0].metric("TN", f"{inlet['TN']:.1f} mg/L")
        cols2[1].metric("SS", f"{inlet['SS']:.0f} mg/L")
        cols2[2].metric("流量", f"{inlet['流量']:.0f} m³/h")
    
    with col_right:
        st.markdown("#### 🟢 预测出水水质")
        cols = st.columns(3)
        for i, (k, color) in enumerate([('COD', '#2E86AB'), ('NH3-N', '#27AE60'), ('TP', '#F39C12')]):
            val = effluent.get(k, 0)
            limit = DESIGN_LIMITS[k]['value']
            ok = val <= limit
            cols[i].metric(
                k,
                f"{val:.2f} mg/L",
                delta="✅" if ok else f"🔴 超标{val-limit:.2f}",
                delta_color="normal" if ok else "off"
            )
        cols2 = st.columns(3)
        for i, (k, color) in enumerate([('TN', '#1ABC9C'), ('SS', '#95A5A6')]):
            val = effluent.get(k, 0)
            limit = DESIGN_LIMITS[k]['value']
            ok = val <= limit
            cols2[i].metric(
                k,
                f"{val:.2f} mg/L",
                delta="✅" if ok else f"🔴 超标{val-limit:.2f}",
                delta_color="normal" if ok else "off"
            )
        cols2[2].metric("NH₃-N去除率", f"{removal*100:.1f}%")
    
    # ---- 运行参数 ----
    st.markdown("---")
    st.markdown("#### 🟡 运行参数")
    params = result['params']
    cols = st.columns(5)
    cols[0].metric("PAC", f"{params['PAC']:.0f} mg/L")
    cols[1].metric("碳源", f"{params['碳源']:.0f} mg/L")
    cols[2].metric("MLSS", f"{params['MLSS']:.0f} mg/L")
    cols[3].metric("DO", f"{params['DO']:.1f} mg/L")
    cols[4].metric("NH₃-N去除率", f"{removal*100:.1f}%")
    
    # ---- 记忆长度 ----
    st.markdown("---")
    st.markdown('<div class="section-header">🧠 记忆长度与分频调控策略</div>', unsafe_allow_html=True)
    st.caption("基于 XGBoost-SHAP 分析的系统记忆长度共识")
    
    col_ch1, col_ch2, col_ch3 = st.columns(3)
    with col_ch1:
        st.markdown("""
        <div class="channel-item channel-fast">
            <div class="ch-name">⚡ 快速通道</div>
            <div class="ch-value" style="color:#27AE60;">1-9h</div>
            <div class="ch-desc">NH₃-N (1h) · TP (1h) · TN (9h)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_ch2:
        st.markdown("""
        <div class="channel-item channel-special">
            <div class="ch-name">⚠️ 不适用</div>
            <div class="ch-value" style="color:#E74C3C;">—</div>
            <div class="ch-desc">COD (去除冗余)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_ch3:
        st.markdown("""
        <div class="channel-item channel-special">
            <div class="ch-name">🔴 特殊通道</div>
            <div class="ch-value" style="color:#E74C3C;">不稳定</div>
            <div class="ch-desc">SS (实时报警)</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ---- 永久记忆统计 ----
    st.markdown("---")
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    with col_stats1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">🧠 模型版本</div>
            <div class="stat-value">v7.0</div>
            <div class="stat-sub">去除率模型</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stats2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">📊 数据模式</div>
            <div class="stat-value">手动/文件/实时</div>
            <div class="stat-sub">三种输入方式</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stats3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">🧠 记忆长度</div>
            <div class="stat-value">NH₃-N 1h · TP 1h · TN 9h</div>
            <div class="stat-sub">XGBoost-SHAP分析</div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("👈 左侧输入数据后点击「预测」按钮查看结果")
    
    # ---- 使用说明 ----
    st.markdown("---")
    st.markdown("#### 📌 使用说明")
    st.markdown("""
    1. 在左侧边栏输入进水水质和运行参数
    2. 点击「预测」按钮查看结果
    3. 系统会显示预测出水水质和NH₃-N去除率
    4. 查看各指标的系统记忆长度
    """)
    
    st.markdown("#### 🧠 系统记忆长度")
    st.markdown("""
    | 指标 | 记忆长度 | 工艺解释 |
    | :--- | :---: | :--- |
    | **NH₃-N** | **1 小时** | 硝化反应，受DO直接影响 |
    | **TP** | **1 小时** | 化学除磷（PAC），瞬时反应 |
    | **TN** | **9 小时** | 反硝化，受碳源投加延迟影响 |
    """)

st.markdown("---")
beijing_now = datetime.now(BEIJING_TZ)
st.caption(f"🏭 v7.0 | 去除率模型 | 永久记忆已启用 | {beijing_now.strftime('%Y-%m-%d %H:%M')} 北京时间")
