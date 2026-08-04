"""
FXD2.0 - 龙华水质净化厂智能预警系统 v7.0
基于去除率模型的 XGBoost-SHAP 智能预警
记忆长度：NH3-N=1h, TP=1h, TN=9h
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
warnings.filterwarnings('ignore')

# ==========================================
# 北京时间时区
# ==========================================
BEIJING_TZ = timezone(timedelta(hours=8))

st.set_page_config(
    page_title="龙华水质净化厂智能预警 v7.0",
    page_icon="🏭",
    layout="wide"
)

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
    'NH3-N': {'hours': 1, 'description': '硝化反应，DO敏感'},
    'TP': {'hours': 1, 'description': '化学除磷，PAC瞬时响应'},
    'TN': {'hours': 9, 'description': '反硝化，碳源依赖'}
}

# ==========================================
# CSS样式
# ==========================================
st.markdown("""
<style>
    .main-title {
        font-size: 26px;
        font-weight: 700;
        color: #1a3a5c;
        padding: 10px 0 14px 0;
        border-bottom: 3px solid #2E86AB;
        margin-bottom: 16px;
    }
    .section-header {
        font-size: 16px;
        font-weight: 600;
        color: #1a3a5c;
        margin: 14px 0 8px 0;
        padding-left: 8px;
        border-left: 4px solid #2E86AB;
    }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 10px 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 6px;
        border-left: 4px solid #2E86AB;
    }
    .metric-card .label {
        font-size: 12px;
        color: #666;
        font-weight: 500;
    }
    .metric-card .value {
        font-size: 20px;
        font-weight: 700;
        color: #1a3a5c;
    }
    .metric-card .sub {
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
    .memory-card {
        background: #F0F8FF;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
        border: 1px solid #D6E4F0;
    }
    .memory-card .mem-value {
        font-size: 28px;
        font-weight: 700;
        color: #1a3a5c;
    }
    .memory-card .mem-label {
        font-size: 14px;
        color: #555;
    }
    .memory-card .mem-desc {
        font-size: 11px;
        color: #888;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 加载模型
# ==========================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model_cache/xgb_final_model.pkl')
        scaler = joblib.load('model_cache/scaler.pkl')
        with open('model_cache/feature_cols.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        return model, scaler, feature_cols
    except FileNotFoundError as e:
        st.error(f"❌ 模型文件不存在: {e}")
        st.info("请确保 model_cache 目录下有 xgb_final_model.pkl, scaler.pkl, feature_cols.pkl")
        return None, None, None

model, scaler, feature_cols = load_model()

if model is None:
    st.stop()

# ==========================================
# 标题
# ==========================================
st.markdown('<div class="main-title">🏭 龙华水质净化厂（二期）智能预警系统 v7.0</div>', unsafe_allow_html=True)

beijing_now = datetime.now(BEIJING_TZ)
st.caption(f"📌 基于去除率模型 · XGBoost-SHAP 记忆长度分析 · {beijing_now.strftime('%Y-%m-%d %H:%M')} 北京时间")

# ==========================================
# 侧边栏：数据输入
# ==========================================
st.sidebar.markdown("## 📊 进水数据输入")
st.sidebar.caption("输入进水水质和运行参数，点击预测")

with st.sidebar.form("input_form"):
    st.markdown("### 🔵 进水水质")
    c1, c2 = st.columns(2)
    with c1:
        cod_in = st.number_input("COD (mg/L)", min_value=0.0, value=200.0, step=5.0)
        nh3_in = st.number_input("NH₃-N (mg/L)", min_value=0.0, value=20.0, step=1.0)
        tp_in = st.number_input("TP (mg/L)", min_value=0.0, value=3.0, step=0.1)
    with c2:
        tn_in = st.number_input("TN (mg/L)", min_value=0.0, value=30.0, step=1.0)
        ss_in = st.number_input("SS (mg/L)", min_value=0.0, value=150.0, step=5.0)
        flow_in = st.number_input("流量 (m³/h)", min_value=0.0, value=10000.0, step=100.0)
    
    st.markdown("### 🟡 运行参数")
    c3, c4 = st.columns(2)
    with c3:
        pac_in = st.number_input("PAC (mg/L)", min_value=0.0, value=30.0, step=1.0)
        carbon_in = st.number_input("碳源 (mg/L)", min_value=0.0, value=50.0, step=1.0)
    with c4:
        mlss_in = st.number_input("MLSS (mg/L)", min_value=0.0, value=4000.0, step=50.0)
        do_in = st.number_input("DO (mg/L)", min_value=0.0, value=2.0, step=0.1)
    
    submitted = st.form_submit_button("🔮 预测", type="primary", use_container_width=True)

# ==========================================
# 特征构造函数（匹配模型）
# ==========================================
def build_features(cod, nh3, tp, tn, ss, flow):
    """构造与模型训练时一致的特征矩阵"""
    data = {}
    for col in feature_cols:
        if '进水COD' in col:
            data[col] = cod
        elif 'NH3-N_detrend' in col:
            data[col] = nh3 * 0.9  # 简化去趋势
        elif 'TP_detrend' in col:
            data[col] = tp * 0.9
        elif '进水TN' in col:
            data[col] = tn
        elif '进水SS' in col:
            data[col] = ss
        elif '流量' in col:
            data[col] = flow
        elif '降雨量' in col:
            data[col] = 0
        elif '风量' in col:
            data[col] = 50000
        elif '污泥浓度均值' in col:
            data[col] = mlss_in
        elif '溶解氧浓度均值' in col:
            data[col] = do_in
        elif '产泥量' in col:
            data[col] = 20
        elif '碳源' in col:
            data[col] = carbon_in
        elif '磁粉' in col:
            data[col] = 0
        elif 'PAC' in col:
            data[col] = pac_in
        elif '阴离子' in col:
            data[col] = 150
        elif '阳离子' in col:
            data[col] = 200
        elif '次氯酸钠' in col:
            data[col] = 0.1
        elif '生化池' in col and 'DO' in col:
            data[col] = do_in
        elif '_diff' in col:
            data[col] = 0
        elif '_roll_mean' in col or '_roll_std' in col:
            data[col] = 0
        else:
            data[col] = 0
    return pd.DataFrame([data])

# ==========================================
# 预测
# ==========================================
if submitted:
    try:
        X = build_features(cod_in, nh3_in, tp_in, tn_in, ss_in, flow_in)
        X_scaled = scaler.transform(X)
        pred_removal = model.predict(X_scaled)[0]
        pred_removal = max(0, min(1, pred_removal))
        
        # 反推出水浓度（去除率 × 进水浓度）
        # 注：模型只预测 NH3-N 的去除率，其他指标用经验值替代
        removal_rates = {
            'COD': 0.93,
            'NH3-N': pred_removal,
            'TP': 0.88,
            'TN': 0.75,
            'SS': 0.92
        }
        effluent = {
            'COD': cod_in * (1 - removal_rates['COD']),
            'NH3-N': nh3_in * (1 - removal_rates['NH3-N']),
            'TP': tp_in * (1 - removal_rates['TP']),
            'TN': tn_in * (1 - removal_rates['TN']),
            'SS': ss_in * (1 - removal_rates['SS'])
        }
        
        # 存储结果到 session_state
        st.session_state['pred_result'] = {
            'removal': removal_rates,
            'effluent': effluent,
            'nh3_removal': pred_removal
        }
        st.session_state['has_prediction'] = True
        
    except Exception as e:
        st.error(f"❌ 预测失败: {e}")

# ==========================================
# 显示结果
# ==========================================
if st.session_state.get('has_prediction', False):
    pred = st.session_state['pred_result']
    effluent = pred['effluent']
    removal = pred['removal']
    
    # ---- 状态栏 ----
    has_abnormal = False
    for k in ['COD', 'NH3-N', 'TP', 'TN', 'SS']:
        if effluent.get(k, 0) > DESIGN_LIMITS[k]['value']:
            has_abnormal = True
            break
    
    col_status, col_mem, col_time = st.columns(3)
    with col_status:
        status_text = "🔴 异常" if has_abnormal else "🟢 正常"
        st.markdown(f"### 📊 系统状态: {status_text}")
    with col_mem:
        st.markdown(f"### 🧠 NH₃-N 去除率: **{removal['NH3-N']*100:.1f}%**")
    with col_time:
        st.markdown(f"### ⏱️ {beijing_now.strftime('%H:%M')}")
    
    st.markdown("---")
    
    # ---- 进出水对比 ----
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### 🔵 进水水质")
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">COD</div>
            <div class="value">{cod_in:.0f} <span style="font-size:14px;font-weight:400;color:#888;">mg/L</span></div>
        </div>
        <div class="metric-card">
            <div class="label">NH₃-N</div>
            <div class="value">{nh3_in:.1f} <span style="font-size:14px;font-weight:400;color:#888;">mg/L</span></div>
        </div>
        <div class="metric-card">
            <div class="label">TP</div>
            <div class="value">{tp_in:.2f} <span style="font-size:14px;font-weight:400;color:#888;">mg/L</span></div>
        </div>
        <div class="metric-card">
            <div class="label">TN</div>
            <div class="value">{tn_in:.1f} <span style="font-size:14px;font-weight:400;color:#888;">mg/L</span></div>
        </div>
        <div class="metric-card">
            <div class="label">SS</div>
            <div class="value">{ss_in:.0f} <span style="font-size:14px;font-weight:400;color:#888;">mg/L</span></div>
        </div>
        <div class="metric-card">
            <div class="label">流量</div>
            <div class="value">{flow_in:.0f} <span style="font-size:14px;font-weight:400;color:#888;">m³/h</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        st.markdown("#### 🟢 预测出水水质")
        for k in ['COD', 'NH3-N', 'TP', 'TN', 'SS']:
            val = effluent.get(k, 0)
            limit = DESIGN_LIMITS[k]['value']
            ok = val <= limit
            color = "#1B7A4A" if ok else "#C0392B"
            status = "✅ 达标" if ok else f"🔴 超标 {val-limit:.2f}"
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {'#27AE60' if ok else '#E74C3C'};">
                <div class="label">{k} <span class="limit-ref">限值≤{limit}</span></div>
                <div class="value" style="color:{color};">{val:.3f} <span style="font-size:14px;font-weight:400;color:#888;">mg/L</span></div>
                <div class="sub">{status}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # ---- 运行参数 ----
    st.markdown("---")
    st.markdown("#### 🟡 当前运行参数")
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
    with col_p1:
        st.metric("PAC", f"{pac_in:.0f} mg/L")
    with col_p2:
        st.metric("碳源", f"{carbon_in:.0f} mg/L")
    with col_p3:
        st.metric("MLSS", f"{mlss_in:.0f} mg/L")
    with col_p4:
        st.metric("DO", f"{do_in:.1f} mg/L")
    with col_p5:
        st.metric("去除率", f"{removal['NH3-N']*100:.1f}%")
    
    # ---- 记忆长度 ----
    st.markdown("---")
    st.markdown("#### 🧠 系统记忆长度（XGBoost-SHAP 分析）")
    st.caption("进水负荷变化后，系统响应达到最明显的滞后时间")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    memories = [
        ('NH₃-N', MEMORY['NH3-N']['hours'], MEMORY['NH3-N']['description'], '#27AE60'),
        ('TP', MEMORY['TP']['hours'], MEMORY['TP']['description'], '#2E86AB'),
        ('TN', MEMORY['TN']['hours'], MEMORY['TN']['description'], '#F39C12')
    ]
    for i, (name, hours, desc, color) in enumerate(memories):
        cols = [col_m1, col_m2, col_m3]
        with cols[i]:
            st.markdown(f"""
            <div class="memory-card" style="border-top: 4px solid {color};">
                <div class="mem-label">{name}</div>
                <div class="mem-value" style="color:{color};">{hours} <span style="font-size:16px;">小时</span></div>
                <div class="mem-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.caption("💡 NH₃-N 和 TP 响应极快（1小时内可见变化），TN 约需 9 小时响应（碳源投加延迟）。")

else:
    st.info("👈 左侧输入数据后点击「预测」按钮查看结果")
    
    # ---- 展示示例 ----
    st.markdown("---")
    st.markdown("#### 📌 使用说明")
    st.markdown("""
    1. 在左侧边栏输入进水水质（COD、NH₃-N、TP、TN、SS、流量）
    2. 输入运行参数（PAC、碳源、MLSS、DO）
    3. 点击「预测」按钮查看结果
    4. 系统会显示预测出水水质和超标情况
    5. 查看各指标的系统记忆长度
    """)
    
    st.markdown("---")
    st.markdown("#### 🧠 系统记忆长度")
    st.markdown("""
    | 指标 | 记忆长度 | 工艺解释 |
    | :--- | :---: | :--- |
    | **NH₃-N** | **1 小时** | 硝化反应，受DO直接影响，响应极快 |
    | **TP** | **1 小时** | 化学除磷（PAC），瞬时反应 |
    | **TN** | **9 小时** | 反硝化，受碳源投加延迟影响 |
    """)
    
    st.caption("📌 基于 2023-2026 年 29,928 条小时级数据训练 · XGBoost-SHAP 分析")

# ==========================================
# 页脚
# ==========================================
st.markdown("---")
st.caption(f"🏭 龙华水质净化厂（二期）· 智能预警系统 v7.0 · 去除率模型 · {beijing_now.strftime('%Y-%m-%d %H:%M')} 北京时间")