import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
from fpdf import FPDF
import time
import zipfile
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="SentinelX Analytics",
    page_icon="🛡️",
    layout="wide"
)

# =========================
# LOAD MODEL (AUTO-UNZIP RECOVERY)
# =========================
@st.cache_resource
def load_fraud_model():
    pkl_filename = "Credit_Card_Fraud_Detection.pkl"
    zip_filename = "Credit_Card_Fraud_Detection.zip"
    
    # If the uncompressed .pkl doesn't exist yet, extract it from the .zip file
    if not os.path.exists(pkl_filename):
        if os.path.exists(zip_filename):
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(".")
        else:
            st.error(f"Critical Error: Neither '{pkl_filename}' nor '{zip_filename}' was found in the repository.")
            st.stop()
            
    return joblib.load(pkl_filename)

try:
    model = load_fraud_model()
except Exception as e:
    st.error(f"Model framework load failed: {str(e)}")
    st.stop()

# =========================
# SESSION STATE MANAGEMENT
# =========================
if "history" not in st.session_state:
    st.session_state.history = []
if "records" not in st.session_state:
    st.session_state.records = []
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# =========================
# THEME ENGINE & CUSTOM UI
# =========================
theme = st.sidebar.toggle("🌙 Dark Mode Customizer", value=True)

if theme:
    bg_gradient = "linear-gradient(135deg, #0f172a, #1e1b4b, #020617)"
    card_bg = "rgba(30, 41, 59, 0.7)"
    text_color = "#f8fafc"
    border_color = "rgba(255, 255, 255, 0.15)"
else:
    bg_gradient = "linear-gradient(135deg, #f0fdf4, #e0f2fe, #f8fafc)"
    card_bg = "rgba(255, 255, 255, 0.75)"
    text_color = "#0f172a"
    border_color = "rgba(15, 23, 42, 0.12)"

st.markdown(f"""
<style>
/* App Dynamic Background */
.stApp {{
    background: {bg_gradient};
    background-attachment: fixed;
}}

/* Structural UI Panels */
.sentinel-header {{
    text-align: center;
    padding: 20px 0 10px 0;
}}
.main-title {{
    font-size: 46px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: {text_color};
    margin-bottom: 4px;
}}
.subtitle {{
    font-size: 18px;
    color: {text_color};
    opacity: 0.85;
    margin-bottom: 30px;
}}

/* Glassmorphic Interface Cards */
.glass-card {{
    background: {card_bg};
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    padding: 24px;
    border-radius: 20px;
    border: 1px solid {border_color};
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
    margin-bottom: 24px;
    transition: all 0.3s ease;
}}
.glass-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
}}

/* Dynamic Response Alerts */
.fraud-popup {{
    background: rgba(254, 226, 226, 0.95);
    padding: 20px;
    border-radius: 14px;
    border-left: 8px solid #dc2626;
    color: #991b1b;
    font-size: 22px;
    font-weight: 800;
    text-align: center;
    margin: 20px 0;
    box-shadow: 0 4px 20px rgba(220, 38, 38, 0.15);
}}
.safe-popup {{
    background: rgba(220, 252, 231, 0.95);
    padding: 20px;
    border-radius: 14px;
    border-left: 8px solid #16a34a;
    color: #166534;
    font-size: 22px;
    font-weight: 800;
    text-align: center;
    margin: 20px 0;
    box-shadow: 0 4px 20px rgba(22, 163, 74, 0.15);
}}

/* Typography Override */
h1, h2, h3, h4, p, label, span, div {{
    color: {text_color};
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}}

/* Button System Overrides */
.stButton>button {{
    width: 100%;
    height: 52px;
    border: none;
    border-radius: 14px;
    background: linear-gradient(135deg, #3b82f6, #1d4ed8);
    color: white !important;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    transition: all 0.2s ease;
}}
.stButton>button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
}}

/* Form Handling and Controls Cleanup */
.stNumberInput input, .stTextInput input {{
    border-radius: 10px !important;
    background-color: white !important;
    color: #0f172a !important;
}}
</style>
""", unsafe_allow_html=True)

# =========================
# GATEWAY LOG-IN CHECK
# =========================
if not st.session_state.logged_in:
    st.markdown("<div class='sentinel-header'><div class='main-title'>🔒 SentinelX Enterprise</div></div>", unsafe_allow_html=True)
    
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        username = st.text_input("Operator Username")
        password = st.text_input("Security Access Code", type="password")
        
        if st.button("Authenticate System"):
            if username == "ayush" and password == "1718":
                st.session_state.logged_in = True
                st.success("Authorization Verified.")
                st.rerun()
            else:
                st.error("Identity credentials invalid.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# =========================
# PERSISTENT SYSTEM FRAMEWORK
# =========================
st.markdown("""
<div class="sentinel-header">
    <div class="main-title">🛡️ SentinelX Corporate Engine</div>
    <div class="subtitle">Predictive Deep-Analysis Risk Evaluation Dashboard</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("System Navigator")
menu = st.sidebar.radio(
    "Modules Available",
    ["Dashboard System", "Historical Logs", "Conversational AI Core"]
)

# Static Conversions Mapping System
months = {"January":1, "February":2, "March":3, "April":4, "May":5, "June":6, "July":7, "August":8, "September":9, "October":10, "November":11, "December":12}
days = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}
hour_options = [f"{i} AM" if i != 0 else "12 AM" for i in range(12)] + [f"{i} PM" if i != 0 else "12 PM" for i in range(12)]
hour_map = {opt: idx for idx, opt in enumerate([f"{i} AM" if i!=0 else "12 AM" for i in range(12)] + [f"{i} PM" if i!=0 else "12 PM" for i in range(12)])}

# =========================
# COMPONENT: GAUGE METERS
# =========================
def render_risk_gauge(prob):
    color = "#22c55e"
    if prob > 0.3: color = "#f59e0b"
    if prob > 0.7: color = "#ef4444"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': '%', 'font': {'color': text_color}},
        title={'text': "Risk Probability Context", 'font': {'color': text_color, 'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': text_color},
            'bar': {'color': color},
            'bgcolor': 'rgba(0,0,0,0.1)',
            'steps': [
                {'range': [0, 30], 'color': 'rgba(34, 197, 94, 0.2)'},
                {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ]
        }
    ))
    fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=30, b=10, l=10, r=10))
    return fig

# =========================
# CORE APPLICATION MODULES
# =========================
if menu == "Dashboard System":
    total_tx = len(st.session_state.records)
    fraud_tx = len([r for r in st.session_state.records if r["Prediction"] == "Fraud"])
    safe_tx = total_tx - fraud_tx
    pct_fraud = (fraud_tx / total_tx) * 100 if total_tx > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Processed Transactions", total_tx)
    c2.metric("Flagged Fraudulent", fraud_tx)
    c3.metric("Verified Legitimate", safe_tx)
    c4.metric("Engine Deviation Ratio", f"{pct_fraud:.2f}%")

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📋 Core Transaction Telemetry")
        
        Amount = st.number_input("Transactional Value Asset (INR)", min_value=0.0, value=1000.0)
        MerchantID = st.number_input("Target Merchant ID Registry", min_value=0.0, value=1.0)
        
        Year = st.number_input("Operational Year Context", min_value=2020, max_value=2030, value=2024)
        
        selected_month = st.selectbox("Operational Month Context", list(months.keys()))
        TransactionMonth = months[selected_month]
        
        TransactionDay = st.number_input("Day of Month (1-31)", min_value=1, max_value=31, value=1)
        
        selected_hour = st.selectbox("Transactional Timeline Profile", hour_options)
        TransactionHour = hour_map[selected_hour]
        
        selected_day = st.selectbox("Day Matrix Calendar Point", list(days.keys()))
        TransactionDayOfWeek = days[selected_day]
        
        refund_option = st.selectbox("Transaction Type Profile", ["Purchase", "Refund"])
        tx_type_purchase = 1 if refund_option == "Purchase" else 0
        tx_type_refund = 1 if refund_option == "Refund" else 0
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📍 Geospatial Origin Context")
        
        location = st.selectbox("Origin System Location Vertex", [
            "Chicago", "Dallas", "Houston", "Los Angeles", "New York", 
            "Philadelphia", "Phoenix", "San Antonio", "San Diego", "San Jose"
        ])
        
        cities_ordered = ["Chicago", "Dallas", "Houston", "Los Angeles", "New York", "Philadelphia", "Phoenix", "San Antonio", "San Diego", "San Jose"]
        location_encoded = [1 if location == city else 0 for city in cities_ordered]
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🚀 Execute Enterprise Risk Scan"):
        with st.spinner("Decoding heuristic frameworks via SentinelX AI..."):
            time.sleep(1.5)
            
            features = [
                Amount, 
                MerchantID, 
                Year, 
                TransactionMonth, 
                TransactionDay, 
                TransactionHour, 
                TransactionDayOfWeek,
                tx_type_purchase,
                tx_type_refund
            ] + location_encoded
            
            X = np.array(features).reshape(1, -1)
            prob = model.predict_proba(X)[0][1]
            
            prediction = 0
            if (Amount > 50000) or \
               (TransactionHour < 5 and Amount > 20000) or \
               (tx_type_refund == 1 and Amount > 30000) or \
               (TransactionDayOfWeek in [5, 6] and Amount > 40000) or \
               (prob > 0.30):
                prediction = 1

            result = "Fraud" if prediction == 1 else "Legitimate"
            st.session_state.history.append(prob)
            st.session_state.records.append({
                "Amount": Amount,
                "Location": location,
                "Probability": round(prob * 100, 2),
                "Prediction": result
            })

            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("🎯 Real-Time Engine Verdict Evaluation")
            
            rg_col1, rg_col2 = st.columns([3, 2])
            with rg_col1:
                st.plotly_chart(render_risk_gauge(prob), use_container_width=True)
            with rg_col2:
                if prediction == 1:
                    st.markdown("<div class='fraud-popup'>⚠️ HIGH RISK FRAUDULENT PATTERN IDENTIFIED</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='safe-popup'>✅ VERIFIED TRANSACTION SECURED SAFE</div>", unsafe_allow_html=True)
                st.metric("Risk Confidence Weight Value", f"{prob * 100:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("📊 Engine Historical Analysis Over Time")
            
            fig_pie = go.Figure(data=[go.Pie(labels=['Legitimate Base', 'Flagged Fraud Matrix'], values=[safe_tx + 1, fraud_tx + 1], hole=.4)])
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color=text_color), height=260, margin=dict(t=10, b=10, l=10, r=10))
            
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Distribution Layout Breakdown**")
                st.plotly_chart(fig_pie, use_container_width=True)
            with g2:
                st.markdown("**Transactional Variance Run History**")
                st.line_chart(st.session_state.history)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("💡 Neural Machine Transparent Explanation Mapping")
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            feature_names = [
                "Amount", "MerchantID", "Year", "Month", "Day", "Hour", "DayOfWeek",
                "Type: Purchase", "Type: Refund",
                "City: Chicago", "City: Dallas", "City: Houston", "City: Los Angeles", 
                "City: New York", "City: Philadelphia", "City: Phoenix", "City: San Antonio", 
                "City: San Diego", "City: San Jose"
            ]
            
            shap_data = shap_values[-1] if isinstance(shap_values, list) else shap_values
            shap_data = np.array(shap_data).flatten()
            
            shap_df = pd.DataFrame({"Feature Structure Component": feature_names, "Calculated Matrix Impact": shap_data[:len(feature_names)]})
            shap_df = shap_df.sort_values(by="Calculated Matrix Impact", key=abs, ascending=False)
            
            st.bar_chart(shap_df.set_index("Feature Structure Component"))
            
            top_df = shap_df.head(6)
            waterfall_fig = go.Figure(go.Waterfall(
                name="Variance Weight Value Map", orientation="v", measure=["relative"] * len(top_df),
                x=top_df["Feature Structure Component"], textposition="outside", y=top_df["Calculated Matrix Impact"],
                connector={"line": {"color": "rgb(120, 120, 120)"}}
            ))
            waterfall_fig.update_layout(title="Top Critical Vector Drivers Graph", showlegend=False, height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=text_color, size=12))
            st.plotly_chart(waterfall_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("📥 Export Audit Manifest Documentation Records")
            
            csv_data = pd.DataFrame(st.session_state.records).to_csv(index=False)
            
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(200, 10, txt="SentinelX System Forensic Audit Report File", ln=True)
            pdf.cell(200, 10, txt=f"Risk Verdict Assessment Metric: {result}", ln=True)
            pdf.cell(200, 10, txt=f"Calculated Probability Weight Bounds: {prob * 100:.2f}%", ln=True)
            pdf.output("sentinelx_audit.pdf")
            
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button("⬇ Download Structured Data CSV", csv_data, "sentinelx_report.csv", "text/csv")
            with dl2:
                with open("sentinelx_audit.pdf", "rb") as file:
                    st.download_button(label="⬇ Download Cryptographic PDF Report", data=file, file_name="sentinelx_audit.pdf", mime="application/pdf")
            st.markdown("</div>", unsafe_allow_html=True)

elif menu == "Historical Logs":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📜 Enterprise Database Systems Integrity Log Stream")
    if st.session_state.records:
        history_df = pd.DataFrame(st.session_state.records)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No records are currently held in standard operational system pipeline cache.")
    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "Conversational AI Core":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🤖 Neural Conversational Interface Support Node")
    user_q = st.text_input("Pose a query to the Core Assistant Interface:")
    if user_q:
        if "fraud" in user_q.lower():
            st.write("**SentinelX AI Link:** Detected specific markers referencing anomalous activity validation vectors.")
        elif "safe" in user_q.lower():
            st.write("**SentinelX AI Link:** Input contextual metrics indicate standard compliance vectors within predictable operational ranges.")
        else:
            st.write("**SentinelX AI Link:** Acknowledged. Requesting advanced log queries to verify specific transactional context.")
    st.markdown("</div>", unsafe_allow_html=True)