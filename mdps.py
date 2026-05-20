import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="GlucoSense · Diabetes Screening",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=Crimson+Pro:ital,wght@0,400;0,600;1,400&display=swap');

  html, body, [class*="css"] {
      font-family: 'Sora', sans-serif;
      background: #faf7f4;
      color: #1c1917;
  }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 1.8rem 2.4rem 3rem; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
      background: linear-gradient(170deg, #1c1917 0%, #3b1f1a 55%, #7c2d12 100%) !important;
      border-right: none !important;
  }
  [data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
  .sidebar-logo {
      display: flex; align-items: center; gap: 11px;
      padding: 0.4rem 0 1.6rem;
      border-bottom: 1px solid rgba(255,255,255,0.1);
      margin-bottom: 1.6rem;
  }
  .sidebar-logo-icon {
      width: 42px; height: 42px; border-radius: 12px;
      background: linear-gradient(145deg,#c0392b,#e74c3c,#f39c12);
      display:flex; align-items:center; justify-content:center;
      font-size:1.25rem;
      box-shadow: 0 4px 14px rgba(192,57,43,0.4);
      flex-shrink:0;
  }
  .sidebar-brand { font-size:1.05rem; font-weight:700; letter-spacing:-0.3px; }
  .sidebar-sub   { font-size:0.65rem; opacity:0.55; text-transform:uppercase; letter-spacing:0.7px; margin-top:2px; }

  .sidebar-section-title {
      font-size:0.62rem; font-weight:700; text-transform:uppercase;
      letter-spacing:1.3px; color:rgba(255,255,255,0.4) !important;
      margin:1.4rem 0 0.7rem;
  }
  .fact-card {
      background: rgba(255,255,255,0.07);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 12px;
      padding: 0.95rem 1.1rem;
      margin-bottom: 0.75rem;
  }
  .fact-card .fact-num {
      font-size:1.4rem; font-weight:700; color:#fb923c !important; line-height:1;
  }
  .fact-card .fact-lbl {
      font-size:0.72rem; color:rgba(255,255,255,0.55) !important;
      margin-top:3px; line-height:1.4;
  }
  .tip-row {
      display:flex; gap:9px; align-items:flex-start;
      margin-bottom:0.65rem; font-size:0.78rem;
      color:rgba(255,255,255,0.7) !important; line-height:1.5;
  }
  .tip-row .tip-icon { flex-shrink:0; margin-top:1px; }
  .sidebar-divider {
      border:none; border-top:1px solid rgba(255,255,255,0.1);
      margin:1.3rem 0;
  }
  .risk-factor {
      display:flex; justify-content:space-between; align-items:center;
      padding:7px 0;
      border-bottom:1px solid rgba(255,255,255,0.07);
      font-size:0.77rem;
  }
  .risk-factor:last-child { border-bottom:none; }
  .risk-dot {
      width:8px; height:8px; border-radius:50%; flex-shrink:0;
  }
  .risk-dot.high   { background:#ef4444; }
  .risk-dot.medium { background:#f97316; }
  .risk-dot.low    { background:#22c55e; }

  /* ── Navbar ── */
  .navbar {
      display:flex; align-items:center; gap:12px;
      margin-bottom:1.6rem;
      padding-bottom:1.2rem;
      border-bottom:1px solid #e7e5e4;
  }
  .logo-circle {
      width:44px; height:44px; border-radius:13px;
      background:linear-gradient(145deg,#c0392b,#e74c3c,#f39c12);
      display:flex; align-items:center; justify-content:center;
      font-size:1.2rem;
      box-shadow:0 4px 14px rgba(192,57,43,0.3);
      flex-shrink:0;
  }
  .logo-text .brand  { font-size:1.1rem; font-weight:700; color:#1c1917; letter-spacing:-0.3px; }
  .logo-text .tagline{ font-size:0.67rem; color:#78716c; text-transform:uppercase; letter-spacing:0.6px; }
  .nav-badge {
      margin-left:auto;
      background:#fff3f3; border:1.5px solid #fca5a5;
      border-radius:20px; padding:4px 13px;
      font-size:0.68rem; font-weight:600; color:#b91c1c;
      text-transform:uppercase; letter-spacing:0.4px;
  }

  /* ── Hero ── */
  .hero {
      background:linear-gradient(135deg,#1c1917 0%,#3b1f1a 50%,#7c2d12 100%);
      border-radius:20px; padding:2.2rem 2.4rem 1.9rem;
      margin-bottom:1.4rem; position:relative; overflow:hidden;
      box-shadow:0 10px 36px rgba(28,25,23,0.2);
  }
  .hero::before {
      content:''; position:absolute; top:-55px; right:-55px;
      width:190px; height:190px; border-radius:50%;
      background:radial-gradient(circle,rgba(239,68,68,0.22) 0%,transparent 70%);
  }
  .hero::after {
      content:''; position:absolute; bottom:-35px; left:35%;
      width:140px; height:140px; border-radius:50%;
      background:radial-gradient(circle,rgba(251,146,60,0.15) 0%,transparent 70%);
  }
  .hero-eyebrow {
      font-size:0.68rem; font-weight:600; text-transform:uppercase;
      letter-spacing:1.5px; color:#fca5a5; margin-bottom:0.6rem;
  }
  .hero h1 {
      font-family:'Crimson Pro',serif; font-size:2.3rem; font-weight:600;
      color:#fafaf9; margin:0 0 0.8rem; line-height:1.15;
  }
  .hero h1 em { font-style:italic; color:#fb923c; }
  .hero-desc {
      font-size:0.86rem; color:rgba(255,255,255,0.62);
      line-height:1.75; font-weight:300; margin:0; max-width:560px;
  }
  .hero-stats {
      display:flex; gap:28px; margin-top:1.6rem;
      padding-top:1.3rem; border-top:1px solid rgba(255,255,255,0.1);
  }
  .stat-num { font-size:1.25rem; font-weight:700; color:#fb923c; line-height:1; }
  .stat-lbl { font-size:0.65rem; color:rgba(255,255,255,0.42); text-transform:uppercase; letter-spacing:0.6px; margin-top:3px; }

  /* ── How it works ── */
  .hiw-strip {
      display:flex; gap:10px; margin-bottom:1.5rem; flex-wrap:wrap;
  }
  .hiw-step {
      background:#ffffff; border:1px solid #e7e5e4; border-radius:14px;
      padding:1rem 1.1rem; flex:1; min-width:140px;
      box-shadow:0 2px 8px rgba(28,25,23,0.05);
  }
  .hiw-step .step-num {
      width:26px; height:26px; border-radius:8px;
      background:linear-gradient(135deg,#c0392b,#f39c12);
      color:#fff; font-size:0.75rem; font-weight:700;
      display:flex; align-items:center; justify-content:center;
      margin-bottom:0.55rem;
  }
  .hiw-step .step-title { font-size:0.8rem; font-weight:600; color:#1c1917; margin-bottom:3px; }
  .hiw-step .step-desc  { font-size:0.72rem; color:#78716c; line-height:1.5; }

  /* ── Field hint ── */
  .field-hint {
      background:#fffbf5; border:1px solid #fed7aa;
      border-radius:10px; padding:0.7rem 1rem;
      font-size:0.76rem; color:#92400e; line-height:1.6;
      margin-bottom:1rem; display:flex; gap:8px; align-items:flex-start;
  }

  /* ── Section card ── */
  .card {
      background:#ffffff; border-radius:18px;
      padding:1.6rem 1.8rem 1.2rem;
      margin-bottom:1.2rem;
      box-shadow:0 2px 14px rgba(28,25,23,0.06);
      border:1px solid #e7e5e4;
  }
  .card-header { display:flex; align-items:center; gap:9px; margin-bottom:0.5rem; }
  .card-icon {
      width:30px; height:30px; border-radius:9px;
      display:flex; align-items:center; justify-content:center; font-size:0.9rem; flex-shrink:0;
  }
  .card-icon.red   { background:#fff1f1; }
  .card-icon.amber { background:#fffbeb; }
  .card-title-text { font-size:0.71rem; font-weight:700; text-transform:uppercase; letter-spacing:1.1px; color:#78716c; }
  .card-subtitle   { font-size:0.78rem; color:#a8a29e; margin-bottom:1.1rem; line-height:1.5; }

  /* ── Inputs ── */
  label { font-weight:500 !important; font-size:0.84rem !important; color:#44403c !important; }
  input[type="text"], input[type="number"] {
      border-radius:10px !important; border:1.5px solid #e7e5e4 !important;
      background:#faf9f8 !important; font-size:0.92rem !important;
      transition:border-color 0.2s, box-shadow 0.2s;
  }
  input[type="text"]:focus, input[type="number"]:focus {
      border-color:#c0392b !important;
      box-shadow:0 0 0 3px rgba(192,57,43,0.1) !important;
      background:#fff !important;
  }

  /* ── Button ── */
  div.stButton > button {
      width:100%;
      background:linear-gradient(135deg,#c0392b 0%,#e74c3c 60%,#f39c12 100%);
      color:white; border:none; border-radius:13px;
      padding:0.85rem 1.5rem; font-size:0.96rem; font-weight:700;
      font-family:'Sora',sans-serif; cursor:pointer;
      transition:opacity 0.18s, transform 0.15s, box-shadow 0.18s;
      letter-spacing:0.2px; margin-top:0.8rem;
      box-shadow:0 4px 18px rgba(192,57,43,0.28);
  }
  div.stButton > button:hover { opacity:0.92; transform:translateY(-2px); box-shadow:0 8px 24px rgba(192,57,43,0.35); }
  div.stButton > button:active { transform:translateY(0); }

  /* ── Results ── */
  .result-safe {
      background:linear-gradient(135deg,#f0fdf4,#dcfce7);
      border:1.5px solid #86efac; border-radius:16px;
      padding:1.4rem 1.7rem; margin-top:1.3rem;
      display:flex; align-items:flex-start; gap:15px;
      box-shadow:0 4px 20px rgba(34,197,94,0.12);
  }
  .result-risk {
      background:linear-gradient(135deg,#fff7ed,#ffedd5);
      border:1.5px solid #fdba74; border-radius:16px;
      padding:1.4rem 1.7rem; margin-top:1.3rem;
      display:flex; align-items:flex-start; gap:15px;
      box-shadow:0 4px 20px rgba(234,88,12,0.12);
  }
  .res-icon  { font-size:2.1rem; flex-shrink:0; margin-top:2px; }
  .result-safe .res-label { font-size:1.1rem; font-weight:700; color:#166534; }
  .result-safe .res-sub   { font-size:0.82rem; color:#15803d; margin-top:4px; line-height:1.55; }
  .result-risk .res-label { font-size:1.1rem; font-weight:700; color:#9a3412; }
  .result-risk .res-sub   { font-size:0.82rem; color:#c2410c; margin-top:4px; line-height:1.55; }
  .res-next {
      margin-top:0.9rem; padding-top:0.9rem;
      border-top:1px solid rgba(0,0,0,0.07);
      font-size:0.76rem; font-weight:600; color:#78716c;
      text-transform:uppercase; letter-spacing:0.6px;
  }
  .res-next ul { margin:0.4rem 0 0; padding-left:1.2rem; font-weight:400; text-transform:none; letter-spacing:0; color:inherit; }
  .res-next li { margin-bottom:3px; }

  /* ── Disclaimer ── */
  .disclaimer {
      background:#fafaf9; border:1px solid #e7e5e4; border-radius:12px;
      padding:0.95rem 1.2rem; font-size:0.77rem; color:#78716c;
      margin-top:1.6rem; line-height:1.6;
      display:flex; gap:9px; align-items:flex-start;
  }

  /* ── Footer ── */
  .footer {
      text-align:center; font-size:0.7rem; color:#a8a29e;
      margin-top:1.8rem; padding-top:1.1rem;
      border-top:1px solid #e7e5e4; letter-spacing:0.3px;
  }
</style>
""", unsafe_allow_html=True)

# ── Model ─────────────────────────────────────────────────────────────────────
scaler = StandardScaler()

@st.cache_resource
def load_model():
    return pickle.load(open("trained_model.sav", "rb"))

loaded_model = load_model()

def diabetes_prediction(arr):
    inp = np.asarray(arr, dtype=float).reshape(1, -1)
    return int(loaded_model.predict(inp)[0])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div class="sidebar-logo-icon">🩸</div>
      <div>
        <div class="sidebar-brand">GlucoSense</div>
        <div class="sidebar-sub">Diabetes Screening</div>
      </div>
    </div>

    <div class="sidebar-section-title">📊 Global Diabetes Facts</div>

    <div class="fact-card">
      <div class="fact-num">537M</div>
      <div class="fact-lbl">Adults living with diabetes worldwide (2021)</div>
    </div>
    <div class="fact-card">
      <div class="fact-num">1 in 2</div>
      <div class="fact-lbl">People with diabetes go undiagnosed</div>
    </div>
    <div class="fact-card">
      <div class="fact-num">90%</div>
      <div class="fact-lbl">Of cases are Type 2, largely preventable</div>
    </div>

    <hr class="sidebar-divider"/>

    <div class="sidebar-section-title">⚠️ Common Risk Factors</div>
    <div class="risk-factor">
      <span>High Glucose Level</span><span class="risk-dot high"></span>
    </div>
    <div class="risk-factor">
      <span>High BMI / Obesity</span><span class="risk-dot high"></span>
    </div>
    <div class="risk-factor">
      <span>Family History</span><span class="risk-dot medium"></span>
    </div>
    <div class="risk-factor">
      <span>Age &gt; 45</span><span class="risk-dot medium"></span>
    </div>
    <div class="risk-factor">
      <span>High Blood Pressure</span><span class="risk-dot medium"></span>
    </div>
    <div class="risk-factor">
      <span>Physical Inactivity</span><span class="risk-dot low"></span>
    </div>

    <hr class="sidebar-divider"/>

    <div class="sidebar-section-title">💡 Healthy Tips</div>
    <div class="tip-row"><span class="tip-icon">🥗</span><span>Eat a balanced diet low in refined sugars and saturated fats.</span></div>
    <div class="tip-row"><span class="tip-icon">🏃</span><span>Aim for at least 150 min of moderate exercise weekly.</span></div>
    <div class="tip-row"><span class="tip-icon">⚖️</span><span>Maintaining a healthy weight reduces risk by up to 60%.</span></div>
    <div class="tip-row"><span class="tip-icon">🩺</span><span>Regular blood glucose checks help catch early signs.</span></div>
    """, unsafe_allow_html=True)

# ── Main content ──────────────────────────────────────────────────────────────
def main():

    # Navbar
    st.markdown("""
    <div class="navbar">
      <div class="logo-circle">🩸</div>
      <div class="logo-text">
        <div class="brand">GlucoSense</div>
        <div class="tagline">AI-Powered Diabetes Screening</div>
      </div>
      <div class="nav-badge">Beta v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    # Hero
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">🔬 Machine Learning · Clinical Screening</div>
      <h1>Early Detection of <em>Diabetes Risk</em></h1>
      <p class="hero-desc">
        GlucoSense analyses eight key clinical biomarkers — including glucose levels, BMI, insulin,
        and blood pressure — using a supervised machine learning model trained on real patient data.
        Enter the patient's values below and receive an instant, data-driven screening result.
        No lab visit or waiting time required.
      </p>
      <div class="hero-stats">
        <div class="stat-item"><div class="stat-num">8</div><div class="stat-lbl">Biomarkers</div></div>
        <div class="stat-item"><div class="stat-num">~2s</div><div class="stat-lbl">Result Time</div></div>
        <div class="stat-item"><div class="stat-num">ML</div><div class="stat-lbl">Powered</div></div>
        <div class="stat-item"><div class="stat-num">0</div><div class="stat-lbl">Data Stored</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <div class="hiw-strip">
      <div class="hiw-step">
        <div class="step-num">1</div>
        <div class="step-title">Enter Patient Data</div>
        <div class="step-desc">Fill in all 8 clinical fields with the patient's measured values.</div>
      </div>
      <div class="hiw-step">
        <div class="step-num">2</div>
        <div class="step-title">ML Model Analyses</div>
        <div class="step-desc">Our trained classifier evaluates patterns across all biomarkers.</div>
      </div>
      <div class="hiw-step">
        <div class="step-num">3</div>
        <div class="step-title">Instant Result</div>
        <div class="step-desc">Receive a clear risk classification in under 2 seconds.</div>
      </div>
      <div class="hiw-step">
        <div class="step-num">4</div>
        <div class="step-title">Consult a Doctor</div>
        <div class="step-desc">Use the result as a starting point for professional clinical review.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Field hint
    st.markdown("""
    <div class="field-hint">
      <span>💡</span>
      <span><strong>Tip:</strong> All values should be numeric. Use recent lab reports or clinical
      measurements for best accuracy. All 8 fields are required before running the screening.</span>
    </div>
    """, unsafe_allow_html=True)

    # Card 1: Patient Profile
    st.markdown("""
    <div class="card">
      <div class="card-header">
        <div class="card-icon red">👤</div>
        <div class="card-title-text">Patient Profile</div>
      </div>
      <div class="card-subtitle">Basic demographics and physical measurements of the patient.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input("Pregnancies", placeholder="e.g. 2", help="Number of times pregnant")
    with col2:
        Age = st.text_input("Age (years)", placeholder="e.g. 35", help="Patient's age in years")
    with col3:
        BMI = st.text_input("BMI", placeholder="e.g. 28.4", help="Body Mass Index = weight(kg) / height(m)²")

    col4, col5 = st.columns(2)
    with col4:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function", placeholder="e.g. 0.627", help="Genetic influence score (0.0 – 2.5)")
    with col5:
        SkinThickness = st.text_input("Skin Thickness (mm)", placeholder="e.g. 20", help="Triceps skin fold thickness in mm")

    # Card 2: Clinical Measurements
    st.markdown("""
    <div class="card" style="margin-top:1rem;">
      <div class="card-header">
        <div class="card-icon amber">🧪</div>
        <div class="card-title-text">Clinical Measurements</div>
      </div>
      <div class="card-subtitle">Blood test and vital sign readings — typically obtained from a recent lab report.</div>
    </div>
    """, unsafe_allow_html=True)

    col6, col7, col8 = st.columns(3)
    with col6:
        Glucose = st.text_input("Glucose (mg/dL)", placeholder="e.g. 120", help="Plasma glucose concentration (2-hr oral test)")
    with col7:
        BloodPressure = st.text_input("Blood Pressure (mmHg)", placeholder="e.g. 70", help="Diastolic blood pressure in mmHg")
    with col8:
        Insulin = st.text_input("Insulin (µU/mL)", placeholder="e.g. 85", help="2-hour serum insulin level")

    # Predict button
    if st.button("🔍  Run Diabetes Screening"):
        fields = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                  Insulin, BMI, DiabetesPedigreeFunction, Age]
        if any(f.strip() == "" for f in fields):
            st.warning("⚠️ All 8 fields are required. Please fill in any missing values.")
        else:
            try:
                result = diabetes_prediction(fields)
                if result == 0:
                    st.markdown("""
                    <div class="result-safe">
                      <div class="res-icon">✅</div>
                      <div>
                        <div class="res-label">Non-Diabetic — No Risk Detected</div>
                        <div class="res-sub">
                          The screening analysis found no significant indicators of diabetes based on
                          the provided biomarker values. The patient's readings fall within patterns
                          associated with non-diabetic individuals.
                        </div>
                        <div class="res-next">Recommended next steps
                          <ul>
                            <li>Continue routine annual health check-ups</li>
                            <li>Maintain a balanced diet and active lifestyle</li>
                            <li>Re-screen if symptoms or risk factors change</li>
                          </ul>
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-risk">
                      <div class="res-icon">⚠️</div>
                      <div>
                        <div class="res-label">Elevated Diabetes Risk Detected</div>
                        <div class="res-sub">
                          The model has identified patterns strongly associated with diabetes risk
                          in the provided values. This is a screening indicator, not a clinical diagnosis.
                        </div>
                        <div class="res-next">Recommended next steps
                          <ul>
                            <li>Consult a physician or endocrinologist promptly</li>
                            <li>Request an HbA1c or fasting blood glucose confirmation test</li>
                            <li>Review lifestyle factors: diet, exercise, and weight management</li>
                          </ul>
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)
            except ValueError:
                st.error("❌ Invalid input — please ensure all fields contain numeric values only.")

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
      <span style="flex-shrink:0;margin-top:1px;">ℹ️</span>
      <span>
        <strong>Medical Disclaimer:</strong> GlucoSense is a decision-support tool built on machine learning
        and is intended for informational screening purposes only. It does <em>not</em> replace professional
        medical advice, diagnosis, or treatment. Always consult a licensed healthcare provider before acting
        on any result produced by this tool.
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
      GlucoSense · Built with Streamlit &amp; Scikit-learn · No patient data is stored or transmitted
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()