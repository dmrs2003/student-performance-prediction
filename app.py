import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7ff 0%, #eef2ff 100%);
}

.block-container {
    max-width: 1350px;
    padding-top: 1.5rem;
}

.hero {
    background: linear-gradient(135deg, #4338ca, #7c3aed);
    border-radius: 32px;
    padding: 45px;
    color: white;
    box-shadow: 0 20px 45px rgba(79,70,229,0.28);
}

.hero h1 {
    font-size: 58px;
    font-weight: 900;
    margin-bottom: 12px;
    color: white;
}

.hero p {
    font-size: 21px;
    color: #f3f4ff;
}

.badges {
    display: flex;
    gap: 15px;
    margin-top: 30px;
    flex-wrap: wrap;
}

.badge {
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.25);
    padding: 15px 20px;
    border-radius: 18px;
    color: white;
    font-weight: 700;
    font-size: 16px;
}

.form-card {
    background: white;
    border-radius: 28px;
    padding: 35px;
    margin-top: 28px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.08);
}

.metric-card {
    background: white;
    border-radius: 24px;
    padding: 28px;
    margin-top: 28px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.07);
}

.metric-heading {
    font-size: 28px;
    font-weight: 800;
    color: #1e1b4b;
}

.metric-text {
    color: #64748b;
    font-size: 17px;
    margin-top: 10px;
}

label {
    font-weight: 700 !important;
    color: #1e1b4b !important;
}

.stButton > button {
    width: 100%;
    height: 60px;
    border-radius: 18px;
    border: none;
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    color: white;
    font-size: 20px;
    font-weight: 800;
    margin-top: 20px;
    box-shadow: 0 10px 25px rgba(79,70,229,0.25);
}

.stButton > button:hover {
    background: linear-gradient(90deg, #4338ca, #6d28d9);
    color: white;
}

.result-card {
    background: white;
    border-radius: 28px;
    padding: 30px;
    margin-top: 25px;
    text-align: center;
    box-shadow: 0 15px 35px rgba(79,70,229,0.12);
}

.result-title {
    color: #4f46e5;
    font-size: 24px;
    font-weight: 700;
}

.result-score {
    font-size: 72px;
    font-weight: 900;
    color: #4338ca;
}

.result-status {
    color: #16a34a;
    font-size: 22px;
    font-weight: 700;
}

.footer {
    text-align: center;
    margin-top: 40px;
    color: #64748b;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

df = pd.read_csv("dataset/StudentsPerformance.csv")

y = df["math score"]
X = df.drop("math score", axis=1)
X = pd.get_dummies(X)

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

st.markdown("""
<div class="hero">
    <h1>🎓 Student Performance Prediction</h1>
    <p>Predict student math performance using Machine Learning, academic scores, and learning factors.</p>

</div>
""", unsafe_allow_html=True)

left, right = st.columns([2.2, 1])

with left:
    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    st.markdown(
        "<h2 style='color:#1e1b4b; font-size:38px; margin-bottom:25px;'>👤 Enter Student Details</h2>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["female", "male"])
        race = st.selectbox("Race / Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
        lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])

    with col2:
        education = st.selectbox(
            "Parental Education",
            [
                "some high school",
                "high school",
                "some college",
                "associate's degree",
                "bachelor's degree",
                "master's degree"
            ]
        )
        prep = st.selectbox("Test Preparation Course", ["none", "completed"])

    reading = st.slider("📖 Reading Score", 0, 100, 70)
    writing = st.slider("✍ Writing Score", 0, 100, 70)

    predict_btn = st.button("🚀 Predict Math Score")

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-heading">🧠 ML Powered</div>
        <div class="metric-text">Random Forest Regressor model for accurate score prediction.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card">
        <div class="metric-heading">📊 Data Driven</div>
        <div class="metric-text">Uses academic and demographic student data for predictions.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card">
        <div class="metric-heading">🎯 Smart Insights</div>
        <div class="metric-text">Helps educators and parents understand performance trends.</div>
    </div>
    """, unsafe_allow_html=True)

if predict_btn:
    input_data = pd.DataFrame([{
        "gender": gender,
        "race/ethnicity": race,
        "parental level of education": education,
        "lunch": lunch,
        "test preparation course": prep,
        "reading score": reading,
        "writing score": writing
    }])

    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(input_encoded)[0]

    st.markdown(f"""
    <div class="result-card">
        <div class="result-title">Predicted Math Score</div>
        <div class="result-score">{prediction:.2f}</div>
        <div class="result-status">✅ High Performance</div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(int(prediction))

st.markdown("""
<div class="footer">
💜 Built with Python • Scikit-learn • Streamlit
</div>
""", unsafe_allow_html=True)