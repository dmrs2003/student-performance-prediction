import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f5f7ff 0%, #eef2ff 100%);
}

.block-container {
    max-width: 1350px;
    padding-top: 1.5rem;
}

/* HERO SECTION */
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

/* FORM CARD */
.form-card {
    background: white;
    border-radius: 28px;
    padding: 35px;
    margin-top: 28px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.08);
}

/* METRIC CARDS */
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

/* BUTTON */
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
}

.stButton > button:hover {
    background: linear-gradient(90deg, #4338ca, #6d28d9);
    color: white;
}

/* RESULT */
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

label {
    font-weight: 700 !important;
    color: #1e1b4b !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("dataset/StudentPerformanceFactors.csv")

# Encode categorical columns
label_encoders = {}

categorical_cols = [
    "Extracurricular Activities",
    "Internet Access",
    "Parental Involvement",
    "Access to Resources",
    "Motivation Level",
    "Family Income",
    "Teacher Quality",
    "School Type",
    "Peer Influence",
    "Learning Disabilities",
    "Parental Education Level",
    "Distance from Home",
    "Gender"
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features & Target
X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]

# Train Model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <h1>🎓 Student Performance Prediction</h1>
    <p>
    Predict student exam performance using Machine Learning,
    study behavior, and academic factors.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- LAYOUT ----------------
left, right = st.columns([2.2, 1])

# ---------------- FORM ----------------
with left:

    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    st.markdown(
        "<h2 style='color:#1e1b4b; font-size:38px; margin-bottom:25px;'>📌 Enter Student Details</h2>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        hours_studied = st.slider("📚 Hours Studied", 0, 12, 6)

        attendance = st.slider("📅 Attendance (%)", 0, 100, 80)

        previous_scores = st.slider("📝 Previous Scores", 0, 100, 75)

        sleep_hours = st.slider("😴 Sleep Hours", 0, 12, 7)

    with col2:

        papers = st.slider("📄 Practice Papers", 0, 20, 10)

        extracurricular = st.selectbox(
            "🏀 Extracurricular Activities",
            ["Yes", "No"]
        )

        internet = st.selectbox(
            "🌐 Internet Access",
            ["Yes", "No"]
        )

        parental_edu = st.selectbox(
            "🎓 Parental Education",
            ["High School", "College", "Postgraduate"]
        )

    predict_btn = st.button("🚀 Predict Exam Score")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- RIGHT SIDE ----------------
with right:

    st.markdown("""
    <div class="metric-card">
        <div class="metric-heading">🤖 ML Powered</div>
        <div class="metric-text">
        Random Forest Regressor model for accurate exam score prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card">
        <div class="metric-heading">📊 Data Driven</div>
        <div class="metric-text">
        Uses academic and behavioral student data for predictions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card">
        <div class="metric-heading">🎯 Smart Insights</div>
        <div class="metric-text">
        Helps understand factors affecting student performance.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if predict_btn:

    input_data = pd.DataFrame([{
        "Hours Studied": hours_studied,
        "Attendance": attendance,
        "Previous Scores": previous_scores,
        "Sleep Hours": sleep_hours,
        "Sample Question Papers Practiced": papers,
        "Extracurricular Activities":
            label_encoders["Extracurricular Activities"].transform([extracurricular])[0],

        "Internet Access":
            label_encoders["Internet Access"].transform([internet])[0],

        "Parental Education Level":
            label_encoders["Parental Education Level"].transform([parental_edu])[0],
    }])

    # Add missing columns
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[X.columns]

    prediction = model.predict(input_data)[0]

    st.markdown(f"""
    <div class="result-card">
        <div class="result-title">
            Predicted Exam Score
        </div>

        <div class="result-score">
            {prediction:.2f}
        </div>

        <div class="result-status">
            ✅ High Performance
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(int(prediction))

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
💜 Built with Python • Scikit-learn • Streamlit
</div>
""", unsafe_allow_html=True)