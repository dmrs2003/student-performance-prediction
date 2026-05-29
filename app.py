import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #f5f7ff 0%, #eef2ff 100%); }
.block-container { max-width: 1300px; padding-top: 1.5rem; }
.hero {
    background: linear-gradient(135deg, #4338ca, #7c3aed);
    border-radius: 30px;
    padding: 40px;
    color: white;
    box-shadow: 0 20px 45px rgba(79,70,229,0.25);
}
.hero h1 { font-size: 52px; font-weight: 900; color: white; }
.hero p { font-size: 20px; color: #f3f4ff; }
.card {
    background: white;
    border-radius: 26px;
    padding: 32px;
    margin-top: 25px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.08);
}
.info-card {
    background: white;
    border-radius: 22px;
    padding: 25px;
    margin-top: 25px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.07);
}
.info-card h3 { color: #1e1b4b; font-size: 26px; }
.info-card p { color: #64748b; font-size: 16px; }
label { color: #1e1b4b !important; font-weight: 700 !important; }
.stButton > button {
    height: 58px;
    border-radius: 16px;
    border: none;
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    color: white;
    font-size: 19px;
    font-weight: 800;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #4338ca, #6d28d9);
    color: white;
}
.result {
    background: white;
    border-radius: 26px;
    padding: 30px;
    margin-top: 25px;
    text-align: center;
    box-shadow: 0 15px 35px rgba(79,70,229,0.15);
}
.result h2 { color: #4f46e5; }
.result h1 { color: #4338ca; font-size: 70px; font-weight: 900; }
.footer { text-align: center; margin-top: 35px; color: #64748b; }
</style>
""", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv("dataset/StudentPerformanceFactors.csv")
df = df.dropna()

categorical_cols = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Motivation_Level",
    "Internet_Access",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender"
]

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

st.markdown("""
<div class="hero">
    <h1>🎓 Student Performance Prediction</h1>
    <p>Predict student exam performance using Machine Learning and academic factors.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([2.2, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h2 style='color:#1e1b4b;'>📌 Enter Student Details</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        hours = st.slider("📚 Hours Studied", 0, 40, 20)
        attendance = st.slider("📅 Attendance (%)", 0, 100, 80)
        previous = st.slider("📝 Previous Scores", 0, 100, 75)
        sleep = st.slider("😴 Sleep Hours", 0, 12, 7)
        tutoring = st.slider("👨‍🏫 Tutoring Sessions", 0, 10, 2)

    with col2:
        parental = st.selectbox("👨‍👩‍👧 Parental Involvement", label_encoders["Parental_Involvement"].classes_)
        resources = st.selectbox("📚 Access to Resources", label_encoders["Access_to_Resources"].classes_)
        extra = st.selectbox("🏀 Extracurricular Activities", label_encoders["Extracurricular_Activities"].classes_)
        motivation = st.selectbox("🔥 Motivation Level", label_encoders["Motivation_Level"].classes_)
        internet = st.selectbox("🌐 Internet Access", label_encoders["Internet_Access"].classes_)
        parent_edu = st.selectbox("🎓 Parental Education", label_encoders["Parental_Education_Level"].classes_)

    predict = st.button("🚀 Predict Exam Score", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="info-card">
        <h3>🤖 ML Powered</h3>
        <p>Random Forest Regressor is used for prediction.</p>
    </div>
    <div class="info-card">
        <h3>📊 Data Driven</h3>
        <p>Uses study hours, attendance, previous scores and learning factors.</p>
    </div>
    <div class="info-card">
        <h3>🎯 Smart Insights</h3>
        <p>Helps understand factors affecting exam performance.</p>
    </div>
    """, unsafe_allow_html=True)

if predict:
    input_data = pd.DataFrame([{
        "Hours_Studied": hours,
        "Attendance": attendance,
        "Parental_Involvement": label_encoders["Parental_Involvement"].transform([parental])[0],
        "Access_to_Resources": label_encoders["Access_to_Resources"].transform([resources])[0],
        "Extracurricular_Activities": label_encoders["Extracurricular_Activities"].transform([extra])[0],
        "Sleep_Hours": sleep,
        "Previous_Scores": previous,
        "Motivation_Level": label_encoders["Motivation_Level"].transform([motivation])[0],
        "Internet_Access": label_encoders["Internet_Access"].transform([internet])[0],
        "Tutoring_Sessions": tutoring,
        "Family_Income": 0,
        "Teacher_Quality": 0,
        "School_Type": 0,
        "Peer_Influence": 0,
        "Physical_Activity": 3,
        "Learning_Disabilities": 0,
        "Parental_Education_Level": label_encoders["Parental_Education_Level"].transform([parent_edu])[0],
        "Distance_from_Home": 0,
        "Gender": 0
    }])

    input_data = input_data[X.columns]

    prediction = model.predict(input_data)[0]

    st.markdown(f"""
    <div class="result">
        <h2>Predicted Exam Score</h2>
        <h1>{prediction:.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

    st.progress(int(prediction))

st.markdown("""
<div class="footer">
💜 Built with Python • Scikit-learn • Streamlit
</div>
""", unsafe_allow_html=True)