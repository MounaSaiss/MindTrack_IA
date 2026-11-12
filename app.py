
import streamlit as st
import pandas as pd
import joblib
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# Configuration
st.set_page_config(page_title="MindTrack AI", layout="centered")

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))
from evaluate import evaluate_model

MODEL_PATH = ROOT / "models" / "student_mental_health_model.pkl"


# Style
st.markdown("""
<style>
.stApp {background:#F4F7FA; color:#263445;}
.block-container {max-width:1100px; padding:2rem; margin:auto;}
.hero {
    background:linear-gradient(120deg,#17324D,#246B78);
    padding:25px; border-radius:12px; text-align:center; color:white;
}
.hero h1 {color:white; margin:0;}
.hero p {color:#DCE8EF;}
.section-title {color:#17324D; font-size:21px; font-weight:bold; text-align:center;}
h2,h3 {color:#17324D;}
div[data-testid="stMetric"],[data-testid="stForm"] {
    background:white; padding:15px; border-radius:10px; border:1px solid #DCE5EC;
}
div[data-testid="stMetricValue"] {color:#17324D;}
div.stFormSubmitButton > button {background:#246B78; color:white; border-radius:8px;}
div.stFormSubmitButton > button:hover {background:#17324D; color:white;}
.stTabs [aria-selected="true"] {color:#246B78;}
</style>
""", unsafe_allow_html=True)


# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def get_evaluation():
    return evaluate_model()

model = load_model()
metrics, cm, _, _ = get_evaluation()


# Header and tabs
st.markdown("""
<div class="hero">
    <h1>MindTrack AI</h1>
    <p>Student Mental Health Risk Prediction Platform</p>
</div>
""", unsafe_allow_html=True)

dashboard, prediction = st.tabs(["Dashboard", "Risk Prediction"])


# Dashboard
with dashboard:
    st.markdown('<p class="section-title">Model Performance Overview</p>',
                unsafe_allow_html=True)

    for col, (name, value) in zip(st.columns(5), metrics.items()):
        col.metric(name, f"{value:.2%}")

    left, right = st.columns(2)

    with left:
        st.subheader("Performance Metrics")
        st.bar_chart(pd.DataFrame({"Score": metrics}))

    with right:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        sns.heatmap(
            cm, annot=True, fmt="d",
            cmap="YlGnBu", cbar=False, ax=ax
        )
        ax.set(xlabel="Predicted Risk", ylabel="Actual Risk")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    st.caption("Evaluation results are calculated using the test dataset.")


# Prediction form
with prediction:
    st.markdown('<p class="section-title">Student Risk Assessment</p>',
                unsafe_allow_html=True)
    st.write("Enter the student's information to predict the mental health risk.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Personal Information")
            age = st.number_input("Age", min_value=0, value=22)
            year = st.number_input("Year of Study", min_value=0, value=2)
            gender = st.selectbox("Gender", ["Male", "Female"])
            job = st.selectbox("Part Time Job", ["Yes", "No"])
            history = st.selectbox("Previous Mental Health History", ["Yes", "No"])

        with col2:
            st.subheader("Lifestyle Information")
            sleep = st.number_input("Sleep Hours", min_value=0.0, value=7.0)
            study = st.number_input("Study Hours Per Week", min_value=0.0, value=20.0)
            gpa = st.number_input("GPA", min_value=0.0, value=3.0)
            physical = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
            diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])

        st.divider()
        st.subheader("Daily Habits and Support")
        col3, col4, col5 = st.columns(3)

        with col3:
            social = st.number_input("Social Media Hours", min_value=0.0, value=3.0)
            screen = st.number_input("Screen Time Hours", min_value=0.0, value=6.0)

        with col4:
            financial = st.number_input("Financial Stress", min_value=0.0, value=2.0)
            family = st.number_input("Family Support", min_value=0.0, value=3.0)

        with col5:
            attendance = st.number_input("Attendance Rate", min_value=0.0, max_value=100.0, value=85.0)

        submitted = st.form_submit_button("Predict Mental Health Risk", use_container_width=True)

    if submitted:
        data = pd.DataFrame([{
            "Age": age, "Year_of_Study": year, "Gender": gender,
            "Sleep_Hours": sleep, "Study_Hours_Per_Week": study,
            "GPA": gpa, "Physical_Activity": physical, "Diet_Quality": diet,
            "Social_Media_Hours": social, "Screen_Time_Hours": screen,
            "Financial_Stress": financial, "Family_Support": family,
            "Attendance_Rate": attendance, "Part_Time_Job": job,
            "Previous_Mental_Health_History": history
        }])

        result = model.predict(data)[0]
        probabilities = model.predict_proba(data)[0]

        st.divider()
        st.subheader("Prediction Result")
        st.success(f"Predicted Risk Level: {result}")

        st.subheader("Prediction Probabilities")
        st.bar_chart(pd.DataFrame(
            {"Probability": probabilities}, index=model.classes_
        ))

        st.caption("This prediction is an estimate, not a clinical diagnosis.")


# Footer
st.divider()
st.markdown("""
<div style="text-align:center;color:#718096;font-size:13px;">
MindTrack AI | Student Mental Health Risk Assessment
</div>
""", unsafe_allow_html=True)
