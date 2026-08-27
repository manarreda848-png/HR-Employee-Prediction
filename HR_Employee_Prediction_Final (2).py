
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="HR Employee Prediction",
    page_icon="👥",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

features = [
    "Age",
    "MonthlyIncome",
    "JobLevel",
    "YearsAtCompany",
    "JobInvolvement"
]

model_files = {
    "Attrition": "final_attrition_model.pkl",
    "Overtime": "final_overtime_model.pkl",
    "Job Satisfaction": "final_satisfaction_model.pkl",
    "Performance Rating": "final_performance_model.pkl"
}


@st.cache_resource
def load_models():
    models = {}

    for name, file in model_files.items():
        path = MODEL_DIR / file

        if path.exists():
            models[name] = joblib.load(path)

    return models


@st.cache_data
def load_data():
    path = MODEL_DIR / "employee_analysis_data.csv"

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


models = load_models()
df = load_data()

# Make sure training and website use the same five features.
FEATURE_FILE = MODEL_DIR / "feature_columns.pkl"
if FEATURE_FILE.exists():
    trained_features = joblib.load(FEATURE_FILE)
    if list(trained_features) != features:
        st.error("Feature mismatch: the training models and website must use the same five features.")
        st.stop()


# Sidebar
st.sidebar.title("HR Employee Prediction")

page = st.sidebar.radio(
    "Menu",
    [
        "Dashboard",
        "Employee Prediction",
        "Training Results",
        "HR Recommendations",
        "Employee Similarity",
        "Promotion Insights"
    ]
)


# Dashboard
if page == "Dashboard":

    st.title("HR Employee Prediction")
    st.write("HR analytics and employee prediction dashboard.")

    if df.empty:
        st.warning("Run the training notebook first.")
    else:
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Employees", len(df))
        col2.metric("Average Age", round(df["Age"].mean(), 1))
        col3.metric(
            "Average Income",
            f"{df['MonthlyIncome'].mean():,.0f}"
        )
        col4.metric(
            "Average Years",
            round(df["YearsAtCompany"].mean(), 1)
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Age Distribution")
            age_data = df["Age"].value_counts().sort_index()
            st.bar_chart(age_data)

        with col2:
            st.subheader("Monthly Income")
            income_data = df["MonthlyIncome"].sort_values().reset_index(drop=True)
            st.line_chart(income_data)


# Employee Prediction
elif page == "Employee Prediction":

    st.title("Employee Prediction")
    st.write("Enter a new employee profile. The employee does not need to exist in the dataset.")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=70, value=30)
        income = st.number_input("Monthly Income", min_value=100, max_value=100000, value=5000, step=100)
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)

    with col2:
        years = st.number_input("Years at Company", min_value=0, max_value=50, value=5)
        involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], index=2)

    if st.button("Predict Employee", type="primary"):
        if len(models) < 4:
            st.error("Models are missing. Run the training notebook first.")
        else:
            new_employee = pd.DataFrame([{
                "Age": age,
                "MonthlyIncome": income,
                "JobLevel": job_level,
                "YearsAtCompany": years,
                "JobInvolvement": involvement
            }])

            attrition_model = models["Attrition"]
            attrition_prediction = attrition_model.predict(new_employee[features])[0]

            if hasattr(attrition_model, "predict_proba"):
                classes = list(attrition_model.classes_)
                probs = attrition_model.predict_proba(new_employee[features])[0]
                if 1 in classes:
                    probability = float(probs[classes.index(1)])
                elif "Yes" in classes:
                    probability = float(probs[classes.index("Yes")])
                else:
                    probability = float(max(probs))
            else:
                probability = float(attrition_prediction)

            overtime_prediction = models["Overtime"].predict(new_employee[features])[0]
            satisfaction_prediction = models["Job Satisfaction"].predict(new_employee[features])[0]
            performance_prediction = models["Performance Rating"].predict(new_employee[features])[0]

            if probability >= 0.60:
                risk = "High"
            elif probability >= 0.30:
                risk = "Medium"
            else:
                risk = "Low"

            overtime_text = "Yes" if overtime_prediction == 1 else "No"

            st.divider()
            st.subheader("Employee Prediction Summary")

            summary = pd.DataFrame({
                "Prediction": [
                    "Attrition Probability", "Attrition Risk", "Overtime",
                    "Job Satisfaction", "Performance Rating"
                ],
                "Result": [
                    f"{probability * 100:.1f}%", risk, overtime_text,
                    str(satisfaction_prediction), str(performance_prediction)
                ]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

            st.subheader("Prediction Explanation")
            if risk == "High":
                st.write("The model found a higher attrition risk for this profile. Retention and workload should be reviewed.")
            elif risk == "Medium":
                st.write("The model found a moderate attrition risk for this profile. Regular follow-up is recommended.")
            else:
                st.write("The model found a lower attrition risk for this profile. Normal monitoring can continue.")

            st.subheader("HR Recommendation")
            recommendations = []
            if overtime_text == "Yes":
                recommendations.append("Consider workload redistribution and overtime review.")
            if involvement <= 2:
                recommendations.append("Consider engagement activities and employee development.")
            try:
                if float(satisfaction_prediction) <= 2:
                    recommendations.append("Consider employee support and satisfaction follow-up.")
            except (ValueError, TypeError):
                pass
            if risk == "High":
                recommendations.append("Prioritize retention review for this employee.")
            try:
                if float(performance_prediction) >= 4 and years >= 3:
                    recommendations.append("Consider development or promotion review.")
            except (ValueError, TypeError):
                pass
            if not recommendations:
                recommendations.append("Continue normal employee monitoring and development support.")
            for item in recommendations:
                st.write("- " + item)

# Training Results
elif page == "Training Results":

    st.title("Training Results & Model Comparison")
    st.write("Each prediction target has its own best-performing model based on F1-Score.")

    summary_path = MODEL_DIR / "overall_results.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        st.subheader("Best Model Summary")
        st.dataframe(summary[["Prediction", "Best Model"]], use_container_width=True, hide_index=True)
    else:
        st.warning("Run the training notebook first.")

    result_files = {
        "Attrition": "attrition_results.csv",
        "Overtime": "overtime_results.csv",
        "Job Satisfaction": "satisfaction_results.csv",
        "Performance Rating": "performance_results.csv"
    }

    for name, file in result_files.items():
        path = MODEL_DIR / file
        if path.exists():
            st.subheader(name)
            result = pd.read_csv(path)
            st.dataframe(result.round(3), use_container_width=True, hide_index=True)
            if not result.empty:
                st.success(f"Best Model: {result.iloc[0]['Model']} | F1-Score: {result.iloc[0]['F1-Score']:.3f}")
                st.bar_chart(result.set_index("Model")["F1-Score"])

# HR Recommendations
elif page == "HR Recommendations":

    st.title("HR Recommendations")
    st.write("Rule-based HR insights derived from employee attributes.")

    if df.empty:
        st.warning("Run the training notebook first.")
    else:
        recommendations = []
        for idx, row in df.iterrows():
            actions = []
            if "OverTime" in row and row["OverTime"] == "Yes":
                actions.append("Review workload")
            if "JobSatisfaction" in row and row["JobSatisfaction"] <= 2:
                actions.append("Improve job satisfaction")
            if "JobInvolvement" in row and row["JobInvolvement"] <= 2:
                actions.append("Increase employee engagement")
            if "YearsSinceLastPromotion" in row and row["YearsSinceLastPromotion"] >= 4 and row["PerformanceRating"] >= 3:
                actions.append("Review promotion opportunity")
            if "TrainingTimesLastYear" in row and row["TrainingTimesLastYear"] <= 1:
                actions.append("Consider additional training")
            if actions:
                recommendations.append({"Employee Index": idx, "Recommendations": " • ".join(actions)})

        rec_df = pd.DataFrame(recommendations)
        if len(rec_df):
            st.dataframe(rec_df.head(100), use_container_width=True, hide_index=True)
            st.info(f"{len(rec_df)} employees have at least one rule-based HR recommendation.")
        else:
            st.success("No rule-based HR actions were triggered.")

# Employee Similarity
elif page == "Employee Similarity":

    st.title("Employee Similarity")

    if df.empty:

        st.warning("Run the training notebook first.")

    else:

        st.write(
            "Enter a new employee profile to find similar employees."
        )

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age",
                18,
                70,
                30,
                key="similar_age"
            )

            income = st.number_input(
                "Monthly Income",
                100,
                100000,
                5000,
                100,
                key="similar_income"
            )

            level = st.selectbox(
                "Job Level",
                [1, 2, 3, 4, 5],
                1,
                key="similar_level"
            )

        with col2:

            years = st.number_input(
                "Years at Company",
                0,
                50,
                5,
                key="similar_years"
            )

            involvement = st.selectbox(
                "Job Involvement",
                [1, 2, 3, 4],
                2,
                key="similar_involvement"
            )

        if st.button("Find Similar Employees"):

            scaler = StandardScaler()

            values = scaler.fit_transform(
                df[features]
            )

            new_values = scaler.transform(
                pd.DataFrame([{
                    "Age": age,
                    "MonthlyIncome": income,
                    "JobLevel": level,
                    "YearsAtCompany": years,
                    "JobInvolvement": involvement
                }])[features]
            )

            distances = np.linalg.norm(
                values - new_values,
                axis=1
            )

            result = df.copy()

            result["Distance"] = distances

            result = result.sort_values(
                "Distance"
            ).head(5)

            st.dataframe(
                result[features + ["Distance"]].round(2),
                use_container_width=True,
                hide_index=True
            )


# Promotion Insights
elif page == "Promotion Insights":

    st.title("Promotion Insights")

    if df.empty:

        st.warning("Run the training notebook first.")

    else:

        data = df.copy()

        score = pd.Series(
            0.0,
            index=data.index
        )

        if "JobLevel" in data:
            score += (
                data["JobLevel"]
                / data["JobLevel"].max()
                * 30
            )

        if "YearsAtCompany" in data:
            score += (
                data["YearsAtCompany"].clip(0, 15)
                / 15
                * 25
            )

        if "JobInvolvement" in data:
            score += (
                data["JobInvolvement"]
                / 4
                * 20
            )

        if "PerformanceRating" in data:
            score += (
                data["PerformanceRating"]
                / 4
                * 25
            )

        data["Promotion Recommendation Score"] = score.round(1)

        st.caption("This score is a decision-support indicator, not an automatic promotion decision.")

        columns = [
            "Age",
            "JobLevel",
            "YearsAtCompany",
            "JobInvolvement",
            "PerformanceRating",
            "Promotion Recommendation Score"
        ]

        columns = [
            column for column in columns
            if column in data.columns
        ]

        st.dataframe(
            data.sort_values(
                "Promotion Recommendation Score",
                ascending=False
            ).head(10)[columns],
            use_container_width=True,
            hide_index=True
        )
