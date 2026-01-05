import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("Student Performance Prediction using Student ID")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Remove column if exists
    if "parent_education_level" in df.columns:
        df = df.drop("parent_education_level", axis=1)

    # Clean data
    df = df.drop_duplicates()
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Trend score
    df["trend_score"] = df["test3"] - df["test1"]

    # Label encoding
    le = LabelEncoder()
    df["gender"] = le.fit_transform(df["gender"])

    # Assign label
    def assign_label(x):
        if x > 5:
            return "Improving"
        elif x < -5:
            return "Declining"
        else:
            return "Stable"

    df["performance_label"] = df["trend_score"].apply(assign_label)

    # Model Training (runs once)
    features = ['attendance','study_hours','test1','test2','test3','gender']
    X = df[features]
    y = df["performance_label"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42
    )

    model.fit(X_train, Y_train)

    # -----------------------------------------------------------
    #             STUDENT ID PREDICTION ONLY
    st.subheader("🎯 Predict Performance by Student ID")

    if "student_id" in df.columns:

        student_ids = sorted(df["student_id"].unique())
        selected_id = st.selectbox("Select Student ID", student_ids)

        if st.button("Predict Performance"):
            student_row = df[df["student_id"] == selected_id].iloc[0]

            input_data = np.array([[
                student_row["attendance"],
                student_row["study_hours"],
                student_row["test1"],
                student_row["test2"],
                student_row["test3"],
                student_row["gender"]
            ]])

            result = model.predict(input_data)[0]

            st.success(f"🧾 Student ID: {selected_id}\n### Prediction: *{result}*")

    else:
        st.error("⚠ student_id column not found in dataset")
