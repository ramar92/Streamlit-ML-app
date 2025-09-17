import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix 

st.title("ML App") 
st.header("Upload Your Dataset") 
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"]) 
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.subheader("Data Preview") 
    st.dataframe(df.head())   

    #Preprocessing 
    st.header("Data Preprocessing") 
    st.write("Finding Missing Values") 
    st.write(df.isnull().sum())

    option = st.selectbox("Handle Missing Values", 
                          ["Do Nothing", "Drop Rows", "Fill Mean", "Fill Median", "Fill Mode"])

    df_processed = df.copy()
    if option == "Drop Rows":
        df_processed = df.dropna()
    elif option == "Fill Mean":
        df_processed = df.fillna(df.select_dtypes(include="number").mean())
    elif option == "Fill Median":
        df_processed = df.fillna(df.select_dtypes(include="number").median())
    elif option == "Fill Mode":
        for col in df.columns:
            df_processed[col] = df[col].fillna(df[col].mode()[0]) 
    st.write("After Preprocessing")
    st.dataframe(df_processed.head()) 

    # Correlation Heatmap 
    st.subheader("Correlation Heatmap") 
    numeric_df = df_processed.select_dtypes(include="number") 
    if numeric_df.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # ML Model Training 
    st.header("Train a Machine Learning Model") 
    target = st.selectbox("Select Target Column", df_processed.columns)
    features = [col for col in df_processed.columns if col != target] 

    # Encode categorical data 
    le = LabelEncoder() 
    for col in df_processed.columns:
        if df_processed[col].dtype == "object":
            df_processed[col] = le.fit_transform(df_processed[col].astype(str)) 
    
    X = df_processed[features] 
    y = df_processed[target] 

    X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42) 
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test) 
    acc = accuracy_score(y_test, y_pred) 
    st.write(f"Model Trained! Accuracy: **{acc:.2f}**")  

    cm = confusion_matrix(y_test, y_pred) 
    fig, ax = plt.subplots() 
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Prediction 
    st.header("Make Predictions")
    input_data = {} 
    for col in features:
        if df[col].dtype == "object":
            input_data[col] = st.selectbox(f"Select {col}", df[col].unique())
        else:
            input_data[col] = st.number_input(f"Enter {col}", float(df[col].min()), float(df[col].max()))
    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        # Encode same way
        for col in input_df.columns:
            if input_df[col].dtype == "object":
                input_df[col] = le.fit_transform(input_df[col].astype(str))
        prediction = model.predict(input_df)[0]
        st.success(f"Prediction: {prediction}")


    























    