import streamlit as st 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
st.title("Virtual Bioinformatics Lab")
st.write("Welcome to the first live test of my AI/ML startup")
st.header("Module 1: Data Ingestion")
st.write("upload a biologival dataset (CSV format) to begine analysis")
uploaded_file = st.file_uploader("Drop your CSV file here", type=["CSV"])
if uploaded_file is not None:
   df = pd.read_csv(uploaded_file)
   df = df.dropna(axis=1, how='all')
   df = df.fillna(0)
   st.success("Data succesfully loaded!")
   st.dataframe(df)
   st.header("Module 2: Disease prediction model")
   st.write("Train a Ramdom Forest AI on this dataset.")
   target_column = st.selectbox("Select the column you want to predict:", df.columns) 
   if st.button("Run Machine Learning Model"):
      df = df.dropna(axis=1, how='all')
      df = df.fillna(0)
      X = df.iloc[:,  :-1]
      y = df.iloc[:, -1]
      X = pd.get_dummies(X)
      if y.dtype ==  "objects":
         y = y.astype('category').cat.codes
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      model = RandomForestClassifier()
      model.fit(X_train, y_train)
      predictions = model.predict(X_test)
      accuracy = accuracy_score(y_test, predictions)
      st.write(f"**Model Accuracy:**{accuracy * 100:.2f}%")
      st.success("AI successfully trained on paitients data!")
      
   
