import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt                        
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
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
   algorithm = st.selectbox("Select AI Algorithm", ["Random Forest", "Logistic Regression"])
   if st.button("Run Machine Learning Model"):
      if "id" in df.columns:
         df = df.drop(columns=["id"])
      if "unnamed: 32" in df.columns:
         df = df.drop(columns=["unnamed: 32"])  
      df = df.dropna(axis=1, how='all')
      df = df.fillna(0)
      X = df.drop(columns=[target_column])
      y = df[target_column]
      X = pd.get_dummies(X)
      if y.dtype ==  "objects":
         y = y.astype('category').cat.codes
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      if algorithm == "Random Forest":
         model = RandomForestClassifier(random_state=42)
      elif algorithm == "Logistic Regression":
         model = LogisticRegression(max_iter=2000)
      model.fit(X_train, y_train)
      predictions = model.predict(X_test)
      accuracy = accuracy_score(y_test, predictions)
      st.write(f"**Model Accuracy:**{accuracy * 100:.2f}%")
      st.success("AI successfully trained on paitients data!")
      st.subheader("Live Predictions Results")
      st.write("Comparing  the AI digonosis to the actual patient records")
      results_df = pd.DataFrame({
         "Acctual Outcome": y_test,
         "AI Diagnosis": predictions
      })
      results_df = results_df.sort_index()
      col1, col2 = st.columns(2)
      with col1:
         st.subheader("  Patient Data")
         with st.extender("Show Data"):
            st.dataframe(results_df)
      with col2:
         st.subheader("  Dashboard")
         tab1, tab2, tab3  = st.tabs(["Pie, Line, Bar"])
         with tab1:
            fig, ax = plt.subplots()
            results_df['AI Diagnosis'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, shadow=True)
            st.pyplot(fig)
         with tab2:
            st.line_chart(results_df['AI Diagnosis'].value_counts())
         with tab3:
            st.bar_chart(results_df['AI Diagnosis'].value_counts())     
    
   
