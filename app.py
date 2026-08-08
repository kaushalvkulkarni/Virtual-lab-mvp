import streamlit as st 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.modelselection import train_test_split
from sklearn.metrics import accuracy_score 
st.title("Virtual Bioinformatics Lab")
st.write("Welcome to the first live test of my AI/ML startup")
st.header("Module 1: Data Ingestion")
st.write("upload a biologival dataset (CSV format) to begine analysis")
uploaded_file = st.file_uploader("Drop your CSV file here", type=["CSV"])
if uploaded_file is not None:
   df = pd.read_csv(uploaded_file)
   st.success("Data succesfully loaded!")
   st.dataframe(df)
   
