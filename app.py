import streamlit as st 
import pandas as pd 
st.title("Virtual Bioinformatics Lab")
st.write("Welcome to the first live test of my AI/ML startup")
st.header("Module 1: Data Ingestion")
st.write("upload a biologival dataset (CSV format) to begine analysis")
uploaded_file = st.file_uploder("Drop your CSV file here", type=["CSV"])
if uploded_file is not None:
   df = pd.read_CSV(uplode_file)
   st.success("Data succesfully loaded!")
   st.dataframe("df")
