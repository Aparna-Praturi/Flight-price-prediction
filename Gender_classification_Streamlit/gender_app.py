# Import necessary libraries
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from datetime import datetime
import joblib
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'py_files'))
from gender_classification import Namesplitter, IsVowelEnd, TfidfAndDense



def test_predict(processor_path, model_path, test_data):

    """ This function takes the ppreprocessor, model , input data and predicts the gender"""

    # load preprocessor and model
    with open(processor_path, 'rb') as file:
        processor = joblib.load(file)

    with open(model_path, 'rb') as file:
        model = joblib.load(file)

    # preprocess the data
    x_ver = processor.transform(test_data)

    # predict gender
    a = model.predict(x_ver)

    # make a dictionary
    diction = dict(zip(test_data['name'], a));

    # map values
    class_map = {0: "Female", 1: "Male"}

    return [{key: class_map[value]} for key, value in diction.items()]


# Load model and preprocessor
processor_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'gender_class_preprocessor.joblib')
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'gender_classifier.joblib')


# Stramlit UI for data

# Streamlit UI
st.set_page_config(page_title="Gender Prediction", layout="wide")

# Title of the app
st.title("Gender prediction System")

# Input names
user_input = st.text_input("Enter list of names separated by comma")

# convert to list
if user_input:
    user_list = [item.strip() for item in user_input.split(',')]
    st.write("Your list of strings:", user_input)

    # Convert into a df
    df = pd.DataFrame({'name':user_list})

# predict gender
if st.button('Predict gender'):
    if not user_input:
        st.error("Please enter at least one name!")  
    else:
     results = test_predict(processor_path, model_path, df)
     flattened_results = {key: value for result in results for key, value in result.items()}    
     results_df = pd.DataFrame(list(flattened_results.items()), columns=['Name', 'Gender'])
     
     
     st.write("Gender Identification Results:")
     st.table(results_df)






