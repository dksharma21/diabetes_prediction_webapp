import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/gopiashokan/dataset/main/diabetes_prediction_dataset.csv")

# Preprocessing
enc = OrdinalEncoder()
df["smoking_history"] = enc.fit_transform(df[["smoking_history"]])
df["gender"] = enc.fit_transform(df[["gender"]])

# Define variables
x = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Model
model = RandomForestClassifier().fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)

# Streamlit app layout
st.set_page_config(page_title='Diabetes Prediction', page_icon=':dna:')
st.markdown('<h1 style="text-align: center;">Diabetes Prediction</h1>', unsafe_allow_html=True)

# Add image under the main heading
image_url = "images.png"  # Replace with your image URL
st.image(image_url, width=700)  # Set the width as needed

col1, col2 = st.columns(2, gap='large')

with col1:
    gender = st.selectbox(label='Gender', options=['Male', 'Female', 'Other'])
    gender_dict = {'Female':0.0, 'Male':1.0, 'Other':2.0}

    age = st.number_input(label='Age ', min_value=1, max_value=120, step=1)

    hypertension = st.selectbox(label='Hypertension', options=['No', 'Yes'])
    hypertension_dict = {'No':0, 'Yes':1}

    heart_disease = st.selectbox(label='Heart Disease', options=['No', 'Yes'])
    heart_disease_dict = {'No':0, 'Yes':1}

with col2:
    smoking_history = st.selectbox(label='Smoking History', options=['Never', 'Current', 'Former', 'Ever', 'Not Current', 'No Info'])
    smoking_history_dict = {'Never':4.0, 'No Info':0.0, 'Current':1.0, 'Former':3.0, 'Ever':2.0, 'Not Current':5.0}

    bmi = st.number_input(label='BMI (0-100)', min_value=0.0, max_value=100.0, step=0.1)

    hba1c_level = st.number_input(label='HbA1c - Hemoglobin Level (0-9)', min_value=0.0, max_value=9.0, step=0.1)

    blood_glucose_level = st.number_input(label='Blood Glucose Level (80-200)', min_value=80, max_value=200, step=1)

st.write('')
st.write('Made by Deepender(23112)')
col1, col2 = st.columns([0.438, 0.562])
with col2:
    submit = st.button(label='Predict')

if submit:
    try:
        user_data = np.array([[gender_dict[gender], age, hypertension_dict[hypertension], heart_disease_dict[heart_disease],
                               smoking_history_dict[smoking_history], bmi, hba1c_level, blood_glucose_level]])
        test_result = model.predict(user_data)

        if test_result[0] == 0:
            st.success('Diabetes Result: Negative')
            st.balloons()
        else:
            st.error('Diabetes Result: Positive (Please Consult with Doctor)')
    except:
        st.warning('Please fill all required information')
