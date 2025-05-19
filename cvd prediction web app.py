# -*- coding: utf-8 -*-
"""
Created on Sun May 11 14:08:26 2025

@author: NUGGET
"""

import numpy as np
import pickle
import streamlit as st
import pandas as pd
#loading the save model
loaded_model = pickle.load(open("C:/Users/NUGGET/Desktop/ML MODELS/cardiovascular_trained_model.sav", "rb"))

# Page config
st.set_page_config(page_title="CVD Risk Prediction", layout="wide")

# creating function for prediction
def cvd_prediction(input_data):
    columns = ['age','gender','height', 'weight', 'Systolic blood pressure', 'Diastolic blood pressure',
               'cholesterol', 'Glucose Level', 'smoke', 'Alcohol intake', 'Physical activity']
    input_df = pd.DataFrame([input_data], columns=columns)
    prediction = loaded_model.predict(input_df)[0]
    probability = loaded_model.predict_proba(input_df)[0]
    return prediction, probability

    #print(prediction)
    
    
    # Sidebar section
with st.sidebar:
     
     st.title("🧬 About This App")
     st.markdown("""This web app uses a trained machine learning model (Random Forest) to predict cardiovascular disease risk. **Inputs are based on medical standards.**
     Created by: Imam-Fulani Muhammadjamiu
     Powered by: Streamlit & scikit-learn
        """)

        #if (prediction[0] == 0):
            #return "The person is at a High risk of developing cardiovascular disease"
        #else:
        
            #return "The person is at a low risk of developing cardiovascular disease"
        
    
    
    
    
def main():
    
    
    # title for the web page
    st.title(" 🫀 Cardiovascular Disease Risk Predictor")
    st.markdown("Please fill in the fields below to get your cardiovascular risk level.")
    
    
    
    with st.form("cvd_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.text_input("Age (in years)")
            with st.expander("ℹ️ Why is Age important?"):
                st.markdown("Age increases cardiovascular risk because arteries stiffen over time and heart function may weaken.")
            height = st.text_input("Height (cm)")
            weight = st.text_input("Weight (kg)")
            with st.expander("ℹ️ Why are Height and Weight important?"):
                st.markdown("Used to calculate Body Mass Index (BMI). A high BMI increases risk of heart disease and stroke.")
            Systolic_blood_pressure = st.text_input("Systolic blood pressure (mmHg)")
            with st.expander("ℹ️ Why is Systolic Blood Pressure important?"):
                st.markdown("Systolic pressure measures the force when your heart beats. Elevated systolic blood pressure is linked to stroke and heart failure.")
            gender = st.selectbox("Gender", ["-- Select --", "Male", "Female"])
            with st.expander("ℹ️ Why does Gender matter?"):
                st.markdown("Men are more prone to early heart disease, while women’s risk increases post-menopause.")

            
             

        with col2:
            Diastolic_blood_pressure = st.text_input("Diastolic blood pressure (mmHg)")
            with st.expander("ℹ️ Why is Diastolic Blood Pressure important?"):
                st.markdown("Diastolic pressure measures pressure between heartbeats. High DBP puts continuous strain on the heart.")
            cholesterol = st.selectbox("Cholesterol Level", ["-- Select --", "Normal", "Above Normal", "Well Above Normal"])
            with st.expander("ℹ️ Why does Cholesterol Level matter?"):
                st.markdown("High cholesterol can lead to plaque buildup, which narrows arteries and raises heart attack risk.")
            Glucose_Level = st.selectbox("Glucose Level", ["-- Select --", "Normal", "Above Normal", "Well Above Normal"])
            with st.expander("ℹ️ Why does Glucose Level matter?"):
                st.markdown("High glucose can indicate diabetes or insulin resistance, both of which damage blood vessels.")
            smoke = st.selectbox("Do you smoke?", ["-- Select --", "Yes", "No"])
            with st.expander("ℹ️ Why is Smoking status important?"):
                st.markdown("Smoking raises blood pressure, reduces oxygen in blood, and accelerates plaque buildup.")

            Alcohol_intake = st.selectbox("Do you take alcohol?", ["-- Select --", "Yes", "No"])
            with st.expander("ℹ️ Why is Alcohol Intake considered?"):
                st.markdown("Excessive drinking is linked to high blood pressure, obesity, and abnormal heart rhythms.")
            Physical_activity = st.selectbox("Are you physically active?", ["-- Select --", "Yes", "No"])
            with st.expander("ℹ️ Why is Physical Activity relevant?"):
                st.markdown("Regular exercise helps control blood pressure, cholesterol, and maintains a healthy heart.")


        submitted = st.form_submit_button("🔍 Check My CVD Risk")

    if submitted:
        if "" in [age, height, weight, Systolic_blood_pressure, Diastolic_blood_pressure] or "-- Select --" in [gender, cholesterol, Glucose_Level, smoke, Alcohol_intake, Physical_activity]:
            st.warning("⚠️ Please fill in all fields before submitting.")
        else:
            # Convert inputs
            gender = 1 if gender == "Male" else 0
            cholesterol = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[cholesterol]
            Glucose_Level = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[Glucose_Level]
            smoke = 1 if smoke == "Yes" else 0
            Alcohol_intake = 1 if Alcohol_intake == "Yes" else 0
            Physical_activity = 1 if Physical_activity == "Yes" else 0

            input_data = [
                int(age),gender, float(height), float(weight), int(Systolic_blood_pressure), int( Diastolic_blood_pressure),
                cholesterol, Glucose_Level, smoke, Alcohol_intake, Physical_activity 
            ]

            prediction, prob = cvd_prediction(input_data)
            
       
           
            if cvd_prediction == 0:
                
                st.error("### 🟢 Low Risk")
                st.markdown("**The patient is at low risk of cardiovascular disease.** Please consult a healthcare professional.")
            else:
                st.success("### 🔴 High Risk")
                st.markdown("**The patient appears to be at High risk of cardiovascular disease.** Maintain healthy habits.")

            st.markdown(f"**Model Confidence:**")
            st.markdown(f"- Low Risk: `{prob[1]*100:.2f}%`")
            st.markdown(f"- High Risk: `{prob[0]*100:.2f}%`")
    
    
    
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    