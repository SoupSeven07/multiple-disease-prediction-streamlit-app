import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    st.title("Diabetes Prediction")

    # Basic numerical inputs
    Pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)
    SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0.0, step=0.1)

    # Binary inputs with meaningful labels
    BloodPressure_risk = st.selectbox("Blood Pressure Risk", ["Normal", "High"])
    BloodPressure_risk = 1 if BloodPressure_risk == "High" else 0

    Insulin_risk_Normal = st.selectbox("Is Insulin Risk Normal?", ["Yes", "No"])
    Insulin_risk_Normal = 1 if Insulin_risk_Normal == "Yes" else 0

    # Newage (one-hot: mature, senior → young is default if both are 0)
    Newage = st.selectbox("Age Group", ["Young", "Mature", "Senior"])
    Newage_mature = 1 if Newage == "Mature" else 0
    Newage_senior = 1 if Newage == "Senior" else 0

    # NewGlucose (one-hot)
    NewGlucose = st.selectbox("Glucose Level", ["Low", "Normal", "Overweight", "Secret"])
    NewGlucose_Low = 1 if NewGlucose == "Low" else 0
    NewGlucose_Normal = 1 if NewGlucose == "Normal" else 0
    NewGlucose_Overweight = 1 if NewGlucose == "Overweight" else 0
    NewGlucose_Secret = 1 if NewGlucose == "Secret" else 0

    # Genetics_risk (one-hot: low, very high → medium/other is default if both are 0)
    Genetics_risk = st.selectbox("Genetic Risk", ["Normal", "Low", "Very High"])
    Genetics_risk_low = 1 if Genetics_risk == "Low" else 0
    Genetics_risk_very_high = 1 if Genetics_risk == "Very High" else 0

    # BMI_risk (one-hot: obesity_1, obesity_2, overweight, rare)
    BMI_risk = st.selectbox("BMI Risk Category", ["Normal", "Obesity_1", "Obesity_2", "Overweight", "Rare"])
    BMI_risk_Obesity_1 = 1 if BMI_risk == "Obesity_1" else 0
    BMI_risk_Obesity_2 = 1 if BMI_risk == "Obesity_2" else 0
    BMI_risk_Overweight = 1 if BMI_risk == "Overweight" else 0
    BMI_risk_Rare = 1 if BMI_risk == "Rare" else 0

    # Run prediction
    if st.button("Predict"):
        # Match the input format your model expects (16 features)
        sample_data = np.array([[
            Pregnancies,
            SkinThickness,
            BloodPressure_risk,
            Insulin_risk_Normal,
            Newage_mature,
            Newage_senior,
            NewGlucose_Low,
            NewGlucose_Normal,
            NewGlucose_Overweight,
            NewGlucose_Secret,
            Genetics_risk_low,
            Genetics_risk_very_high,
            BMI_risk_Obesity_1,
            BMI_risk_Obesity_2,
            BMI_risk_Overweight,
            BMI_risk_Rare
        ]])
        
        try:
            # Load the .sav model
            with open("saved_models/features_diabetes_model.sav", "rb") as model_file:
                model = pickle.load(model_file)

            prediction = model.predict(sample_data)

            if prediction[0] == 1:
                st.error("The model predicts: Positive for Diabetes")
            else:
                st.success("The model predicts: Negative for Diabetes")

        except Exception as e:
            st.error(f"Model error: {e}")

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)
