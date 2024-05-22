import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import streamlit as st

# Load data
medical_df = pd.read_csv('insurance.csv')

# Encode categorical variables
medical_df.replace({'sex': {'male': 0, 'female': 1},
                    'smoker': {'yes': 0, 'no': 1},
                    'region': {'southeast': 0, 'southwest': 1, 'northwest': 2, 'northeast': 3}}, inplace=True)

X = medical_df.drop('charges', axis=1)
y = medical_df['charges']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

# Model training
lg = LinearRegression()
lg.fit(X_train, y_train)

# Streamlit UI
st.set_page_config(
    page_title="ML based Medical Insurance Prediction",
    page_icon=":hospital:",
    layout="wide"
)

# Logo and title
st.sidebar.image("cit.png")
st.sidebar.title("A ML based Decision Support system for the health Insurance premium")
st.sidebar.header("Project Supervisor")
st.sidebar.write("Dr. Pankaj Pratap Singh, Assistant Professor")
st.sidebar.write("Dept. of Computer Science & Engineering")
st.sidebar.header("Project Members")
st.sidebar.write("Amarjeet Kumar Singh")
st.sidebar.write("Bhagyashree Nath")
st.sidebar.write("Nerswn Baglary")

# Main content
st.title("Welcome to MediPredict")
st.markdown("<style>h1{margin-top: 0;}</style>", unsafe_allow_html=True)

# Create a nice form for user input
with st.form("user_input_form"):
    st.subheader("Enter Person Details")
    age = st.number_input("Age", min_value=0, step=1, format="%d", placeholder="Enter your age")
    sex = st.radio("Sex", options=['Male', 'Female'], help="Select your gender")
    bmi = st.number_input("BMI", min_value=0.0, step=0.1, format="%f", placeholder="Enter your BMI")
    children = st.number_input("Number of Dependents", min_value=0, step=1, format="%d", placeholder="Enter number of children")
    smoker = st.radio("Smoker", options=['Yes', 'No'], help="Select whether you are a smoker or not")
    region = st.selectbox("Region", options=['Southeast', 'Southwest', 'Northwest', 'Northeast'], help="Select your region")
    
    # Additional input fields for display purposes only
    weight = st.text_input("Weight", placeholder="Enter your Weight")
    hereditary_diseases = st.selectbox("Hereditary Disease", options=['No Disease', 'Epilepsy', 'Eye Disease', 'Alzheimer', 'Arthritis', 'Heart Disease', 'Cancer', 'Obesity' ], help="Select If any Hereditary Disease")
    Blood_Pressure  = st.text_input("Blood Pressure ", placeholder="Enter your BP")
    Diabetes = st.radio("Diabetes", options=['Yes', 'No'], help="Are You Diabetic")
    Regular_Excercise = st.radio("Regular Excercise", options=['Yes', 'No'], help="Are You Doing Regular_Excercise")
    martial_status = st.selectbox("Martial Status", options=['Married', 'Unmarried' ], help="Select Your Martial Status")
    no_of_siblings  = st.text_input("Siblings", placeholder="Enter No of Siblings")
    email = st.text_input("Email Address", placeholder="Enter your email")
    phone = st.text_input("Phone Number", placeholder="Enter your phone number")

    st.markdown("<style>div[data-baseweb='button']{background-color: #4CAF50; color: white; border-radius: 5px;}</style>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Predict Insurance Charges")

# Encode categorical variables
sex = 1 if sex == 'Female' else 0
smoker = 1 if smoker == 'No' else 0
region = ['Southeast', 'Southwest', 'Northwest', 'Northeast'].index(region)

# Make prediction if form is submitted
if submitted:
    np_df = np.array([age, sex, bmi, children, smoker, region]).reshape(1, -1)
    prediction = lg.predict(np_df)
    st.write("Estimated Medical Insurance Charges:", prediction[0])