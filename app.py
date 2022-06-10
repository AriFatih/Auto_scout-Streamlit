import pandas as pd
import streamlit as st
import numpy as np
import pickle
from xgboost import XGBRegressor

html_temp = """
<div style="background-color:tomato;padding:1.5px">
<h1 style="color:white;text-align:center;">Auto-Scout Car Price Predictor </h1>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)



st.write("""


This app predicts the price of second-hand cars!


""")

st.sidebar.header('Please give the features of the car to predict its price!')


# Collects user input features into dataframe

#def user_input_features():

def user_input_features():
    Make_model = st.sidebar.selectbox("Select Your Car's Make&Model", ("Audi A3", "Audi A1", "Opel Insignia","Opel Astra", "Opel Corsa", "Renault Clio", "Renault Duster", "Renault Espace", ))
    Gearing_Type = st.sidebar.selectbox("Select your car's gearing type", ('Manual','Automatic', 'Semi-automatic'))
    hp_kw = st.sidebar.slider('HP', 45, 300, 85)
    km = st.sidebar.slider('Km', 0, 350000, 20500)
    Age = st.sidebar.slider('Age', 0, 6, 0)
    data = {'make_model': Make_model,
        'Gearing_Type': Gearing_Type,
        'hp_kw': hp_kw,
        'km' : km,
        'Age' : Age
        }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

#To get all the features for get_dummies, we concatenate with the original df.
input_raw = pd.read_csv('df_with_five_features.csv')
input_df2 = input_raw.drop(columns=['price'])
df = pd.concat([input_df, input_df2], axis=0)

#
df = pd.get_dummies(df, drop_first =True)

#Getting the first row which ouur input data
df = df[:1]

# Reads in saved  regression model
load_clf = pickle.load(open('best_model_auto_scout.pkl','rb'))

st.table(input_df,)
if st.button("Predict"):
    pred = load_clf.predict(df)
    st.write(pred[0])
    st.balloons()















