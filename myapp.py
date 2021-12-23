import streamlit as st
import pandas as pd
from pycaret.classification import *
import shap
import streamlit.components.v1 as components

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

st.sidebar.write('# Heart Disease')
st.sidebar.write("""This database 
contains 76 attributes, but all published experiments 
refer to using a subset of 14 of them. In particular,
the Cleveland database is the only one that has been 
used by ML researchers to this date. The "goal" field 
refers to the presence of heart disease in the patient. 
It is integer valued from 0 (no presence) to 4.""")
st.sidebar.title('What attributes will we use?')
data={'Abbreviation':[
	'age',
	'sex',
	'cp',
	'trestbps',
	'chol',
	'fbs',
	'restecg',
	'thalach',
	'exang',
	'oldpeak',
	'slope',
	'ca',
	'thal',
	'target'],
	'Meaning':[
		'Age in years',
		'Sex: (1-male, 0-famale)',
		'Chest Pain Type(4 values)',
		'Resting Blood Preasure',
		'Serum Cholesteral in mg/dl',
		'Fasting Blood Sugar > 120 mg/dl (1-true,0-false)',
		'resting electrocardiographic results (values 0,1,2)',
		'maximum heart rate achieved',
		'exercise induced angina (1-yes, 0-no)',
		'oldpeak = ST depression induced by exercise relative to rest',
		'the slope of the peak exercise ST segment',
		'number of major vessels (0-3) colored by flourosopy',
		'3 = normal; 6 = fixed defect; 7 = reversable defect',
		'1 or 0']}
st.sidebar.table(pd.DataFrame(data))

st.write(""" # Heart Disease App 
*This app predicts whether you have a heart disease or not*""")
df = pd.read_csv('heart.csv')

st.title('Wanna try?')
age = st.slider('How old are you?', 0, 130, 18)
sex = st.selectbox('Sex',('1', '0'))
cp = st.selectbox('Chest Pain', (0,1,2,3)) 
trestbps = st.slider('Resting Blood Preasure',100,200,120)
chol = st.slider('Serum Cholesteral in mg/dl',100,400,200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl',('1', '0'))
restecg = st.selectbox('Resting electrocardiographic results (values 0,1,2)',(0, 1, 2))
thalach = st.slider('Maximum heart rate achieved',50,250,160)
exang = st.selectbox('Exercise induced angina',('1','0'))
oldpeak = st.slider('Oldpeak = ST depression induced by exercise relative to rest',0.0,5.0,0.5)
slope = st.slider('The slope of the peak exercise ST segment',0,2,0)
ca = st.slider('Number of major vessels (0-3) colored by flourosopy',0,3,0)
thal = st.selectbox('3 = normal (1); 6 = fixed defect (2); 7 = reversable defect (3)',(1,2,3))

#ML Part
if st.button('Press to Predict'):
     model = load_model('model')
     datos={'age':[str(age)],'sex':[sex],'cp':[str(cp)],'trestbps':[str(trestbps)],
     	'chol':[str(chol)],'fbs':[fbs],'restecg':[str(restecg)],'thalach':[str(thalach)],
	'exang':[exang],'oldpeak':[str(oldpeak)],'slope':[str(slope)],'ca':[str(ca)],'thal':[str(thal)]}
     datosFormat=pd.DataFrame(datos)
     prediction = model.predict(datosFormat)
     print(prediction)
     st.title('Results')
     st.subheader('Entered Data:')
     st.table(datosFormat)
     if prediction:
        st.subheader('Prediction Result: Positive')
     else:
        st.subheader('Prediction Result: Negative')     
     st.title('Explainable AI (XAI) using SHAP')
     df = pd.read_csv('heart.csv')
     train_pipe = model[:-1].transform(datosFormat)
     explainer = shap.TreeExplainer(model.named_steps["trained_model"])
     shap_values = explainer.shap_values(train_pipe)
     st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0], train_pipe),400)