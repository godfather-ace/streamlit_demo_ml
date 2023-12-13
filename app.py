import pickle
import streamlit as st

#loading the trained models using pickle
log_reg = pickle.load(open('streamlit_deploy/lr_model.pkl','rb')) 
decision_tree = pickle.load(open('streamlit_deploy/dt_model.pkl','rb')) 

st.title('IRIS ML Classification Web App')

ml_model = ['Logistic Regression','DecisionTree Classifier']
option = st.sidebar.selectbox('Select the ML model which you want to use', ml_model)

sepal_length = st.slider('Select Sepal Length', 0.0, 10.0, step = 1.0)
sepal_width = st.slider('Select Sepal Width',0.0, 10.0, step = 1.0)
petal_length = st.slider('Select Petal Length',0.0, 10.0, step = 1.0)
petal_width = st.slider('Select Petal Width',0.0, 10.0, step = 1.0)

test  = [[sepal_length, sepal_width, petal_length, petal_width]]

if st.button('Predict'):
    if option=="Logistic Regression":
        st.success(log_reg.predict(test)[0])
    else:
        st.success(decision_tree.predict(test)[0])
        
        
st.number_input("Enter a number", 0, 100, step = 5)