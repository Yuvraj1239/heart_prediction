import streamlit as st
import pickle
import numpy as np


#model=pickle.load(open('model.pkl','rb'))
model2=pickle.load(open('model2.pkl','rb'))




st.image("heart.jpeg")

def predict():
    pregnencies = st.number_input("pregnencies",key=12)
    glucose = st.number_input("glucose",key=13)
    bloodpressure =st.number_input("bloodpressure",key=14)
    skinthickness = st.number_input("skinthickness",key=15)
    insuline = st.number_input("insuline",key=16)
    BMI = st.number_input("BMI",key=17 )
    diabetespedigreefunction = st.number_input("diabetespedigreefunction",key=18)
    age = st.number_input("age",key=19)
    list1 = [pregnencies,glucose,bloodpressure,skinthickness,insuline,BMI,diabetespedigreefunction,age]
    
    final=[np.array(list1)]
    if st.button("predict_heartdisease"):
        prediction=model2.predict(final)
        if prediction==0:
            st.write('you are heart patient')
            
        else:
            st.write('you are not heart patient')
            

#def predict2():
    #age = st.number_input("age",key=44)
    #sex = st.number_input("sex",key=0)
    #cp = st.number_input("cp",key=1)
    #tresbps = st.number_input("tresbps",key=2)
    #chol = st.number_input("chol",key=3)
    #fbs = st.number_input("fbs",key=4)
    #restech = st.number_input("restech",key=5)
    #thalach = st.number_input("tha,key=0lach",key=6)
    #exang = st.number_input("exang",key=7)
    #oldpeak = st.number_input("oldpeak",key=8)
    #slope = st.number_input("slope",key=9)
    #ca = st.number_input("ca",key=10)
    #thal = st.number_input("thal",key=11)
    #list2 =[age,sex,cp,tresbps,chol,fbs,restech,thalach,exang,oldpeak,slope,ca,thal]
    #final2=[np.array(list2)]
    #if st.button("predict_diabetes"):
        #prediction=model.predict(final2)
        #if prediction==0:
            #st.write('you are diabetes patient')
            #engine.say('you are diabetes patient')
        #else:
            #st.write('you are not diabetes patient')
            #engine.say('you are not diabetes patient')*/


predict()
#predict2()


