# WebApp starts here
import numpy as np
import pandas as pd
from pywebio.input import *
from pywebio.output import *
from pywebio.session import *
from pywebio.platform import *
from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH, start_server
from flask import Flask, send_from_directory
import argparse

#load the model
import pickle
model = pickle.load(open('logmod.pkl', 'rb'))
app = Flask(__name__)

#To get the dummy columns
dummy = pd.read_csv('dummy')
dummy.drop('Unnamed: 0', axis = 1, inplace = True)
dummyCols = dummy.columns

input_data = []
def heart():
    put_text('Check your patients heart health by filling out the form below.').style('font-size: 20px')
    info = input_group("Heart disease form",
        [radio('Input your sex',options=['Male','Female'],name='Sex', required=True),
         radio('Input chest paint type:', options=['Typical angina',\
                                                     'Atypical angina',\
                                                    'Non-anginal pain',\
                                                    'Asymptomatic'],\
                 name='ChestPain',required=True),
         input("Input resting blood pressure (in mm Hg): ", name='RestingBloodPressure', type=FLOAT, required=True),
         input("Input serum cholestoral in mg/dl: ", name='Cholesterol',type=FLOAT,required=True),
         radio("Fasting blood sugar > 120 mg/dl",options=['Yes','No'],name='FastingBloodSugar',required=True),
        radio('Resting electrocardiographic results',\
                 options=['Normal',\
                          'Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)',\
                          'Showing probable or definite left ventricular hypertrophy by Estes\' criteria'],\
                name = 'Resting_electrocardiographic',required=True),
        input("Input maximum heart rate achieved: ",name='MaxHeartRate',type=FLOAT,required=True),
        radio("Does the patient have exercise induced angina: ", options=['Yes','No'],name ='ExerciseInducedAngina',required=True),
        input("Input ST depression induced by exercise relative to rest (Value should vary between 0 and 7): "\
                    ,type=FLOAT,\
             name='STDepression',required=True),
        radio("Input the slope of the peak exercise ST segment: ",options=['Upsloping',\
                                                                              'Flat',\
                                                                              'Downsloping'],name='Slope',required=True),
        radio("Number of major vessels colored by fluorosopy", options=['0','1','2','3'],
                name='MajorVessels',required=True),
        radio("Input thalessemia level", options=['Normal',\
                                                     'Fixed defect',\
                                                     'Reversable defect'],
                name='Thalessemia',required=True)])

    #Create dictionaries
    sex_dict = {
        'Male': 1,
        'Female': 0
    }
    
    cp_dict = {
        'Typical angina' : 0,
         'Atypical angina' : 1,
         'Non-anginal pain': 2,
         'Asymptomatic': 3
    }
    
    fbs_dict = {
        'Yes' : 1,
        'No' : 0
    }
    
    restecg_dict = {
        'Normal' : 0,
        'Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)' : 1, 
        'Showing probable or definite left ventricular hypertrophy by Estes\' criteria' : 2
    }
    
    exang_dict = { 
        'Yes' : 1,
        'No' : 0
    }
    
    slope_dict ={
        'Upsloping' : 0,
        'Flat' : 1,
        'Downsloping' : 2
    }
    
    thal_dict = {
        'Normal' : 0,
        'Fixed defect' : 1,
        'Reversable defect' : 2
    }
    
    input_data = [sex_dict[info['Sex']], 
                   cp_dict[info['ChestPain']],
                   fbs_dict[info['FastingBloodSugar']],
                   restecg_dict[info['Resting_electrocardiographic']],
                   exang_dict[info['ExerciseInducedAngina']],
                   slope_dict[info['Slope']],
                   int(info['MajorVessels']),
                   thal_dict[info['Thalessemia']],
                   info['RestingBloodPressure'],
                   info['Cholesterol'],
                   info['MaxHeartRate'],
                   info['STDepression']]
    
    catColumns = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
    to_viz = ['trestbps', 'chol', 'thalach', 'oldpeak']
    for i in to_viz:
        catColumns.append(i)

    input_df = pd.DataFrame(columns = catColumns)
    input_df.loc[0] = input_data
    catColumns = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
    to_viz = ['trestbps', 'chol', 'thalach', 'oldpeak']
    new_input_to_produce = []
    new_input_to_produce2 = []
    
    #Creating dummy col
    for col in catColumns:
        new_input_to_produce.append(pd.get_dummies(input_df[col], drop_first=False, prefix=col, dtype=int))  
    new_data1 = pd.concat(new_input_to_produce, axis = 1)
    
    #Continuous columns to add
    for v in to_viz: 
        new_input_to_produce2.append(input_df[v])
    new_data2 = pd.concat(new_input_to_produce2, axis = 1)

    final_data = pd.concat([new_data1, new_data2], axis = 1)
    final_data = pd.concat([dummy, final_data])
    final_data = final_data.fillna(0)

    if model.predict(final_data.tail(1).to_numpy()) == [1]: 
        popup("You have a higher risk of heart disease (accuracy: 91%)")
    else: 
        popup("You do not have a heart disease (accuracy: 91%)")

#Deploy app in Heroku        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--port", type=int, default=8080)
    args = parser.parse_args()
    start_server(heart, port = args.port)
        
#Run app locally
app.add_url_rule('/WebApp','webio_view',webio_view(heart),
               methods=['GET','POST','OPTIONS'])
app.run('localhost',port=80)
#http://localhost/WebApp
if __name__ == '__main__':
        heart()

