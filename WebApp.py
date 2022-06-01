#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

#Format the dataset
df = pd.read_csv('heart_cleveland_upload.csv')
columns = df.columns[1:-1]

#Create dummies for categorical columns
catColumns = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
new_to_produce = []
for col in catColumns: 
    new_to_produce.append(pd.get_dummies(df[col], drop_first=False, prefix=col, dtype=int))
dataLog = pd.concat(new_to_produce, axis = 1).sort_index()

dataLog['condition'] = df['condition']
columns_to_fill = dataLog.columns
columns_to_fill = columns_to_fill[:-1]
df_to_fill = pd.DataFrame(columns = columns_to_fill)


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
         input("Input resting blood pressure (in mm Hg): ",name='RestingBloodPressure',type=FLOAT,required=True),
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
    
    input_data = [[sex_dict[info['Sex']],                    cp_dict[info['ChestPain']],                    info['RestingBloodPressure'],                    info['Cholesterol'],                    fbs_dict[info['FastingBloodSugar']],                    restecg_dict[info['Resting_electrocardiographic']],                    info['MaxHeartRate'],                    exang_dict[info['ExerciseInducedAngina']],                    info['STDepression'],                   slope_dict[info['Slope']],                   int(info['MajorVessels']),                   thal_dict[info['Thalessemia']]]]
    
    input_df = pd.DataFrame(input_data, columns = columns)
    
    catColumns = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
    new_input_to_produce = []
    for col in columns:
        if col not in catColumns: 
            new_input_to_produce.append(input_df[col])
        else: 
            new_input_to_produce.append(pd.get_dummies(input_df[col], drop_first=False, prefix=col, dtype=int))
    dataLogInput = pd.concat(new_input_to_produce, axis = 1).sort_index()
    
    final_new_input_df = dataLogInput.join(df_to_fill[df_to_fill.columns.difference(dataLogInput.columns)])
    final_new_input_df = final_new_input_df.fillna(0)
    final_new_input_df = final_new_input_df.reindex(columns=df_to_fill.columns)
    
    to_feed = final_new_input_df.iloc[0].to_numpy()
    
    if model.predict(to_feed.reshape(1,-1)) == [1]: 
        popup("You have a higher risk of heart disease (accuracy: 91%)")
    else: 
        popup("You do not have a heart disease (accuracy: 91%)")

#Deploy app in Heroku        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--port", type=int, default=8080)
    args = parser.parse_args()
    start_server(heart, port = args.port)
        
#Run app as localhost
#app.add_url_rule('/WebApp','webio_view',webio_view(heart),
#                methods=['GET','POST','OPTIONS'])
#app.run('localhost',port=80)
#http://localhost/WebApp
#if __name__ == '__main__':
#        heart()

