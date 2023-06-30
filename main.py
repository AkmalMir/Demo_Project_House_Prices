# python -m streamlit run main.py

import streamlit as st
import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
#from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from lime.lime_tabular import LimeTabularExplainer
import streamlit.components.v1 as components
#import IPython
#import pipreqs

#if __name__ == "__main__":
#    pipreqs.get_pip_requirements()
#

MODEL = lgb.LGBMRegressor() # #


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()
model_prediction = st.container()
explanation = st.container()

with header:
    st.markdown("![image](https://www.rent.com.au/blog/wp-content/uploads/2017/01/melbourne-australia-night-city-view-skyscrapers-lights-river-bridge-1080P-wallpaper-1068x668.jpg)")
    st.title("House Price Prediction - Melbourne, Australia Project!")
    st.text("In this project i look into Houses Prices in Melbourne, Australia")
# Project Title: Melbourn House Price Prediction
# Explanation
st.cache_data()
with dataset:
    st.header("House Prices in Melbourne")
    st.text("dataset in Kaggle.com")
    df = pd.read_csv('melb_data.csv')
    st.write(df.head())
with features:
    st.subheader("Distribution of Number Rooms")
    price_dist = pd.DataFrame(df['Rooms'].value_counts())
    st.bar_chart(price_dist)
    numeric_cols = df.select_dtypes('number').fillna(0)
    y = numeric_cols.pop("Price")
    cols_select = ['Rooms',
                   'Bathroom',
                   'Distance',
                   'Car',
                   'Landsize'
                   ]
    X = numeric_cols[cols_select].astype(int)

st.cache_resource()
with model_training:
    model = MODEL 
    model.fit(X,y)

st.sidebar.header("Select Params")
with model_prediction:
#with predicting():
    def newfeatures():
        ROOMS = st.sidebar.slider('Rooms', int(X['Rooms'].min()), int(X['Rooms'].max())) #X['Rooms'].mean(), format='%d'
        BATHROOMS = st.sidebar.slider('Bathroom', int(X['Bathroom'].min()), int(X['Bathroom'].max())) #X['Bathroom'].mean(), format='%d'
        DISTANCE = st.sidebar.slider('Distance from Central Business District', int(X['Distance'].min()), int(X['Distance'].max()))
        CAR = st.sidebar.slider('Number of carspots', int(X['Car'].min()), int(X['Car'].max()))
        LANDSIZE = st.sidebar.slider('Landsize', int(X['Landsize'].min()), int(X['Landsize'].max()))
        data = {"ROOMS": int(ROOMS), 
                "BATHROOM": int(BATHROOMS),
                "Distance from CBD": int(DISTANCE),
                "Number of carspots": int(CAR),
                "LANDSIZE": int(LANDSIZE),
                }
        features = pd.DataFrame(data,index=[0]) #columns=['ROOMS', 'BATHROOMS']
        return features

    df = newfeatures()

    st.header("<---Change Params with sliders  for Prediction")
    st.write(df)
    print(df)
    st.write("---")



    #with model_training:
    #st.header("Model")
    #st.text("Choose Data to Predict")

    prediction = model.predict(df).round(0)

    st.header("Predicted House Price")
    st.write(prediction)
    st.write("---")

    #with explanation:
    # Split the dataset into a training set and a test set
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=99)
    #model = MODEL #RandomForestRegressor(random_state=42)
    #model.fit(X_train,y_train)
    #preds = model.predict(X_test)

    #with explanation:
    #st.header("Model Metrics")
    #st.write("Mean absolute error of the model is: ")
    #st.write(int(mean_absolute_error(y_test,preds)))
    #st.write("Root_mean_squared_error of the model is: ")
    #st.write(int(mean_squared_error(y_test,preds,squared=False)))
    #st.write("r2_score of the model is: ")
    #st.write(r2_score(y_test,preds).round(2)) 

    #df_metric = pd.DataFrame({
    #    "Metrics": ["MAE", "RMSE", "R2 Score"],
    #    "Value": [int(mean_absolute_error(y_test, preds)), int(mean_squared_error(y_test, preds, squared=False)), r2_score(y_test, preds).round(2)]
    #})
    #st.write(df_metric)

#st.write(df_metric)
with explanation:
    #plt.title("Feature importance based on SHAP Values")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    #
    st.header("The Graph Explains Which Feature is Most Important")
    st.set_option('deprecation.showPyplotGlobalUse', False)

    shap.summary_plot(shap_values,X)
    st.pyplot(bbox_inches='tight')
    st.write("---")

    # visualize the first prediction's explanation
    # Define the feature names for the dataset

    # LIMEEEEEEEEEEE

    st.header("Prediction Explanation")
    feature_names = X.columns

    # Define the lime explainer
    explainer = LimeTabularExplainer(X.values, feature_names=feature_names, mode='regression')

    # Choose an instance for explanation
    instance_idx = 0
    instance = df.iloc[instance_idx] #X

    # Generate an explanation for the instance
    explanation = explainer.explain_instance(instance.values, 
                                            model.predict, 
                                            num_features=len(feature_names))

    # Print the local prediction and explanation
    print("Local Prediction:", model.predict(instance.values.reshape(1, -1)))
    #st.write(explanation.show_in_notebook(show_table=True),unsafe_allow_html=True)

    html = explanation.as_html()
    components.html(html, height=800) #, 
    #######################################################
