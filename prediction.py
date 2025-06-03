import streamlit as st
import joblib
import numpy as np

def run():
    # Load the trained Ridge regression model
    ridge_model = joblib.load('ridge_best_model.joblib')

    st.title("House Price Prediction App")

    st.markdown("""
    ### This app predicts the price of a house based on various features  
    ### Please enter the desired feature values, then see the estimated home prices in the area!

    ### Feature Description:

    | Feature   | Description |
    |-----------|-------------|
    | `crim`    | Per capita crime rate by town |
    | `zn`      | Proportion of residential land zoned for lots over 25,000 sq.ft. |
    | `indus`   | Proportion of non-retail business acres per town |
    | `chas`    | Charles River dummy variable (1 if tract bounds river; 0 otherwise) |
    | `nox`     | Nitrogen oxide concentration (parts per 10 million) |
    | `rm`      | Average number of rooms per dwelling |
    | `age`     | Proportion of owner-occupied units built prior to 1940 |
    | `dis`     | Weighted distances to five Boston employment centers |
    | `ptratio` | Pupil-teacher ratio by town |
    | `black`   | 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents |
    | `lstat`   | Percentage of lower status population |
    """)

    st.image('houses.jpg', caption='House Prices')

    st.sidebar.header("Input Features")

    def user_input_features():
        crim = st.sidebar.slider('Crime rate per capita (crim)', 0.0, 100.0, 0.1)
        zn = st.sidebar.slider('Residential land proportion (zn)', 0.0, 100.0, 18.0)
        indus = st.sidebar.slider('Non-retail business land (indus)', 0.0, 30.0, 7.0)
        chas = st.sidebar.selectbox('Borders Charles River? (chas)', [0, 1])
        nox = st.sidebar.slider('Nitric oxides concentration (nox)', 0.0, 1.0, 0.5)
        rm = st.sidebar.slider('Average number of rooms (rm)', 3.0, 10.0, 6.0)
        age = st.sidebar.slider('Proportion of old units (age)', 0.0, 100.0, 50.0)
        dis = st.sidebar.slider('Distance to employment centers (dis)', 1.0, 12.0, 5.0)
        ptratio = st.sidebar.slider('Pupil-teacher ratio (ptratio)', 10.0, 30.0, 18.0)
        black = st.sidebar.slider('Proportion of black population (black)', 0.0, 400.0, 350.0)
        lstat = st.sidebar.slider('% lower status of population (lstat)', 0.0, 40.0, 12.0)

        features = {
            'crim': crim,
            'zn': zn,
            'indus': indus,
            'chas': chas,
            'nox': nox,
            'rm': rm,
            'age': age,
            'dis': dis,
            'ptratio': ptratio,
            'black': black,
            'lstat': lstat
        }

        return np.array(list(features.values())).reshape(1, -1), features

    input_data, features_dict = user_input_features()

    st.subheader("Selected Input Features")
    st.write(features_dict)

    prediction = ridge_model.predict(input_data)
    st.subheader("Predicted House Price")
    st.write(f"${prediction[0]*1000:,.2f} (in USD)")

    st.markdown("""
    ---  
    **Note:**  
    - `chas`: 1 = Yes (borders Charles River), 0 = No  
    - Predicted value is the **median value of owner-occupied homes** in thousands of dollars  
    - This is a simplified version of a housing price prediction app  
    - **Created by Muhammad Wahyu Ghifari**  
    """)
