
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
# Ensure 'best_model.pkl' is in the same directory as this app or accessible via path
try:
    model = joblib.load('best_model.pkl')
except FileNotFoundError:
    st.error("Error: 'best_model.pkl' not found. Please ensure the model file is in the correct path.")
    st.stop()

# Recreate LabelEncoders from the original dataset for consistency
# Ensure 'Salary_Data.csv' is in the same directory as this app or accessible via path
try:
    df_raw = pd.read_csv('/content/Salary_Data.csv')

    # Fill nulls for numerical columns with mean (as done during training)
    df_raw['Age'] = df_raw['Age'].fillna(df_raw['Age'].mean())
    df_raw['Years of Experience'] = df_raw['Years of Experience'].fillna(df_raw['Years of Experience'].mean())
    # Salary is the target, not used for encoding, but included for completeness if needed

    # Fit LabelEncoders for categorical columns (as done during training)
    gender_le = LabelEncoder()
    df_raw['Gender'] = df_raw['Gender'].fillna(df_raw['Gender'].mode()[0])
    gender_le.fit(df_raw['Gender'])

    education_le = LabelEncoder()
    df_raw['Education Level'] = df_raw['Education Level'].fillna(df_raw['Education Level'].mode()[0])
    education_le.fit(df_raw['Education Level'])

    job_title_le = LabelEncoder()
    df_raw['Job Title'] = df_raw['Job Title'].fillna(df_raw['Job Title'].mode()[0])
    job_title_le.fit(df_raw['Job Title'])

except FileNotFoundError:
    st.error("Error: 'Salary_Data.csv' not found. Please ensure the data file is in the correct path for encoding.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during LabelEncoder fitting: {e}")
    st.stop()

# Streamlit App Title
st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# Input fields for numerical features
age = st.number_input('Age', min_value=18, max_value=65, value=30)
years_of_experience = st.number_input('Years of Experience', min_value=0.0, max_value=40.0, value=5.0, step=0.5)

# Input fields for categorical features using selectbox with known classes
gender_options = list(gender_le.classes_)
gender_selected = st.selectbox('Gender', options=gender_options)

education_options = list(education_le.classes_)
education_selected = st.selectbox('Education Level', options=education_options)

# For Job Title, use a selectbox with all known job titles
job_title_options = list(job_title_le.classes_)
job_title_selected = st.selectbox('Job Title', options=job_title_options, help="Select a job title from the list.")

if st.button('Predict Salary'):
    try:
        # Encode categorical features using the fitted LabelEncoders
        gender_encoded = gender_le.transform([gender_selected])[0]
        education_encoded = education_le.transform([education_selected])[0]
        job_title_encoded = job_title_le.transform([job_title_selected])[0]

        # Create a DataFrame for the input, ensuring column order and names match training data
        input_data = pd.DataFrame([[
            age,
            gender_encoded,
            education_encoded,
            job_title_encoded,
            years_of_experience
        ]],
            columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']
        )

        # Make prediction
        prediction = model.predict(input_data)[0]

        st.success(f'Predicted Salary: ${prediction:,.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}. Please check your inputs.")
