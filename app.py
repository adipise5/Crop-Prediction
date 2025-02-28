import joblib
import streamlit as st 
import pandas as pd
import os

# Load Model, Scaler & Polynomial Features
model = joblib.load('model.pkl')
sc = joblib.load('sc.pkl')
pf = joblib.load('pf.pkl')

# Load Dataset
df_final = pd.read_csv('test.csv')
df_main = pd.read_csv('main.csv')

# Custom CSS for Styling
st.markdown(
    """
    <style>
    /* Set Background to White */
    .stApp {
        background-color: white !important;
    }
    
    /* Main Title */
    .title {
        color: #2E3B55; 
        text-align: center;
        font-size: 32px;
        font-weight: bold;
    }

    /* Subtitle */
    .subtitle {
        color: #4A6FA5;
        text-align: center;
        font-size: 20px;
    }

    /* Input Box Styling */
    .input-box {
        background: #F8F9FA;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #E0E0E0;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }

    /* Predict Button */
    .stButton>button {
        background-color: #4A6FA5;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px;
        border: none;
    }

    /* Prediction Result */
    .result-box {
        background: #E3F2FD;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: #1E4A72;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to Update Columns for Categorical Features
def update_columns(df, true_columns):
    df[true_columns] = True
    other_columns = df.columns.difference(true_columns)
    df[other_columns] = False
    return df

# Prediction Function
def prediction(input):
    categorical_col = input[:2]
    input_df = pd.DataFrame({
        'average_rainfall': [input[2]],
        'presticides_tonnes': [input[3]],
        'avg_temp': [input[4]]
    })

    input_df1 = df_final.head(1).iloc[:, 3:]
    true_columns = [f'Country_{categorical_col[0]}', f'Item_{categorical_col[1]}']
    input_df2 = update_columns(input_df1, true_columns)

    final_df = pd.concat([input_df, input_df2], axis=1).values
    test_input = sc.transform(final_df)
    test_input1 = pf.transform(test_input)
    predict = model.predict(test_input1)

    result = (int(((predict[0]/100) * 2.47105) * 100) / 100)
    return f"🌾 **Predicted Crop Yield:** {result} quintal/acre.\n\n📌 1 acre of land is expected to produce **{result} quintals** of crop yield."

# Main Streamlit App
def main():
    st.markdown("<h1 class='title'>🌾 Yield Crop Prediction Model</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='subtitle'>📝 Enter Crop Details Below</h2>", unsafe_allow_html=True)

    st.markdown("---")

    # Create a Big Box for Inputs

    country = st.selectbox("🌍 Select Country:", df_main['area'].unique())
    crop = st.selectbox("🌱 Select Crop:", df_main['item'].unique())

    col1, col2 = st.columns(2)

    with col1:
        average_rainfall = st.number_input("💧 Average Rainfall (mm/year):", min_value=0.0, format="%.2f")
        avg_temp = st.number_input("🌡️ Average Temperature (°C):", min_value=-10.0, max_value=50.0, format="%.2f")

    with col2:
        presticides = st.number_input("🛡️ Pesticides Use (tonnes):", min_value=0.0, format="%.2f")

    st.markdown("</div>", unsafe_allow_html=True)  # End Input Box

    # Predict Button
    if st.button("🚜 Predict Yield", use_container_width=True):
        result = prediction([country, crop, average_rainfall, presticides, avg_temp])

        st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)

# Run App
if __name__ == '__main__':
    main()
