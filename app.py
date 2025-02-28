import joblib
import streamlit as st 
import pandas as pd
import base64
import os

# Load Model, Scaler & Polynomial Features
model = joblib.load('model.pkl')
sc = joblib.load('sc.pkl')
pf = joblib.load('pf.pkl')

# Load Dataset
df_final = pd.read_csv('test.csv')
df_main = pd.read_csv('main.csv')

# Function to Set Background Image
def set_bg(image_path):
    if os.path.exists(image_path):  # Ensure image exists
        with open(image_path, "rb") as image_file:
            base64_str = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url(data:image/png;base64,{base64_str}) no-repeat center center fixed;
                background-size: cover;
                backdrop-filter: blur(5px);
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Background image not found! Using default theme.")

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
    set_bg("background.jpg")  # Change to your image file or use a URL-based background

    st.markdown(
        "<h1 style='text-align: center; color: white; font-size: 32px;'>🌾 Yield Crop Prediction Model</h1>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Create a big box for all inputs
    st.markdown(
        """
        <style>
        .big-box {
            background: rgba(255, 255, 255, 0.2);
            padding: 20px;
            border-radius: 15px;
            border: 2px solid #ffffff;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        }
        </style>
        <div class='big-box'>
        """,
        unsafe_allow_html=True
    )

    # Form Layout (Inside Big Box)
    st.markdown("<h3 style='text-align: center; color: white;'>📝 Enter Crop Details Below</h3>", unsafe_allow_html=True)

    country = st.selectbox("🌍 Select Country:", df_main['area'].unique())
    crop = st.selectbox("🌱 Select Crop:", df_main['item'].unique())

    col1, col2 = st.columns(2)

    with col1:
        average_rainfall = st.number_input("💧 Average Rainfall (mm/year):", min_value=0.0, format="%.2f")
        avg_temp = st.number_input("🌡️ Average Temperature (°C):", min_value=-10.0, max_value=50.0, format="%.2f")

    with col2:
        presticides = st.number_input("🛡️ Pesticides Use (tonnes):", min_value=0.0, format="%.2f")

    # End Big Box
    st.markdown("</div>", unsafe_allow_html=True)

    # Predict Button
    if st.button("🚜 Predict Yield", use_container_width=True):
        result = prediction([country, crop, average_rainfall, presticides, avg_temp])

        st.markdown(
            f"""
            <div style="background-color: #2E8B57; padding: 15px; border-radius: 10px;">
            <h3 style="text-align: center; color: white;">{result}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

# Run App
if __name__ == '__main__':
    main()
