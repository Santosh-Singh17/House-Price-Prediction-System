import streamlit as st
import pandas as pd
import joblib

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ==========================
# Load Model
# ==========================
model = joblib.load("house_price_model.pkl")

# ==========================
# Custom CSS
# ==========================
st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

.title{
    text-align:center;
    color:#003366;
    font-size:42px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:20px;
}

.stButton>button{
    background-color:#0066cc;
    color:white;
    width:100%;
    height:50px;
    border-radius:10px;
    font-size:20px;
    font-weight:bold;
}

.stButton>button:hover{
    background-color:#004c99;
}

.result{
    background:#d4edda;
    padding:20px;
    border-radius:10px;
    text-align:center;
    font-size:28px;
    color:green;
    font-weight:bold;
}

.footer{
    text-align:center;
    color:gray;
    padding-top:30px;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# Sidebar
# ==========================

st.sidebar.title("🏠 House Price Prediction")

st.sidebar.info("""
This application predicts house prices using a trained Machine Learning model.

Model Used:
- Random Forest Regressor

Dataset:
- Housing.csv
""")

st.sidebar.success("Developed using Streamlit")

# ==========================
# Title
# ==========================

st.markdown("<h1 class='title'>🏠 House Price Prediction System</h1>", unsafe_allow_html=True)

st.markdown("<p class='subtitle'>Predict House Prices using Machine Learning</p>", unsafe_allow_html=True)

st.write("")

# ==========================
# Input Form
# ==========================

col1,col2=st.columns(2)

with col1:

    area=st.number_input(
        "Area (sq ft)",
        min_value=500,
        max_value=20000,
        value=5000
    )

    bedrooms=st.slider(
        "Bedrooms",
        1,
        10,
        3
    )

    bathrooms=st.slider(
        "Bathrooms",
        1,
        10,
        2
    )

    stories=st.slider(
        "Stories",
        1,
        5,
        2
    )

    parking=st.slider(
        "Parking",
        0,
        5,
        1
    )

with col2:

    mainroad=st.selectbox(
        "Main Road",
        ["Yes","No"]
    )

    guestroom=st.selectbox(
        "Guest Room",
        ["Yes","No"]
    )

    basement=st.selectbox(
        "Basement",
        ["Yes","No"]
    )

    hotwaterheating=st.selectbox(
        "Hot Water Heating",
        ["Yes","No"]
    )

    airconditioning=st.selectbox(
        "Air Conditioning",
        ["Yes","No"]
    )

    prefarea=st.selectbox(
        "Preferred Area",
        ["Yes","No"]
    )

    furnishingstatus=st.selectbox(
        "Furnishing Status",
        [
            "Furnished",
            "Semi-Furnished",
            "Unfurnished"
        ]
    )

# ==========================
# Encoding Inputs
# ==========================

mainroad = 1 if mainroad=="Yes" else 0
guestroom = 1 if guestroom=="Yes" else 0
basement = 1 if basement=="Yes" else 0
hotwaterheating = 1 if hotwaterheating=="Yes" else 0
airconditioning = 1 if airconditioning=="Yes" else 0
prefarea = 1 if prefarea=="Yes" else 0

if furnishingstatus=="Furnished":
    furnishingstatus=0
elif furnishingstatus=="Semi-Furnished":
    furnishingstatus=1
else:
    furnishingstatus=2
    
    
    
    
# ==========================
# Prediction Button
# ==========================

st.write("")
predict = st.button("🏠 Predict House Price")

# ==========================
# Prediction Logic
# ==========================

if predict:

    input_data = [[
        area,
        bedrooms,
        bathrooms,
        stories,
        mainroad,
        guestroom,
        basement,
        hotwaterheating,
        airconditioning,
        parking,
        prefarea,
        furnishingstatus
    ]]

    prediction = model.predict(input_data)

    price = prediction[0]

    st.markdown("---")

    st.markdown(
        f"""
        <div class="result">
        💰 Predicted House Price <br><br>
        ₹ {price:,.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    st.subheader("📋 House Details")

    result = pd.DataFrame({
        "Feature":[
            "Area",
            "Bedrooms",
            "Bathrooms",
            "Stories",
            "Parking",
            "Main Road",
            "Guest Room",
            "Basement",
            "Hot Water Heating",
            "Air Conditioning",
            "Preferred Area",
            "Furnishing Status"
        ],

        "Value":[
            area,
            bedrooms,
            bathrooms,
            stories,
            parking,
            "Yes" if mainroad else "No",
            "Yes" if guestroom else "No",
            "Yes" if basement else "No",
            "Yes" if hotwaterheating else "No",
            "Yes" if airconditioning else "No",
            "Yes" if prefarea else "No",
            furnishingstatus
        ]
    })

    st.dataframe(result, use_container_width=True)

    st.success("Prediction Completed Successfully!")

    st.balloons()
    
    
    
# ==========================
# Feature Summary
# ==========================

st.markdown("---")

st.subheader("📊 Selected House Features")

feature_df = pd.DataFrame({
    "Feature": [
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Stories",
        "Parking"
    ],
    "Value": [
        area,
        bedrooms,
        bathrooms,
        stories,
        parking
    ]
})

st.bar_chart(feature_df.set_index("Feature"))

# ==========================
# About Project
# ==========================

st.markdown("---")

st.subheader("ℹ About This Project")

st.info("""
### House Price Prediction System

This web application predicts house prices using Machine Learning.

### Technologies Used

✅ Python

✅ Pandas

✅ Scikit-Learn

✅ Streamlit

### Machine Learning Algorithm

🌳 Random Forest Regressor

### Features

✔ User Friendly Interface

✔ Fast Prediction

✔ Machine Learning Based

✔ Responsive Design

✔ Real-Time Prediction
""")

# ==========================
# Dataset Information
# ==========================

with st.expander("📁 Dataset Information"):

    st.write("Dataset Name : Housing.csv")

    st.write("Target Variable : Price")

    st.write("Number of Features : 12")

    st.write("""
Input Features

• Area

• Bedrooms

• Bathrooms

• Stories

• Main Road

• Guest Room

• Basement

• Hot Water Heating

• Air Conditioning

• Parking

• Preferred Area

• Furnishing Status
""")

# ==========================
# Footer
# ==========================

st.markdown("---")

st.markdown("""
<div class="footer">

<h3>🏠 House Price Prediction System</h3>

<p>Developed using Streamlit & Machine Learning</p>

<p>© 2026 All Rights Reserved</p>

</div>
""", unsafe_allow_html=True)