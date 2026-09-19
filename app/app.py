import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# Load trained model
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "gradient_boosting_model.joblib"
)

model = joblib.load(MODEL_PATH)


st.set_page_config(
    page_title="Sydney Housing Price Predictor",
    page_icon="🏠"
)

st.title("Sydney Housing Price Predictor")

st.write(
    "Enter the property details below to estimate the sale price "
    "using the trained Gradient Boosting model."
)

st.info(
    "This model was trained using house sales from St Clair, "
    "Baulkham Hills and St Ives between 2016 and 2021."
)


with st.form("prediction_form"):

    suburb = st.selectbox(
        "Suburb",
        ["St Clair", "Baulkham Hills", "St Ives"]
    )

    num_bed = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=10,
        value=4
    )

    num_bath = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=10,
        value=2
    )

    num_parking = st.number_input(
        "Parking spaces",
        min_value=0,
        max_value=10,
        value=2
    )

    property_size = st.number_input(
        "Property size (m²)",
        min_value=100,
        max_value=10000,
        value=800
    )

    sale_year = st.selectbox(
        "Sale year",
        [2016, 2017, 2018, 2019, 2020, 2021],
        index=5
    )

    st.subheader("Market information")

    cash_rate = st.number_input(
        "Cash rate (%)",
        min_value=0.10,
        max_value=1.75,
        value=0.10,
        step=0.01
    )

    inflation_index = st.number_input(
        "Property inflation index",
        min_value=159.0,
        max_value=220.1,
        value=172.6,
        step=0.1
    )

    submitted = st.form_submit_button(
        "Predict Sale Price"
    )


if submitted:

    size_per_bedroom = property_size / num_bed

    input_data = pd.DataFrame({
        "suburb": [suburb],
        "num_bed": [num_bed],
        "num_bath": [num_bath],
        "num_parking": [num_parking],
        "property_size": [property_size],
        "cash_rate": [cash_rate],
        "property_inflation_index": [inflation_index],
        "sale_year": [sale_year],
        "size_per_bedroom": [size_per_bedroom]
    })

    prediction = model.predict(input_data)[0]

    st.success("Prediction completed")

    st.metric(
        "Estimated Sale Price",
        f"${prediction:,.0f}"
    )

    st.caption(
        "This prediction is an estimate from a small historical dataset "
        "and should not be treated as a professional property valuation."
    )