import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.set_page_config(
    page_title="Cars24 Price Predictor",
    page_icon="🚗",
    layout="wide"
)


@st.cache_resource
def load_model():
    model   = pickle.load(open('best_model.pkl',    'rb'))
    scaler  = pickle.load(open('scaler.pkl',        'rb'))
    columns = pickle.load(open('model_columns.pkl', 'rb'))
    return model, scaler, columns

@st.cache_data
def load_data():
    return pd.read_csv('cars24_cleaned_happy.csv')

model, scaler, model_columns = load_model()
df = load_data()


with st.sidebar:
    st.image("logo.png",
             width=150)
    st.markdown("---")
    st.header("📊 About This Project")
    st.info("""
    **Data Collection:**
    - 15,673 listings scraped from Cars24
    - 8 major Indian cities
    - Real market data

    **Model Performance:**
    - Algorithm: XGBoost
    - R² Score: **0.9023**
    - MAE: **₹0.55 Lakhs**
    - MAPE: **12.76%**

    **Built by:** Bhargav Praveen Chintapalli
    """)

    st.markdown("---")
    st.header("📈 Dataset Highlights")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Records",  f"{len(df):,}")
        st.metric("Cities",   "8")
    with col2:
        st.metric("Brands",   f"{df['Brand'].nunique()}")
        st.metric("Models",   f"{df['Model'].nunique()}")

    st.markdown("---")
    st.markdown("**Key Finding:**")
    st.markdown("Car Age is the **#1 price driver** — 10x more important than any other feature")


st.title("🚗 Cars24 Price Predictor")
st.markdown("#### Predict the fair market value of any used car in India")
st.markdown("*Trained on 15,673 real Cars24 listings across 8 cities*")
st.markdown("---")


st.subheader("🔍 Enter Car Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🏷️ Car Identity**")

    # Step 1 — Brand selection
    brand = st.selectbox(
        "Brand",
        options=sorted(df['Brand'].unique().tolist()),
        help="Select the car manufacturer"
    )

    # Step 2 — Engine type filtered by Brand
    available_engines = df[
        df['Brand'] == brand
    ]['Engine_Type'].unique().tolist()

    engine_type = st.selectbox(
        "Engine Type",
        options=sorted(available_engines),
        help="Only shows engine types available for selected brand"
    )

    # Step 3 — Model filtered by BOTH Brand + Engine Type
    available_models = df[
        (df['Brand'] == brand) &
        (df['Engine_Type'] == engine_type)
    ]['Model'].unique().tolist()

    if len(available_models) == 0:
        st.error(f"❌ No {engine_type} models found for {brand}")
        car_model = None
    else:
        car_model = st.selectbox(
            "Model",
            options=sorted(available_models),
            help=f"Showing {len(available_models)} models for {brand} {engine_type}"
        )
        st.caption(f"✅ {len(available_models)} models available")

with col2:
    st.markdown("**⚙️ Specifications**")

    # Transmission filtered by Brand + Engine + Model
    if car_model:
        available_trans = df[
            (df['Brand'] == brand) &
            (df['Engine_Type'] == engine_type) &
            (df['Model'] == car_model)
        ]['Transmission'].unique().tolist()
    else:
        available_trans = ['Manual', 'Auto']

    transmission = st.selectbox(
        "Transmission",
        options=available_trans,
        help="Only shows transmissions available for selected model"
    )

    is_top_trim = st.radio(
        "Variant Type",
        options=[0, 1],
        format_func=lambda x: "🔝 Top Trim (ZXI/Asta/Alpha/AMT)"
                               if x == 1 else "⚙️ Base / Mid Trim",
        help="Is this a premium or basic variant?"
    )

    # Show real price range for this exact combination
    if car_model:
        combo_prices = df[
            (df['Brand'] == brand) &
            (df['Model'] == car_model) &
            (df['Engine_Type'] == engine_type)
        ]['Price_In_Lakhs']

        if len(combo_prices) > 0:
            st.markdown("**📊 Real Market Range:**")
            range_col1, range_col2 = st.columns(2)
            with range_col1:
                st.metric("Min Price",
                          f"₹{combo_prices.min():.2f}L")
            with range_col2:
                st.metric("Max Price",
                          f"₹{combo_prices.max():.2f}L")
            st.caption(f"Based on {len(combo_prices)} real listings")

with col3:
    st.markdown("**📍 Location & Condition**")

    location = st.selectbox(
        "City",
        options=sorted(df['Location'].unique().tolist()),
        help="City where car is listed"
    )

    state = st.selectbox(
        "State",
        options=sorted(df['State'].unique().tolist()),
        help="State where car is registered"
    )

    manufacture_year = st.slider(
        "Year of Manufacture",
        min_value=2000,
        max_value=2024,
        value=2019,
        help="Which year was the car manufactured?"
    )
    car_age = 2025 - manufacture_year
    st.caption(f"Car Age: **{car_age} years**")

    driven_kms = st.number_input(
        "Kilometres Driven",
        min_value=500,
        max_value=300000,
        value=50000,
        step=1000,
        help="Total KMs driven"
    )


def predict_price(brand, car_model, engine_type, transmission,
                  location, state, car_age, driven_kms, is_top_trim):

    # Step 1: Start with numeric features
    input_dict = {
        'Driven_Kms':  driven_kms,
        'Car_Age':     car_age,
        'is_top_trim': is_top_trim
    }
    input_df = pd.DataFrame([input_dict])

    # Step 2: Add one-hot encoded columns as 0s first
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Step 3: Set the correct category flags to 1
    # Brand
    brand_col = f'Brand_{brand}'
    if brand_col in model_columns:
        input_df[brand_col] = 1

    # Model
    model_col = f'Model_{car_model}'
    if model_col in model_columns:
        input_df[model_col] = 1

    # Engine Type
    engine_col = f'Engine_Type_{engine_type}'
    if engine_col in model_columns:
        input_df[engine_col] = 1

    # Transmission
    trans_col = f'Transmission_{transmission}'
    if trans_col in model_columns:
        input_df[trans_col] = 1

    # Location
    loc_col = f'Location_{location}'
    if loc_col in model_columns:
        input_df[loc_col] = 1

    # State
    state_col = f'State_{state}'
    if state_col in model_columns:
        input_df[state_col] = 1

    # Step 4: Reorder columns to EXACTLY match training
    input_df = input_df[model_columns]

    # Step 5: Scale
    input_scaled = scaler.transform(input_df)

    # Step 6: Predict + reverse log transform
    log_pred = model.predict(input_scaled)[0]
    price    = np.expm1(log_pred)

    return price

st.markdown("---")

predict_col, _, _ = st.columns([1, 1, 1])
with predict_col:
    predict_clicked = st.button(
        "🔮 Predict Fair Market Price",
        use_container_width=True,
        disabled=(car_model is None)  # Disable if no valid model
    )

# Safety check before predicting
if predict_clicked:
    if car_model is None:
        st.error("❌ Please select a valid Brand + Engine Type combination first!")
    else:
        with st.spinner("Analysing 136 features..."):
            predicted_price = predict_price(
                brand, car_model, engine_type, transmission,
                location, state, car_age, driven_kms, is_top_trim
            )
        
        st.success(f"✅ Predicted Price: **₹{predicted_price:.2f} Lakhs**")
        st.balloons()


st.markdown("---")
st.subheader("📊 Market Insights From Our Dataset")

ins_col1, ins_col2, ins_col3 = st.columns(3)

with ins_col1:
    st.markdown("**💰 Price by City**")
    city_avg = df.groupby('Location')['Price_In_Lakhs'].mean().sort_values(ascending=False)
    st.bar_chart(city_avg)

with ins_col2:
    st.markdown("**🏷️ Top 8 Brands by Avg Price**")
    brand_avg = df.groupby('Brand')['Price_In_Lakhs'].mean().sort_values(ascending=False).head(8)
    st.bar_chart(brand_avg)

with ins_col3:
    st.markdown("**📅 Depreciation Curve**")
    age_avg = df.groupby('Car_Age')['Price_In_Lakhs'].mean()
    st.line_chart(age_avg)


st.markdown("---")
st.caption(
    "Developed by Bhargav Praveen Chintapalli"
)
