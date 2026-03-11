import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ══════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════
st.title("🚗 Cars24 Price Predictor")
st.markdown("#### Predict the fair market value of any used car in India")
st.markdown("*Trained on 14,907 real Cars24 listings across 8 cities*")
st.markdown("---")

# ══════════════════════════════════════════════════
# INPUT SECTION
# ══════════════════════════════════════════════════
st.subheader("🔍 Enter Car Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🏷️ Car Identity**")

    brand = st.selectbox(
        "Brand",
        options=sorted(df['Brand'].unique().tolist()),
        help="Select the car manufacturer"
    )

    # Filter models by selected brand
    brand_models = df[df['Brand'] == brand]['Model'].unique().tolist()
    car_model = st.selectbox(
        "Model",
        options=sorted(brand_models),
        help="Models filtered by selected brand"
    )

    engine_type = st.selectbox(
        "Engine Type",
        options=df['Engine_Type'].unique().tolist(),
        help="Petrol, Diesel or Other (CNG/Electric)"
    )

with col2:
    st.markdown("**📍 Location**")

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

    transmission = st.selectbox(
        "Transmission",
        options=df['Transmission'].unique().tolist(),
        help="Manual or Automatic"
    )

with col3:
    st.markdown("**🔧 Car Condition**")

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
        help="Total KMs driven by the car"
    )

    is_top_trim = st.radio(
        "Variant Type",
        options=[0, 1],
        format_func=lambda x: "🔝 Top Trim (ZXI/Asta/Alpha/AMT)" if x == 1
                               else "⚙️ Base / Mid Trim",
        help="Is this a premium or basic variant?"
    )

# ══════════════════════════════════════════════════
# PREDICTION FUNCTION
# ══════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════
# PREDICT BUTTON
# ══════════════════════════════════════════════════
st.markdown("---")

predict_col, _, _ = st.columns([1, 1, 1])
with predict_col:
    predict_clicked = st.button(
        "🔮 Predict Fair Market Price",
        use_container_width=True
    )

if predict_clicked:
    with st.spinner("Analysing 136 features..."):
        predicted_price = predict_price(
            brand, car_model, engine_type, transmission,
            location, state, car_age, driven_kms, is_top_trim
        )

    st.markdown("---")
    st.subheader("💰 Prediction Results")

    # ── Main price display ─────────────────────────
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)

    with res_col1:
        st.metric(
            label="🎯 Predicted Price",
            value=f"₹{predicted_price:.2f} Lakhs"
        )
    with res_col2:
        lower = predicted_price * 0.90
        st.metric(
            label="📉 Negotiate At",
            value=f"₹{lower:.2f} Lakhs",
            delta=f"-₹{predicted_price - lower:.2f}L",
            delta_color="inverse"
        )
    with res_col3:
        upper = predicted_price * 1.10
        st.metric(
            label="📈 Max Fair Price",
            value=f"₹{upper:.2f} Lakhs",
            delta=f"+₹{upper - predicted_price:.2f}L"
        )
    with res_col4:
        monthly_emi = (predicted_price * 100000 * 0.09 / 12)
        st.metric(
            label="💳 Approx EMI",
            value=f"₹{monthly_emi:,.0f}/mo",
            help="Estimated at 9% interest, 5-year loan"
        )

    # ── Price range bar ────────────────────────────
    st.markdown("---")
    st.markdown("**📊 Price Range Analysis**")

    range_col1, range_col2 = st.columns([2, 1])

    with range_col1:
        # Similar cars in dataset
        similar = df[
            (df['Brand'] == brand) &
            (df['Car_Age'].between(car_age - 1, car_age + 1))
        ]['Price_In_Lakhs']

        if len(similar) > 5:
            st.markdown(f"*Based on {len(similar)} similar cars in our dataset:*")
            sim_col1, sim_col2, sim_col3 = st.columns(3)
            with sim_col1:
                st.metric("Lowest Similar", f"₹{similar.min():.2f}L")
            with sim_col2:
                st.metric("Average Similar", f"₹{similar.mean():.2f}L")
            with sim_col3:
                st.metric("Highest Similar", f"₹{similar.max():.2f}L")
        else:
            st.info("Not enough similar cars in dataset for comparison")

    with range_col2:
        # Buyer verdict
        if len(similar) > 5:
            avg_similar = similar.mean()
            diff = predicted_price - avg_similar
            if diff < -0.5:
                st.success("🟢 GOOD DEAL\nBelow market average!")
            elif diff > 0.5:
                st.warning("🔴 ABOVE MARKET\nNegotiate the price down")
            else:
                st.info("🟡 FAIR PRICE\nIn line with market")

    # ── Car Summary ────────────────────────────────
    st.markdown("---")
    st.subheader("🚘 Car Summary")

    sum_col1, sum_col2 = st.columns(2)
    with sum_col1:
        st.markdown(f"**Brand**         : {brand}")
        st.markdown(f"**Model**         : {car_model}")
        st.markdown(f"**Engine Type**   : {engine_type}")
        st.markdown(f"**Transmission**  : {transmission}")
    with sum_col2:
        st.markdown(f"**Year**          : {manufacture_year} ({car_age} years old)")
        st.markdown(f"**KMs Driven**    : {driven_kms:,} km")
        st.markdown(f"**City**          : {location}")
        st.markdown(f"**Variant**       : {'Top Trim' if is_top_trim else 'Base/Mid Trim'}")

    # ── Market insight ─────────────────────────────
    st.markdown("---")
    if predicted_price < 2:
        st.info("💡 **Budget Segment** — Great entry-level buy. Parts and maintenance are cheap.")
    elif predicted_price < 5:
        st.info("💡 **Value Segment** — Most popular price band. High demand, easy resale.")
    elif predicted_price < 8:
        st.info("💡 **Mid Premium** — Good balance of features and value. Selective buyers.")
    else:
        st.info("💡 **Premium Segment** — Luxury features. Smaller buyer pool, longer to sell.")

# ══════════════════════════════════════════════════
# MARKET INSIGHTS SECTION
# ══════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "Built with ❤️ | Data: Cars24 (15,673 listings) | "
    "Model: XGBoost (R²=0.90) | "
    "Deployed on Streamlit Cloud"
)
