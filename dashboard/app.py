import streamlit as st
import pandas as pd
import joblib
from pathlib import Path



st.set_page_config(
    page_title="HousePrice AI",
    page_icon="🏠",
    layout="wide"
)


project_dir = Path(__file__).resolve().parents[1]
model = joblib.load(project_dir / "models" / "house_price_model.pkl")
default_features = (
    pd.read_csv(project_dir / "data" / "dataFeature.csv")
    .drop(columns=["SalePrice"])
    .loc[:, model.feature_names_in_]
    .iloc[[0]]
    .copy()
)



st.title("🏠 HousePrice AI")
st.subheader("Prédiction du prix d'un logement")

st.write(
    "Saisissez les caractéristiques du logement "
    "pour obtenir une estimation de son prix."
)


# ==========================================
# Formulaire
# ==========================================

with st.form("house_form"):

    col1, col2, col3 = st.columns(3)

    with col1:

        overall_qual = st.number_input(
            "Qualité globale",
            min_value=1,
            max_value=10,
            value=5
        )

        gr_liv_area = st.number_input(
            "Surface habitable (sq ft)",
            min_value=0,
            value=1500
        )

        year_built = st.number_input(
            "Année de construction",
            min_value=1800,
            max_value=2026,
            value=2000
        )

    with col2:

        total_bsmt_sf = st.number_input(
            "Surface sous-sol",
            min_value=0,
            value=800
        )

        garage_cars = st.number_input(
            "Places de garage",
            min_value=0,
            max_value=10,
            value=2
        )

        full_bath = st.number_input(
            "Nombre de salles de bain",
            min_value=0,
            max_value=10,
            value=2
        )

    with col3:

        bedroom_abv_gr = st.number_input(
            "Chambres",
            min_value=0,
            max_value=20,
            value=3
        )

        fireplaces = st.number_input(
            "Cheminées",
            min_value=0,
            max_value=10,
            value=1
        )

        garage_area = st.number_input(
            "Surface garage",
            min_value=0,
            value=400
        )

    submitted = st.form_submit_button(
        "🔮 Prédire le prix"
    )


# ==========================================
# Prediction
# ==========================================

if submitted:

    data = default_features.copy()
    data["OverallQual"] = overall_qual
    data["GrLivArea"] = gr_liv_area
    data["YearBuilt"] = year_built
    data["TotalBsmtSF"] = total_bsmt_sf
    data["GarageCars"] = garage_cars
    data["FullBath"] = full_bath
    data["BedroomAbvGr"] = bedroom_abv_gr
    data["Fireplaces"] = fireplaces
    data["GarageArea"] = garage_area

    data["TotalSF"] = data["GrLivArea"] + data["TotalBsmtSF"]
    data["TotalBath"] = (
        data["FullBath"] + 0.5 * data["HalfBath"]
        + data["BsmtFullBath"] + 0.5 * data["BsmtHalfBath"]
    )
    data["HouseAge"] = data["YrSold"] - data["YearBuilt"]
    data["YearsSinceRemod"] = data["YrSold"] - data["YearRemodAdd"]

    prediction = model.predict(data)[0]

    st.success(
        f"💰 Prix estimé : {prediction:,.0f}"
    )