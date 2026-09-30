from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="HousePrice AI", page_icon="H", layout="wide")

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "dataFeature.csv"
MODEL_PATH = PROJECT_DIR / "models" / "house_price_model.pkl"
MODEL_BUNDLE_PATH = PROJECT_DIR / "models" / "house_price_models.pkl"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():
    if MODEL_BUNDLE_PATH.exists():
        return joblib.load(MODEL_BUNDLE_PATH)
    model = joblib.load(MODEL_PATH)
    return {
        "Modèle optimisé": {
            "model": model,
            "mae": None,
            "rmse": None,
            "r2": None,
            "target_transform": "price",
        }
    }


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
    :root {
        --ink: #17352f;
        --muted: #687873;
        --paper: #f4f6f1;
        --line: #dce4dc;
        --leaf: #2d7358;
        --coral: #d87854;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    html, body, [class*="css"] { font-family: 'DM Sans', 'Trebuchet MS', sans-serif; }
    h1, h2, h3 { color: var(--ink); font-family: 'Playfair Display', Georgia, serif; }
    h1 { font-size: 2.65rem; letter-spacing: 0; }
    [data-testid="stSidebar"] { background: #17352f; }
    [data-testid="stSidebar"] * { color: #f4f6f1; }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 16px 18px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    div.stButton > button, div.stFormSubmitButton > button {
        background: var(--leaf);
        border: 0;
        border-radius: 4px;
        color: white;
        font-weight: 700;
        min-height: 44px;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background: #225b45;
        color: white;
    }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); }
    .eyebrow { color: var(--coral); font-size: 0.76rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
    .intro { color: var(--muted); margin-top: -0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


try:
    market_data = load_data()
    model_catalog = load_models()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(f"Impossible de charger les données ou les modèles : {error}")
    st.stop()

if "SalePrice" not in market_data.columns:
    st.error("La colonne SalePrice est absente de data/dataFeature.csv.")
    st.stop()

model_names = list(model_catalog)
if not model_names:
    st.error("Aucun modèle entraîné n'est disponible.")
    st.stop()

with st.sidebar:
    st.markdown("## HousePrice AI")
    st.caption("ANALYSE IMMOBILIÈRE")
    page = st.radio("Espace", ["Marché", "Estimer un bien", "Données"])
    st.markdown("---")
    selected_model_name = st.selectbox("Modèle de prédiction", model_names)
    selected_model_info = model_catalog[selected_model_name]
    selected_model = (
        selected_model_info["model"]
        if isinstance(selected_model_info, dict)
        else selected_model_info
    )
    if isinstance(selected_model_info, dict) and selected_model_info.get("rmse") is not None:
        st.caption(f"RMSE test : {selected_model_info['rmse']:,.0f} $  ·  R² : {selected_model_info['r2']:.3f}")
    else:
        st.caption("Modèle prêt à estimer")

st.markdown('<p class="eyebrow">Habitat · données · estimation</p>', unsafe_allow_html=True)
st.title("Le marché, côté maison.")
st.markdown('<p class="intro">Repères de marché et estimation à partir des caractéristiques du bien.</p>', unsafe_allow_html=True)


if page == "Marché":
    prices = market_data["SalePrice"].dropna()
    metric_columns = st.columns(4)
    metric_columns[0].metric("Biens analysés", f"{len(market_data):,}")
    metric_columns[1].metric("Prix médian", f"{prices.median():,.0f} $")
    metric_columns[2].metric("Prix moyen", f"{prices.mean():,.0f} $")
    metric_columns[3].metric("Surface médiane", f"{market_data['GrLivArea'].median():,.0f} sq ft")

    chart_left, chart_right = st.columns([1, 1.5])
    with chart_left:
        st.subheader("Répartition des prix")
        price_edges = pd.cut(prices, bins=12, retbins=True)[1]
        price_labels = [
            f"{price_edges[index] / 1000:.0f}k–{price_edges[index + 1] / 1000:.0f}k"
            for index in range(len(price_edges) - 1)
        ]
        price_bands = pd.cut(prices, bins=price_edges, labels=price_labels, include_lowest=True)
        price_counts = price_bands.value_counts(sort=False).rename_axis("Tranche de prix").reset_index(name="Biens")
        st.bar_chart(price_counts, x="Tranche de prix", y="Biens", color="#d87854", height=320)
    with chart_right:
        st.subheader("Surface et valeur")
        scatter_data = market_data[["GrLivArea", "SalePrice", "OverallQual"]].dropna()
        st.scatter_chart(
            scatter_data,
            x="GrLivArea",
            y="SalePrice",
            color="OverallQual",
            height=320,
        )

    st.subheader("Prix médian selon la qualité")
    quality_prices = (
        market_data.groupby("OverallQual", as_index=False)["SalePrice"]
        .median()
        .rename(columns={"OverallQual": "Qualité (1–10)", "SalePrice": "Prix médian ($)"})
    )
    st.bar_chart(quality_prices, x="Qualité (1–10)", y="Prix médian ($)", color="#2d7358", height=260)

elif page == "Estimer un bien":
    st.subheader("Portrait du bien")
    st.caption(f"Estimation avec : {selected_model_name}")

    with st.form("house_form"):
        col_home, col_space, col_rooms = st.columns(3)
        with col_home:
            st.markdown("**Le bien**")
            overall_qual = st.slider("Qualité générale", 1, 10, 6)
            overall_cond = st.slider("État général", 1, 10, 5)
            year_built = st.number_input("Année de construction", 1800, 2026, 2000)
            year_remodeled = st.number_input("Année de rénovation", 1800, 2026, 2000)
        with col_space:
            st.markdown("**Les surfaces**")
            gr_liv_area = st.number_input("Surface habitable (sq ft)", 0, 10000, 1500, step=50)
            total_bsmt_sf = st.number_input("Surface sous-sol (sq ft)", 0, 10000, 800, step=50)
            garage_area = st.number_input("Surface garage (sq ft)", 0, 3000, 400, step=25)
            garage_cars = st.number_input("Places de garage", 0, 10, 2)
        with col_rooms:
            st.markdown("**Les pièces**")
            bedroom_abv_gr = st.number_input("Chambres", 0, 20, 3)
            full_bath = st.number_input("Salles de bain", 0, 10, 2)
            half_bath = st.number_input("Demi-salles de bain", 0, 10, 1)
            fireplaces = st.number_input("Cheminées", 0, 10, 1)
        submitted = st.form_submit_button("Estimer le prix", width="stretch")

    if submitted:
        feature_names = getattr(selected_model, "feature_names_in_", market_data.drop(columns=["SalePrice"]).columns)
        default_features = (
            market_data.drop(columns=["SalePrice"])
            .loc[:, feature_names]
            .iloc[[0]]
            .copy()
        )
        default_features["OverallQual"] = overall_qual
        default_features["OverallCond"] = overall_cond
        default_features["GrLivArea"] = gr_liv_area
        default_features["YearBuilt"] = year_built
        default_features["YearRemodAdd"] = year_remodeled
        default_features["TotalBsmtSF"] = total_bsmt_sf
        default_features["GarageCars"] = garage_cars
        default_features["GarageArea"] = garage_area
        default_features["FullBath"] = full_bath
        default_features["HalfBath"] = half_bath
        default_features["BedroomAbvGr"] = bedroom_abv_gr
        default_features["Fireplaces"] = fireplaces
        default_features["TotalSF"] = gr_liv_area + total_bsmt_sf
        default_features["TotalBath"] = (
            full_bath + 0.5 * half_bath
            + default_features["BsmtFullBath"].iloc[0]
            + 0.5 * default_features["BsmtHalfBath"].iloc[0]
        )
        default_features["HouseAge"] = default_features["YrSold"] - year_built
        default_features["YearsSinceRemod"] = default_features["YrSold"] - year_remodeled

        prediction = float(selected_model.predict(default_features)[0])
        if isinstance(selected_model_info, dict) and selected_model_info.get("target_transform") == "log1p":
            import numpy as np

            prediction = float(np.expm1(prediction))
        prediction = max(0, prediction)
        median_price = float(market_data["SalePrice"].median())

        st.markdown("---")
        result_col, profile_col = st.columns([1, 1.2])
        with result_col:
            st.metric("Estimation", f"{prediction:,.0f} $", f"{prediction - median_price:+,.0f} $ vs médiane")
            st.caption("Écart calculé par rapport au prix médian du jeu de données.")
        with profile_col:
            st.markdown("**Votre bien en bref**")
            profile = pd.DataFrame(
                {
                    "Caractéristique": ["Surface totale", "Chambres", "Salles de bain", "Qualité", "Année"],
                    "Valeur": [
                        f"{gr_liv_area + total_bsmt_sf:,} sq ft",
                        str(bedroom_abv_gr),
                        f"{full_bath} + {half_bath / 2:g}",
                        f"{overall_qual}/10",
                        str(year_built),
                    ],
                }
            )
            st.dataframe(profile, hide_index=True, width="stretch")

elif page == "Données":
    prices = market_data["SalePrice"].dropna()
    st.subheader("Jeu de données enrichi")
    data_columns = st.columns(3)
    data_columns[0].metric("Observations", f"{len(market_data):,}")
    data_columns[1].metric("Variables", f"{market_data.shape[1] - 1:,}")
    data_columns[2].metric("Valeurs manquantes", f"{int(market_data.isna().sum().sum()):,}")

    tab_preview, tab_summary = st.tabs(["Aperçu", "Structure"])
    with tab_preview:
        preview_columns = [
            column for column in ["Id", "OverallQual", "GrLivArea", "TotalBsmtSF", "YearBuilt", "GarageCars", "SalePrice"]
            if column in market_data.columns
        ]
        st.dataframe(market_data[preview_columns].head(30), hide_index=True, width="stretch")
        st.download_button(
            "Télécharger les données enrichies",
            data=market_data.to_csv(index=False).encode("utf-8"),
            file_name="house_prices_features.csv",
            mime="text/csv",
        )
    with tab_summary:
        summary = market_data.describe(include="all").transpose().reset_index(names="Variable")
        st.dataframe(summary, hide_index=True, width="stretch")
        st.caption(f"Source : data/dataFeature.csv · {len(market_data.columns)} colonnes, cible : SalePrice")