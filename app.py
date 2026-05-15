import os
import time
from datetime import date

import folium
import joblib
import streamlit as st
from catboost import CatBoostRegressor
from openai import OpenAI
from streamlit.errors import StreamlitSecretNotFoundError
from streamlit_folium import st_folium
from utils import (
    generate_pricing_output,
    get_pricing_explanation,
    prepare_input_for_model,
    preset_1,
    resolve_api_key,
    validate,
)

MAINTENANCE_MODE = False  # change to False when ready

if MAINTENANCE_MODE:
    st.title("🚧 Maintenance mode")
    st.write("The app is temporarily unavailable. Please try later.")
    st.stop()

st.set_option("client.showErrorDetails", True)


# Wczytanie modeli
@st.cache_resource
def load_models():
    kmeans = joblib.load("models/kmeans_model.joblib")

    cat_model = CatBoostRegressor()
    cat_model.load_model("models/catboost_model_test.bin")

    explainer = joblib.load("models/final_explainer.joblib")

    return kmeans, cat_model, explainer


kmeans, cat_model, explainer = load_models()

DEFAULT_PROMPT = (
    "Describe what you see in this image in simple language. "
    "Mention key objects, scene, and possible context."
)
FALLBACK_MODELS = ["gpt-4.1-mini", "gpt-4.1-nano"]


def get_secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except StreamlitSecretNotFoundError:
        return default


DISABLE_MANUAL_API_KEY_INPUT = str(
    get_secret("DISABLE_MANUAL_API_KEY_INPUT", "")
).lower() in {"1", "true", "yes"}


def get_int_config(name, default):
    raw_value = get_secret(name, os.environ.get(name, default))
    try:
        value = int(raw_value)
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


MAX_REQUESTS_PER_MINUTE = get_int_config("MAX_REQUESTS_PER_MINUTE", 5)
MAX_REQUESTS_PER_DAY = get_int_config("MAX_REQUESTS_PER_DAY", 100)


def fetch_available_gpt_models(client):
    models = client.models.list()
    available_model_ids = {model.id for model in models.data}
    allowed_models = [
        model for model in FALLBACK_MODELS if model in available_model_ids
    ]
    return allowed_models


# Streamlit app layout
st.set_page_config(page_title="Real estate price estimation", page_icon=":camera:")
st.title("Real estate price estimation - Poznań")
st.write(
    "Hi, welcome to the real estate price estimation app! Input parameters of the real estate and write description of the property. We will generate a price estimation \n with explanation provided by state-of-the-art AI language model."
)
st.caption(
    f"Rate limits: {MAX_REQUESTS_PER_MINUTE} request(s)/minute and {MAX_REQUESTS_PER_DAY} request(s)/day per user session."
)

api_key = resolve_api_key()
if not api_key:
    api_key = get_secret("OPENAI_API_KEY", "")

if not DISABLE_MANUAL_API_KEY_INPUT:
    manual_api_key = st.text_input(
        "OpenAI API key (optional if OPENAI_API_KEY is set)",
        type="password",
        help="If empty, the app will use OPENAI_API_KEY from environment variables or Streamlit secrets.",
    )
    if manual_api_key:
        api_key = manual_api_key

map_lat = None
map_lon = None

if api_key:
    preset_dict = {}

    preset_row_1 = preset_1()
    preset_1_label = (
        "New apartment, 100m², 3 rooms, 5th floor in 6-floor building, to completion"
    )

    preset_dict[preset_1_label] = preset_row_1

    preset = st.selectbox(
        "Choose example property",
        options=["None"] + list(preset_dict.keys()),
    )

    if preset != "None":
        preset_row = preset_dict[preset]

        st.session_state.market = preset_row["market"]
        st.session_state.map_lat = preset_row["map_lat"]
        st.session_state.map_lon = preset_row["map_lon"]
        st.session_state.equipment = preset_row["equipment"]
        st.session_state.extras = preset_row["extras"]
        st.session_state.security = preset_row["security"]
        st.session_state.media = preset_row["media"]
        st.session_state.construction_status = preset_row["construction_status"]
        st.session_state.area = float(preset_row["area"])
        st.session_state.no_rooms = int(preset_row["no_rooms"])
        st.session_state.floor_no = int(preset_row["floor_no"])
        st.session_state.building_floors_num = int(preset_row["building_floors_num"])
        st.session_state.building_type = preset_row["building_type"]
        st.session_state.selected_location = [
            float(preset_row["map_lat"]),
            float(preset_row["map_lon"]),
        ]

    st.header("Select property location")

    # Initialize session state
    if "selected_location" not in st.session_state:
        st.session_state.selected_location = None

    # Center on Poznań
    center = [52.4064, 16.9252]

    # Create map
    m = folium.Map(location=center, zoom_start=12)

    # Add click popup (optional)
    m.add_child(folium.LatLngPopup())

    # Add marker if location already selected
    if st.session_state.selected_location:
        folium.CircleMarker(
            location=st.session_state.selected_location,
            radius=8,
            color="red",
            fill=True,
            fill_opacity=0.9,
        ).add_to(m)

    # Render map
    output = st_folium(m, height=500, width=700)

    # Handle click safely
    if output and output.get("last_clicked"):
        map_lat = output["last_clicked"]["lat"]
        map_lon = output["last_clicked"]["lng"]

        # Save location
        st.session_state.selected_location = [map_lat, map_lon]

        st.rerun()  # refresh to show marker instantly

    # Display selected location (outside click block!)
    if st.session_state.selected_location:
        map_lat, map_lon = st.session_state.selected_location

        st.success(f"Selected location: {map_lat:.5f}, {map_lon:.5f}")

        # Validation
        if not (52.2 <= map_lat <= 52.6 and 16.6 <= map_lon <= 17.2):
            st.warning("Location is outside Poznań area")

    st.header("Input property details and description")
    market = st.selectbox("Market", options=["primary", "secondary"], key="market")

    construction_status = st.selectbox(
        "Construction Status",
        options=["to renovation", "ready to use", "to completion", "I don't know"],
        index=0,
        key="construction_status",
    )

    area = st.number_input(
        "Area (m²)", min_value=0.0, max_value=1000.0, step=1.0, key="area"
    )

    no_rooms = st.number_input(
        "Number of rooms (integer value only)",
        min_value=1,
        max_value=15,
        step=1,
        key="no_rooms",
    )

    floor_no = st.number_input(
        "Floor number (integer value only)",
        min_value=1,
        max_value=30,
        step=1,
        key="floor_no",
    )

    building_floors_num = st.number_input(
        "Number of floors in the building (integer value only)",
        min_value=1,
        max_value=30,
        step=1,
        key="building_floors_num",
    )

    building_type = st.selectbox(
        "Building Type (optional)",
        options=["block", "apartment"],
        key="building_type",
    )

    equipment = st.multiselect(
        "Select equipment available in the property (optional)",
        ["stove", "furniture", "dishwasher", "fridge", "tv", "oven", "washing machine"],
        key="equipment",
    )

    extras = st.multiselect(
        "Select extras available in the property (optional)",
        [
            "basement",
            "separate kitchen",
            "balcony",
            "garage",
            "lift",
            "garden",
            "terrace",
            "usable room",
            "two storey",
            "air conditioning",
        ],
        key="extras",
    )

    security = st.multiselect(
        "Select security features of the real estate (optional)",
        [
            "closed area",
            "anti burglary door",
            "entryphone",
            "alarm",
            "monitoring",
            "roller shutters",
        ],
        key="security",
    )

    media = st.multiselect(
        "Select media features of the real estate (optional)",
        ["internet", "phone", "cable-television"],
        key="media",
    )

    description = st.text_area(
        "Description (optional)",
        value=None,
        height=120,
    )

    conf = st.number_input(
        "Confidence level [%] of the price range (e.g. if 90%, then there is 5% chance that the actual price is below\n the lower bound and 5% chance that it is above the upper bound)",
        min_value=10,
        max_value=100,
        step=1,
        value=50,
    )

    errors = validate(
        market,
        construction_status,
        area,
        no_rooms,
        floor_no,
        building_floors_num,
        building_type,
        map_lat,
        map_lon,
    )

    # Choosing GPT model for explanation

    client = OpenAI(api_key=api_key)
    try:
        available_models = fetch_available_gpt_models(client)
    except Exception as e:
        available_models = FALLBACK_MODELS
        st.warning(
            f"Could not fetch models from API. Using fallback list. Details: {e}"
        )

    if not available_models:
        available_models = FALLBACK_MODELS
        st.warning("No GPT models were returned by the API. Using fallback model list.")

    default_model = (
        "gpt-4.1-mini" if "gpt-4.1-mini" in available_models else available_models[0]
    )
    default_index = available_models.index(default_model)
    model_choice = st.selectbox("Select model", available_models, index=default_index)

    custom_model = st.text_input(
        "Custom model ID (optional) used for explanation",
        value="",
        help="If set, this value overrides the model selected above.",
    )

    if len(errors) == 0:
        try:
            if st.button("Generate Pricing Prediction", type="primary"):
                if "request_timestamps" not in st.session_state:
                    st.session_state["request_timestamps"] = []
                if "daily_usage" not in st.session_state:
                    st.session_state["daily_usage"] = {
                        "date": date.today().isoformat(),
                        "count": 0,
                    }

                now = time.time()
                recent_timestamps = [
                    ts for ts in st.session_state["request_timestamps"] if now - ts < 60
                ]
                st.session_state["request_timestamps"] = recent_timestamps

                if len(recent_timestamps) >= MAX_REQUESTS_PER_MINUTE:
                    st.error(
                        "Rate limit reached: too many requests in the last minute. Please wait."
                    )
                    st.stop()

                today = date.today().isoformat()
                if st.session_state["daily_usage"]["date"] != today:
                    st.session_state["daily_usage"] = {"date": today, "count": 0}

                if st.session_state["daily_usage"]["count"] >= MAX_REQUESTS_PER_DAY:
                    st.error(
                        "Daily limit reached for this session. Please try again tomorrow."
                    )
                    st.stop()

                st.session_state["request_timestamps"].append(now)
                st.session_state["daily_usage"]["count"] += 1

                with st.spinner("Preparing input for the model..."):
                    input_df = prepare_input_for_model(
                        market,
                        map_lat,
                        map_lon,
                        construction_status,
                        area,
                        no_rooms,
                        floor_no,
                        building_floors_num,
                        building_type,
                        equipment,
                        extras,
                        security,
                        media,
                        kmeans,
                    )
                with st.spinner("Generating price estimation..."):
                    predicted_price, lower_bound, upper_bound = generate_pricing_output(
                        input_df, cat_model, conf
                    )
                st.subheader("Result:")
                st.markdown(
                    f"""
                <div style="
                    padding: 20px;
                    border-radius: 12px;
                    background-color: #f0f2f6;
                    text-align: center;
                    color: #000000;
                ">
                    <h2>🏠 Estimated Price [per m²]</h2>
                    <h1 style="color:#2E86C1;">{int(round(predicted_price, -1))} PLN</h1>
                    <h2> Estimated Price Range [per m²] with {conf}% coverage</h2>
                    <h1 style="color:#2E86C1;">{int(round(lower_bound, -1))} – {int(round(upper_bound, -1))} PLN</h1>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                with st.spinner("Calling Open AI model..."):
                    description_final = description if description else DEFAULT_PROMPT
                    description_final += "Prepare the explanation in bullet points"
                    final_model = custom_model.strip() or model_choice
                    description = get_pricing_explanation(
                        client,
                        description_final,
                        explainer,
                        input_df,
                        model_choice,
                        predicted_price,
                    )
                st.subheader(f"Explanation provided by the {model_choice} model:")
                st.write(description)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Please fix the following errors before submitting:")
        for error in errors:
            st.error(f"- {error}")

else:
    if DISABLE_MANUAL_API_KEY_INPUT:
        st.warning(
            "No API key configured. Set OPENAI_API_KEY in environment variables or Streamlit secrets."
        )
    else:
        st.warning(
            "Please provide an API key in the field above or set OPENAI_API_KEY locally."
        )
