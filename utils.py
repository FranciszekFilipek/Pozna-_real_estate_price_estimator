import os

import numpy as np
import osmnx as ox
import pandas as pd
from dotenv import load_dotenv
from scipy.spatial import cKDTree
from scipy.stats import norm

load_dotenv()


def get_pricing_explanation(
    client, description, explainer, input_df, model, predicted_price
):
    # shap_values = explainer(input_df)
    cat_cols = [
        "market",
        "no_rooms",
        "construction_status",
        "building_type",
        "floor_no",
        "location_cluster",
    ]
    input_df[cat_cols] = input_df[cat_cols].astype(str)
    shap_values = explainer(input_df)

    feature_names = input_df.columns
    feature_values = input_df.iloc[0]

    contributions = []

    for feature, value, shap_val in zip(
        feature_names, feature_values, shap_values.values[0]
    ):
        contributions.append(
            {"feature": feature, "value": value, "impact": shap_val.item()}
        )
    contributions = input_df.columns[np.argsort(shap_values.values[0])][::-1][:10]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You explain real estate valuation results "
                    "to average non-technical users."
                ),
            },
            {
                "role": "user",
                "content": f"""
            Predicted apartment price: {predicted_price:,.0f} PLN

        Feature contributions:
        {contributions}

        Explain:
        - what increased the price
        - what decreased the price
        - use simple language
        - avoid technical ML jargon
        - keep it under 120 words

        Reference description which I provided:
        {description}
        """,
            },
        ],
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()


def resolve_api_key():
    return os.environ.get("OPENAI_API_KEY", "")


def get_length(feature):
    if isinstance(feature, list):
        return len(feature)
    else:
        return 0


def transform_estate_params(
    construction_status, equipment, extras, security, media, no_rooms
):
    construction_status_mapping = {
        "to renovation": "to_renovation",
        "ready to use": "ready_to_use",
        "to completion": "to_completion",
        "I don't know": "nan",
    }
    construction_status = construction_status_mapping.get(
        construction_status, construction_status
    )

    equipment_count = get_length(equipment)
    extras_count = get_length(extras)
    security_types = get_length(security)
    media_count = get_length(media)

    if no_rooms in [1, 2, 3]:
        no_rooms = str(no_rooms)
    elif no_rooms >= 4:
        no_rooms = "more_than_3"

    return (
        construction_status,
        equipment_count,
        extras_count,
        security_types,
        media_count,
        no_rooms,
    )


def validate(
    market,
    construction_status,
    area,
    no_rooms,
    floor_no,
    building_floors_num,
    building_type,
    map_lat,
    map_lon,
):
    # Implement validation logic here (e.g., check for required fields, validate data types, etc.)
    errors = []

    # Geographic coordinates validation
    if map_lat is None or map_lon is None:
        errors.append("Latitude and longitude must be provided")

    # MARKET
    if market not in ["primary", "secondary"]:
        errors.append("Invalid market selection")

    # CONSTRUCTION STATUS
    if construction_status not in ["to renovation", "ready to use", "to completion"]:
        errors.append("Invalid construction status")

    if construction_status == "I don't know":
        construction_status = "nan"

    # AREA
    if area is None or area <= 0:
        errors.append("Area must be greater than 0")

    # ROOMS
    if no_rooms < 1 or no_rooms > 15:
        errors.append("Number of rooms must be between 1 and 15")

    # FLOOR
    if floor_no < 1 or floor_no > 30:
        errors.append("Floor number must be between 1 and 30")

    # BUILDING FLOORS
    if building_floors_num < 1 or building_floors_num > 30:
        errors.append("Building floors must be between 1 and 30")

    # LOGICAL CHECK
    if floor_no > building_floors_num:
        errors.append("Floor cannot exceed total building floors")

    # BUILDING TYPE
    if building_type not in ["block", "apartment"]:
        errors.append("Invalid building type")

    return errors


def get_time_features():
    start_date = pd.Timestamp("2018-01-01")
    current_date = pd.Timestamp.now()

    current_month = current_date.month
    month_sin = np.sin(2 * np.pi * current_month / 12)
    month_cos = np.cos(2 * np.pi * current_month / 12)

    time_months = (current_date.year - start_date.year) * 12 + (
        current_date.month - start_date.month
    )

    return month_sin, month_cos, time_months


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def count_points_within_radius(tree, point, radius):
    return len(tree.query_ball_point(point, r=radius))


def get_geographic_features(map_lat, map_lon, kmeans_model):
    center_lat = 52.38666269
    center_lon = 16.95863093

    dist_to_center = haversine(map_lat, map_lon, center_lat, center_lon)

    place = "Poznań, Poland"
    ox.settings.use_cache = True
    ox.settings.log_console = False
    shops = ox.features_from_place(place, tags={"shop": True})
    shops = shops[shops.geometry.type == "Point"]
    # współrzędne sklepów
    shop_coords = np.array([(p.y, p.x) for p in shops.geometry])
    # KDTree
    tree = cKDTree(shop_coords)
    # mieszkania
    estate_location = np.array([map_lat, map_lon])

    dist_shop, _ = tree.query(estate_location, k=1)

    shops_500m = count_points_within_radius(tree, estate_location, radius=0.005)

    location_cluster = kmeans_model.predict([[map_lat, map_lon]])[0]

    return dist_to_center, dist_shop, shops_500m, location_cluster


def prepare_input_for_model(
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
    kmeans_model,
):

    (
        construction_status,
        equipment_count,
        extras_count,
        security_types,
        media_count,
        no_rooms,
    ) = transform_estate_params(
        construction_status, equipment, extras, security, media, no_rooms
    )
    month_sin, month_cos, time_months = get_time_features()
    dist_to_center, dist_shop, shops_500m, location_cluster = get_geographic_features(
        map_lat, map_lon, kmeans_model
    )

    return pd.DataFrame(
        [
            {
                "market": market,
                "no_rooms": no_rooms,
                "m": area,
                "map_lon": map_lon,
                "map_lat": map_lat,
                "month_sin": month_sin,
                "month_cos": month_cos,
                "construction_status": construction_status,
                "security_types": security_types,
                "building_type": building_type,
                "building_floors_num": building_floors_num,
                "floor_no": floor_no,
                "dist_to_center": dist_to_center,
                "dist_to_nearest_shop": dist_shop,
                "shops_500m": shops_500m,
                "equipment_count": equipment_count,
                "extras_count": extras_count,
                "media_count": media_count,
                "time_months": time_months,
                "location_cluster": location_cluster,
            }
        ]
    )


def generate_pricing_output(input_pd, model, confidence_level):
    rmse_catboost_1 = 500
    predicted_price = model.predict(input_pd)[0]
    quantile = norm.ppf(1 - (1 - confidence_level / 100) / 2)
    lower_bound = predicted_price - quantile * rmse_catboost_1
    upper_bound = predicted_price + quantile * rmse_catboost_1
    return predicted_price, lower_bound, upper_bound


def preset_1():
    return {
        "market": "secondary",
        "map_lat": 52.4064,
        "map_lon": 16.8635,
        "construction_status": "to_completion",
        "area": 100,
        "no_rooms": 3,
        "floor_no": 5,
        "building_floors_num": 6,
        "building_type": "apartment",
        "equipment": [],
        "extras": [],
        "security": [],
        "media": [],
    }
