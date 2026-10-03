
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Airbnb Analytics Dashboard",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Airbnb Data Analysis Dashboard")
st.caption(
    "Explore Airbnb listings, prices, room types, ratings "
    "and availability using interactive filters."
)


# --------------------------------------------------
# 2. Load the cleaned dataset
# --------------------------------------------------
DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "listings_cleaned.csv"
)


@st.cache_data
def load_data(path):
    data = pd.read_csv(path, low_memory=False)

    if "price" in data.columns:
        data["price"] = pd.to_numeric(
            data["price"].astype("string")
            .str.replace(r"[$,]", "", regex=True)
            .str.strip(),
            errors="coerce"
        )

    numeric_columns = [
        "number_of_reviews",
        "reviews_per_month",
        "availability_365",
        "review_scores_rating",
        "latitude",
        "longitude",
        "minimum_nights",
        "accommodates"
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

    return data


if not DATA_PATH.exists():
    st.error(
        f"Dataset not found: {DATA_PATH}\n\n"
        "Run the Jupyter Notebook first and export "
        "listings_cleaned.csv into the data folder."
    )
    st.stop()

df = load_data(str(DATA_PATH))

if df.empty:
    st.error("The dataset contains no rows.")
    st.stop()


# --------------------------------------------------
# 3. Detect relevant columns
# --------------------------------------------------
neighborhood_col = next(
    (
        c for c in [
            "neighbourhood_cleansed",
            "neighbourhood",
            "neighborhood"
        ]
        if c in df.columns
    ),
    None
)

room_col = "room_type" if "room_type" in df.columns else None

if neighborhood_col:
    df[neighborhood_col] = (
        df[neighborhood_col].fillna("Unknown").astype(str)
    )

if room_col:
    df[room_col] = df[room_col].fillna("Unknown").astype(str)


# --------------------------------------------------
# 4. Sidebar filters
# --------------------------------------------------
st.sidebar.header("Dashboard Filters")

filtered = df.copy()

if neighborhood_col:
    neighborhoods = sorted(
        filtered[neighborhood_col].unique().tolist()
    )

    selected_neighborhoods = st.sidebar.multiselect(
        "Select neighborhoods",
        options=neighborhoods,
        default=neighborhoods
    )

    filtered = filtered[
        filtered[neighborhood_col].isin(selected_neighborhoods)
    ]

if room_col:
    room_types = sorted(filtered[room_col].unique().tolist())

    selected_rooms = st.sidebar.multiselect(
        "Select room types",
        options=room_types,
        default=room_types
    )

    filtered = filtered[
        filtered[room_col].isin(selected_rooms)
    ]

if "price" in filtered.columns:
    valid_price = filtered["price"].dropna()
    valid_price = valid_price[valid_price > 0]

    if not valid_price.empty:
        lower = float(valid_price.min())
        upper = float(valid_price.max())

        if lower < upper:
            price_range = st.sidebar.slider(
                "Nightly price range",
                min_value=lower,
                max_value=upper,
                value=(lower, upper)
            )

            filtered = filtered[
                filtered["price"].between(
                    price_range[0], price_range[1]
                )
            ]
        else:
            st.sidebar.caption(
                f"All available prices are {lower:.2f}."
            )
    else:
        st.sidebar.info("No valid prices available.")


# --------------------------------------------------
# 5. Key performance indicators
# --------------------------------------------------
st.subheader("Key Metrics")

total_listings = len(filtered)

avg_price = (
    filtered.loc[filtered["price"] > 0, "price"].mean()
    if "price" in filtered.columns
    else np.nan
)

avg_rating = (
    filtered["review_scores_rating"].mean()
    if "review_scores_rating" in filtered.columns
    else np.nan
)

avg_availability = (
    filtered["availability_365"].mean()
    if "availability_365" in filtered.columns
    else np.nan
)

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Listings", f"{total_listings:,}")
k2.metric(
    "Average Nightly Price",
    f"{avg_price:,.2f}" if pd.notna(avg_price) else "N/A"
)
k3.metric(
    "Average Rating",
    f"{avg_rating:.2f}/5" if pd.notna(avg_rating) else "N/A"
)
k4.metric(
    "Average Available Nights",
    f"{avg_availability:.1f}"
    if pd.notna(avg_availability) else "N/A"
)

st.caption(
    "Metrics are calculated from the filtered listings. "
    "Prices are in the dataset's currency. Available nights "
    "are not the same as confirmed bookings or occupancy."
)


# --------------------------------------------------
# 6. Charts: neighborhood and room type
# --------------------------------------------------
st.divider()
st.subheader("Listing Distribution")

left, right = st.columns(2)

with left:
    if neighborhood_col and not filtered.empty:
        neighborhood_counts = (
            filtered[neighborhood_col]
            .value_counts()
            .head(15)
            .rename_axis("Neighborhood")
            .reset_index(name="Listings")
        )

        fig = px.bar(
            neighborhood_counts,
            x="Listings",
            y="Neighborhood",
            orientation="h",
            title="Top 15 Neighborhoods by Listings",
            template="plotly_white"
        )

        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Neighborhood data is unavailable.")

with right:
    if room_col and not filtered.empty:
        room_counts = (
            filtered[room_col]
            .value_counts()
            .rename_axis("Room Type")
            .reset_index(name="Listings")
        )

        fig = px.pie(
            room_counts,
            names="Room Type",
            values="Listings",
            title="Listings by Room Type",
            hole=0.35
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Room-type data is unavailable.")


# --------------------------------------------------
# 7. Pricing analysis
# --------------------------------------------------
st.divider()
st.subheader("Pricing Analysis")

left, right = st.columns(2)

if "price" in filtered.columns:
    price_data = filtered[
        filtered["price"].notna() & (filtered["price"] > 0)
    ].copy()

    if not price_data.empty:
        # Use the 99th percentile for chart readability only
        price_cap = price_data["price"].quantile(0.99)
        price_plot = price_data[
            price_data["price"] <= price_cap
        ]

        with left:
            fig = px.histogram(
                price_plot,
                x="price",
                nbins=40,
                title="Nightly Price Distribution",
                labels={"price": "Nightly price"},
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            if room_col:
                room_prices = (
                    price_plot.groupby(room_col)["price"]
                    .median()
                    .reset_index()
                )

                fig = px.bar(
                    room_prices,
                    x=room_col,
                    y="price",
                    title="Median Price by Room Type",
                    labels={
                        room_col: "Room type",
                        "price": "Median nightly price"
                    },
                    template="plotly_white"
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Room type is unavailable.")
    else:
        st.info("No valid prices are available for charting.")


# --------------------------------------------------
# 8. Price vs. reviews
# --------------------------------------------------
st.divider()
st.subheader("Price vs. Reviews")

if (
    "price" in filtered.columns
    and "number_of_reviews" in filtered.columns
):
    scatter_data = filtered.dropna(
        subset=["price", "number_of_reviews"]
    )

    scatter_data = scatter_data[
        (scatter_data["price"] > 0)
        & (scatter_data["price"] <= scatter_data["price"].quantile(0.99))
    ]

    if not scatter_data.empty:
        fig = px.scatter(
            scatter_data,
            x="number_of_reviews",
            y="price",
            color=room_col if room_col else None,
            hover_data=[
                c for c in [
                    "name",
                    neighborhood_col,
                    "review_scores_rating"
                ]
                if c and c in scatter_data.columns
            ],
            opacity=0.65,
            title="Nightly Price vs. Number of Reviews",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("There are no records for this scatter plot.")
else:
    st.info("Price or review-count columns are unavailable.")


# --------------------------------------------------
# 9. Map of listing locations
# --------------------------------------------------
st.divider()
st.subheader("Listing Locations")

if (
    "latitude" in filtered.columns
    and "longitude" in filtered.columns
):
    map_data = filtered.dropna(
        subset=["latitude", "longitude"]
    )

    if not map_data.empty:
        map_data = map_data[
            map_data["latitude"].between(-90, 90)
            & map_data["longitude"].between(-180, 180)
        ]

        if not map_data.empty:
            map_data = map_data.head(5000)

            st.map(
                map_data[["latitude", "longitude"]],
                latitude="latitude",
                longitude="longitude"
            )
            st.caption(
                "The map displays up to 5,000 filtered listings "
                "to keep the dashboard responsive."
            )
    else:
        st.info("No valid coordinates are available.")
else:
    st.info("Latitude and longitude columns are unavailable.")


# --------------------------------------------------
# 10. Property details table
# --------------------------------------------------
st.divider()
st.subheader("Filtered Property Details")

preferred_columns = [
    "id",
    "name",
    neighborhood_col,
    "room_type",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "availability_365"
]

display_columns = [
    c for c in preferred_columns
    if c and c in filtered.columns
]

if display_columns:
    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True
    )
else:
    st.dataframe(filtered, use_container_width=True)


# --------------------------------------------------
# 11. Download filtered data
# --------------------------------------------------
st.divider()
st.subheader("Export Data")

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Listings as CSV",
    data=csv_data,
    file_name="airbnb_filtered_listings.csv",
    mime="text/csv"
)

st.success("Dashboard data loaded successfully.")