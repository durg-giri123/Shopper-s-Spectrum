import streamlit as st
import joblib
import numpy as np

# =========================
# Load Models
# =========================

cluster_model = joblib.load("cluster_model.pkl")
scaler = joblib.load("scaler.pkl")
similarity_df = joblib.load("similarity_matrix.pkl")

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

# =========================
# Sidebar
# =========================

st.sidebar.title("🛒 Shopper Spectrum")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Customer Segmentation",
        "Product Recommendation"
    ]
)

# =========================
# Recommendation Function
# =========================

def recommend_products(product_name, n=5):

    try:
        product_name = product_name.upper()

        products = similarity_df.index.str.upper()

        if product_name not in products:
            return ["❌ Product Not Found"]

        actual_name = similarity_df.index[
            products.get_loc(product_name)
        ]

        recommendations = (
            similarity_df[actual_name]
            .sort_values(ascending=False)
            [1:n+1]
        )

        return recommendations.index.tolist()

    except Exception as e:
        return [f"Error: {e}"]


# =========================
# HOME PAGE
# =========================

if menu == "Home":

    st.title("🛒 Shopper Spectrum")

    st.markdown("""
    ## Customer Segmentation & Product Recommendation System

    This project helps businesses:

    - 🎯 Identify High-Value Customers
    - ⚠️ Detect At-Risk Customers
    - 📈 Improve Customer Retention
    - 🛍 Recommend Similar Products
    - 💰 Increase Revenue through Personalization
    """)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Customers", "4,338")

    with col2:
        st.metric("Products", "3,877")

    with col3:
        st.metric("Clusters", "5")

    st.markdown("---")

    st.info(
        "Built using RFM Analysis, K-Means Clustering and Recommendation System."
    )

# =========================
# CUSTOMER SEGMENTATION
# =========================

elif menu == "Customer Segmentation":

    st.title("👥 Customer Segmentation")

    recency = st.number_input(
        "Recency (Days)",
        min_value=0,
        value=30
    )

    frequency = st.number_input(
        "Frequency",
        min_value=0,
        value=5
    )

    monetary = st.number_input(
        "Monetary Value",
        min_value=0.0,
        value=1000.0
    )

    if st.button("Predict Segment"):

        data = np.array([
            [recency, frequency, monetary]
        ])

        scaled_data = scaler.transform(data)

        cluster = cluster_model.predict(
            scaled_data
        )[0]

        st.markdown("---")

        if cluster == 4:
            st.success("🌟 VIP Customer")
            st.write(
                "Very high spending and highly engaged customer."
            )

        elif cluster == 3:
            st.success("💎 Loyal Customer")
            st.write(
                "Purchases frequently and contributes strong revenue."
            )

        elif cluster == 2:
            st.success("🏆 High Value Customer")
            st.write(
                "Generates significant revenue for the business."
            )

        elif cluster == 1:
            st.warning("⚠️ At Risk Customer")
            st.write(
                "Has not purchased recently and may churn."
            )

        else:
            st.info("🙂 Regular Customer")
            st.write(
                "Average purchasing behavior."
            )

# =========================
# PRODUCT RECOMMENDATION
# =========================

elif menu == "Product Recommendation":

    st.title("🛍 Product Recommendation System")

    product_name = st.text_input(
        "Enter Product Name"
    )

    if st.button("Get Recommendations"):

        recommendations = recommend_products(
            product_name
        )

        st.subheader("Recommended Products")

        for item in recommendations:
            st.success(item)

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Shopper Spectrum | Customer Segmentation & Product Recommendation System"
)