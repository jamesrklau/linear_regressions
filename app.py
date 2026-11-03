
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split

from regression_toolkit.data import validate_dataset
from regression_toolkit.evaluation import evaluate_models


# Configure the browser page
st.set_page_config(
    page_title="Regression Model Comparison",
    page_icon="📊",
    layout="wide",
)

# Dashboard heading
st.title("Regression Model Comparison Toolkit")

st.write(
    "Upload a dataset, compare regression models, "
    "and explore their predictions."
)

# Sidebar configuration
st.sidebar.header("Dataset Settings")

test_size = st.sidebar.slider(
    "Validation split (%)",
    min_value=10,
    max_value=40,
    value=25,
    step=5,
)

random_state = st.sidebar.number_input(
    "Data split random state",
    min_value=0,
    value=0,
    step=1,
)

model_random_state = st.sidebar.number_input(
    "Model random state",
    min_value=0,
    value=1,
    step=1,
)

# CSV uploader
uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        # Read and preview the uploaded CSV
        df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")

        # Target selection
        target_column = st.selectbox(
            "Select the column you want to predict",
            options=df.columns,
        )

        # Run model comparison
        if st.button("Run Model Comparison", type="primary"):

            with st.spinner("Training and evaluating models..."):

                # Validate and prepare the dataset
                X, y = validate_dataset(df, target_column)

                # Split into training and validation sets
                train_X, val_X, train_y, val_y = train_test_split(
                    X,
                    y,
                    test_size=test_size / 100,
                    random_state=random_state,
                )

                # Evaluate all regression models
                results, prediction_results = evaluate_models(
                    train_X,
                    train_y,
                    val_X,
                    val_y,
                    model_random_state=model_random_state,
                )

            st.success("Model comparison completed!")

            st.subheader("Model Performance")

            st.dataframe(
                results,
                hide_index=True,
                use_container_width=True,
            )

            st.subheader("Best Performing Model")

            best_model = results.iloc[0]

            st.metric(
                label="Best Model (lowest MAE)",
                value=best_model["Model"],
                delta=f"MAE: {best_model['MAE']:.3f}",
                delta_color="off",
            )

    except Exception as error:
        st.error(f"Something went wrong: {error}")

else:
    st.info("Upload a CSV file to get started.")
