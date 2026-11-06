
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split

from regression_toolkit.data import validate_dataset
from regression_toolkit.evaluation import evaluate_models


# Configure the browser page
st.set_page_config(
    page_title="Regression Model Comparison",
    page_icon="📊",
    layout="wide",
)

if "results" not in st.session_state:
    st.session_state.results = None

if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None

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

        # Clear previous results when a new dataset is uploaded
        if st.session_state.get("uploaded_file_name") != uploaded_file.name:
            st.session_state.results = None
            st.session_state.prediction_results = None
            st.session_state.uploaded_file_name = uploaded_file.name

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

                st.session_state.results = results
                st.session_state.prediction_results = prediction_results

            
    except Exception as error:
        st.error(f"Something went wrong: {error}")

else:
    st.info("Upload a CSV file to get started.")

if st.session_state.results is not None:

    results = st.session_state.results
    prediction_results = st.session_state.prediction_results

    st.subheader("Model Performance")

    st.dataframe(
        results,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Download Results")

    # Download model comparison metrics
    metrics_csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Model Metrics",
        data=metrics_csv,
        file_name="model_comparison_results.csv",
        mime="text/csv",
    )

    # Download actual and predicted values
    predictions_csv = prediction_results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Predictions",
        data=predictions_csv,
        file_name="model_predictions.csv",
        mime="text/csv",
    )

    best_model = results.iloc[0]

    st.subheader("Best Performing Model")

    st.metric(
        label="Best Model (lowest MAE)",
        value=best_model["Model"],
        delta=f"MAE: {best_model['MAE']:.3f}",
        delta_color="off",
    )

    # Your bar chart goes here
    st.subheader("Model Performance Comparison")

    fig = px.bar(
        results,
        x="Model",
        y="MAE",
        color="Model",
        title="Mean Absolute Error by Model",
        labels={
            "MAE": "Mean Absolute Error",
            "Model": "Regression Model",
        },
    )

    fig.update_layout(
        showlegend=False,
        yaxis_title="Mean Absolute Error",
        xaxis_title="",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Your actual vs. predicted chart goes here
    st.subheader("Actual vs. Predicted")
    
    selected_model = st.selectbox(
        "Choose a model",
        options=results["Model"].tolist(),
    )

    # Get predictions for the selected model
    plot_data = prediction_results[["actual", selected_model]].copy()

    plot_data = plot_data.rename(
        columns={
            "actual": "Actual",
            selected_model: "Predicted",
        }
    )

    fig = px.scatter(
        plot_data,
        x="Actual",
        y="Predicted",
        title=f"Actual vs. Predicted: {selected_model}",
        labels={
            "Actual": "Actual Values",
            "Predicted": "Predicted Values",
        },
    )

    # Add a perfect-prediction reference line
    minimum = min(
        plot_data["Actual"].min(),
        plot_data["Predicted"].min(),
    )

    maximum = max(
        plot_data["Actual"].max(),
        plot_data["Predicted"].max(),
    )

    fig.add_shape(
        type="line",
        x0=minimum,
        y0=minimum,
        x1=maximum,
        y1=maximum,
        line=dict(dash="dash"),
    )

    st.plotly_chart(fig, use_container_width=True)


    # Residual analysis
    st.subheader("Residual Analysis")

    # Calculate residuals for the selected model
    residual_data = plot_data.copy()

    residual_data["Residual"] = (
        residual_data["Actual"] - residual_data["Predicted"]
    )

    # Residuals vs. predicted values
    st.subheader("Residuals vs. Predicted Values")

    residual_fig = px.scatter(
        residual_data,
        x="Predicted",
        y="Residual",
        title=f"Residual Analysis: {selected_model}",
        labels={
            "Predicted": "Predicted Values",
            "Residual": "Residual (Actual - Predicted)",
        },
    )

    # Add a zero-reference line
    residual_fig.add_hline(
        y=0,
        line_dash="dash",
        annotation_text="Zero Error",
    )

    st.plotly_chart(
        residual_fig,
        use_container_width=True,
    )

    # Residual distribution
    st.subheader("Residual Distribution")

    histogram_fig = px.histogram(
        residual_data,
        x="Residual",
        nbins=20,
        title=f"Distribution of Residuals: {selected_model}",
        labels={
            "Residual": "Prediction Error",
            "count": "Number of Predictions",
        },
    )

    histogram_fig.add_vline(
        x=0,
        line_dash="dash",
        annotation_text="Zero Error",
    )

    st.plotly_chart(
        histogram_fig,
        use_container_width=True,
    )

    
