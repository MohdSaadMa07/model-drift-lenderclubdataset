import streamlit as st
import pandas as pd
import joblib

# Load the saved pipeline
pipeline = joblib.load("models/loan_rf_pipeline.pkl")


features = ['loan_amnt', 'installment', 'int_rate', 'annual_inc', 'emp_length',
            'term', 'grade', 'home_ownership', 'verification_status']
num_cols = ['loan_amnt', 'installment', 'int_rate', 'annual_inc', 'emp_length']
cat_cols = ['term', 'grade', 'home_ownership', 'verification_status']

# Data preparation function
def prepare_new_data(df):
    df = df.copy()
    for col in features:
        if col not in df.columns:
            df[col] = 0 if col in num_cols else 'Unknown'
    return df[features]

st.title("Loan Default Prediction & Drift Analysis")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df_prepared = prepare_new_data(df)
    df['predicted_default'] = df['predicted_default'].map(
    {0: 'No Default', 1: 'Default'}
)

    # Predict
    df['predicted_default'] = pipeline.predict(df_prepared)

    st.write("Predictions:")
    st.dataframe(df[['predicted_default']].head())
st.subheader("Feature Drift Check (Mean Shift)")

train_means = pipeline.named_steps['preprocessor']\
    .transformers_[0][1].mean_

current_means = df_prepared[num_cols].mean().values

drift = abs(current_means - train_means)

drift_df = pd.DataFrame({
    'feature': num_cols,
    'mean_shift': drift
})

st.dataframe(drift_df)

    # Optional: yearly accuracy if target exists
    if 'issue_d' in df.columns and 'target' in df.columns:
        df['issue_d'] = pd.to_datetime(df['issue_d'])
        df['year'] = df['issue_d'].dt.year
        yearly_acc = df.groupby('year').apply(lambda x: (x['predicted_default'] == x['target']).mean())
        st.write("Yearly Accuracy:")
        st.bar_chart(yearly_acc)
