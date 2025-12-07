import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Disease Prediction", layout="wide")

DATA_DIR = "datasets"
FILES = {
    "diabetes": "diabetes.csv",
    "breast_cancer": "BreastCancer.csv",
    "heartdisease": "heartdisease.csv"
}

def load_csv_safe(path):
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    def strip_if_obj(col):
        if col.dtype == object:
            return col.str.strip()
        return col
    df = df.apply(strip_if_obj)
    df.replace({"?": np.nan, "": np.nan, "NA": np.nan, "na": np.nan}, inplace=True)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass
    return df

def prepare_diabetes(df):
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.fillna(df.median(numeric_only=True))
    return df

def prepare_breast(df):
    df.columns = [c.strip().replace(" ","") for c in df.columns]
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    if "Class" in df.columns:
        cls = df["Class"].unique()
        if set(cls.astype(str)).issubset({"2","4","2.0","4.0"}):
            df["Class"] = pd.to_numeric(df["Class"], errors='coerce').map({2:0,4:1})
        elif set(cls.astype(str)).issubset({"B","M","b","m","B "," M"}):
            df["Class"] = df["Class"].astype(str).str.upper().map({"B":0,"M":1})
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.fillna(df.median(numeric_only=True))
    return df

def prepare_heart(df):
    df.columns = [c.strip() for c in df.columns]
    if "Target" in df.columns and "target" not in df.columns:
        df.rename(columns={"Target":"target"}, inplace=True)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.fillna(df.median(numeric_only=True))
    return df

DISPLAY_TO_KEY = {
    "Diabetes": "diabetes",
    "Breast Cancer": "breast_cancer",
    "Heart Disease": "heartdisease"
}
dataset_display = st.sidebar.selectbox("Choose disease dataset", list(DISPLAY_TO_KEY.keys()))
dataset_choice = DISPLAY_TO_KEY[dataset_display]

file_path = os.path.join(DATA_DIR, FILES[dataset_choice])
if not os.path.isfile(file_path):
    st.error(f"Dataset file not found: {file_path}\nPut CSV into the 'datasets' folder with exact filename.")
    st.stop()

raw_df = load_csv_safe(file_path)

if dataset_choice == "diabetes":
    df = prepare_diabetes(raw_df)
    target_col = "Outcome"
elif dataset_choice == "breast_cancer":
    df = prepare_breast(raw_df)
    target_col = "Class" if "Class" in raw_df.columns else "diagnosis"
elif dataset_choice == "heartdisease":
    df = prepare_heart(raw_df)
    target_col = "target" if "target" in df.columns else "Target"
else:
    st.error("Unsupported dataset choice.")
    st.stop()

if target_col not in df.columns:
    st.error(f"Target column '{target_col}' not found in {file_path}. Columns: {df.columns.tolist()}")
    st.stop()

st.title("Disease Prediction App (Diabetes, Breast Cancer, Heart Disease)")
st.write("Select dataset, provide feature values in the sidebar, then click Predict.")
st.subheader("Dataset sample")
st.dataframe(df.head(6))

st.sidebar.markdown("## Input features")
features = [c for c in df.columns if c != target_col]

input_vals = {}
for col in features:
    series = df[col]
    if pd.api.types.is_numeric_dtype(series):
        unique_vals = series.dropna().unique()
        if 2 <= len(unique_vals) <= 10 and np.all(np.equal(np.mod(unique_vals, 1), 0)):
            opt = sorted([int(v) for v in unique_vals])
            input_vals[col] = st.sidebar.selectbox(f"{col}", options=opt, index=0)
        else:
            min_v = float(series.min())
            max_v = float(series.max())
            mean_v = float(series.mean())
            if min_v == max_v:
                input_vals[col] = st.sidebar.number_input(f"{col}", value=mean_v)
            else:
                input_vals[col] = st.sidebar.number_input(f"{col}", min_value=min_v, max_value=max_v, value=mean_v)
    else:
        vals = series.dropna().astype(str).value_counts().index.tolist()
        if vals:
            input_vals[col] = st.sidebar.selectbox(f"{col}", options=vals, index=0)
        else:
            input_vals[col] = st.sidebar.text_input(f"{col}", value="")

@st.cache_resource
def get_trained_model():
    dd = df.copy()
    X = dd[features].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype('category').cat.codes
    y = dd[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, X.columns.tolist(), target_col

if st.sidebar.button("Predict"):
    try:
        model, scaler, feat_cols, tgt = get_trained_model()
        inp = pd.DataFrame([input_vals], columns=features)
        for c in inp.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                inp[c] = inp[c].astype(str).astype('category').cat.set_categories(df[c].astype(str).astype('category').cat.categories)
                inp[c] = inp[c].astype('category').cat.codes
            else:
                inp[c] = pd.to_numeric(inp[c], errors='coerce')
        inp = inp.fillna(df[features].median(numeric_only=True))
        inp_scaled = scaler.transform(inp)
        pred = model.predict(inp_scaled)[0]
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(inp_scaled)[0].max()
        st.subheader("Prediction result")
        if dataset_choice == "diabetes":
            label_text = "Has Diabetes" if int(pred) == 1 else "No Diabetes"
        elif dataset_choice == "breast_cancer":
            label_text = "Malignant (Breast Cancer)" if int(pred) == 1 else "Benign"
        else:
            label_text = "Has Heart Disease" if int(pred) == 1 else "No Heart Disease"
        st.success(f"{label_text}")
        if prob is not None:
            st.info(f"Model confidence (max class prob): {prob:.3f}")
        st.write("### Input values used for prediction")
        st.json(input_vals)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)
