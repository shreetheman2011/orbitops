import streamlit as st
import pandas as pd
import numpy as np
import io, pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="OrbitOps - NASA Exoplanet Classifier",
    layout="wide",
    page_icon="https://assets.spaceappschallenge.org/media/images/Gemini_Generated_Image_4zb8yt4zb8y.2e16d0ba.fill-300x250.png"
)

# ---------------------- HEADER ----------------------
st.markdown(
    """
   <div style="background:linear-gradient(90deg,#001f3f,#0074D9);padding:1.2rem;border-radius:10px;display:flex;align-items:center;justify-content:center;">
        <img src="https://assets.spaceappschallenge.org/media/images/Gemini_Generated_Image_4zb8yt4zb8y.2e16d0ba.fill-300x250.png" 
             alt="Orbit Ops Logo" width="90" style="margin-right:15px; border-radius: 10px;" />
        <h1 style="color:white;margin:0;">Orbit Ops</h1>
    </div>
    <p style="color:#d9e6f2;text-align:center;">Interactive Machine Learning on Kepler, K2 & TESS Datasets</p>
    """,
    unsafe_allow_html=True
)

# ---------------------- SIDEBAR ----------------------
st.sidebar.image("https://assets.spaceappschallenge.org/media/images/Gemini_Generated_Image_4zb8yt4zb8y.2e16d0ba.fill-300x250.png", width=120)
st.sidebar.title("⚙️ Controls")

# Dataset selection
st.sidebar.header("Datasets")
builtin_files = {
    "Kepler (cumulative)": "cumulative_2025.09.29_18.45.43.csv",
    "TESS (TOI)": "TOI_2025.09.29_18.46.06.csv",
    "K2 (planet & candidate)": "k2pandc_2025.09.29_18.46.17.csv"
}
selected_files = [name for name in builtin_files if st.sidebar.checkbox(name)]

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Model options
st.sidebar.header("Model")
model_choice = st.sidebar.selectbox("Choose model", ["Random Forest", "Logistic Regression"])
test_size = st.sidebar.slider("Test split", 0.1, 0.5, 0.25, 0.05)

if model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, 50)
    max_depth = st.sidebar.slider("max_depth", 0, 50, 10, 1)
else:
    C = st.sidebar.slider("C (LogReg strength)", 0.01, 10.0, 1.0)

# ---------------------- LOAD DATA ----------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path, comment="#", low_memory=False)

dfs = []
for name in selected_files:
    df = load_data(builtin_files[name])
    df["__source"] = name
    dfs.append(df)

if uploaded_file:
    up = pd.read_csv(uploaded_file, comment="#", low_memory=False)
    up["__source"] = "uploaded"
    dfs.append(up)

if not dfs:
    st.warning("Please select or upload at least one dataset.")
    st.stop()

data = pd.concat(dfs, ignore_index=True, sort=False)

# ---------------------- LABEL PROCESSING ----------------------
def find_label_column(df):
    for c in df.columns:
        if "disp" in c.lower() or "disposition" in c.lower():
            return c
    return None

label_col = find_label_column(data)
if not label_col:
    st.error("No disposition/label column found.")
    st.stop()

# Normalize labels
def normalize_label(val):
    if pd.isna(val):
        return None
    s = str(val).lower()
    if "conf" in s or s in ["kp", "cp"]:
        return "confirmed"
    if "cand" in s or "pc" in s:
        return "candidate"
    if "false" in s or "fp" in s:
        return "false positive"
    return s

data["label"] = data[label_col].apply(normalize_label)

# ---------------------- FEATURE SELECTION ----------------------
numeric_cols = data.select_dtypes(include=[np.number]).columns
feature_candidates = [c for c in numeric_cols if any(x in c.lower() for x in ["period","duration","depth","rad","teq","mass","sma"])]

features = st.multiselect("Select features", feature_candidates, default=feature_candidates[:6])

if not features:
    st.error("Please select at least one numeric feature.")
    st.stop()

model_df = data[features + ["label"]].dropna()
label_map = {"false positive":0, "candidate":1, "confirmed":2}
model_df = model_df[model_df["label"].isin(label_map)]
X = model_df[features].astype(float)
y = model_df["label"].map(label_map)

# ---------------------- TRAIN MODEL ----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, stratify=y)

if model_choice == "Random Forest":
    kwargs = {"n_estimators": n_estimators, "random_state": 42}
    if max_depth > 0: kwargs["max_depth"] = max_depth
    model = RandomForestClassifier(**kwargs)
else:
    model = LogisticRegression(C=C, max_iter=2000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------------- METRICS ----------------------
acc = accuracy_score(y_test, y_pred)
col1, col2, col3 = st.columns(3)
col1.metric("Dataset size", f"{len(data):,}")
col2.metric("Training samples", f"{len(X_train):,}")
col3.metric("Accuracy", f"{acc*100:.2f}%")

# ---------------------- TABS ----------------------
st.subheader("🔍 Model Evaluation")
tabs = st.tabs(["Classification Report", "Confusion Matrix", "Feature Importance", "PCA Visualization"])

with tabs[0]:
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

with tabs[1]:
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

with tabs[2]:
    if model_choice == "Random Forest":
        st.bar_chart(pd.Series(model.feature_importances_, index=features))
    else:
        coefs = np.mean(np.abs(model.coef_), axis=0)  # average across classes
        st.bar_chart(pd.Series(coefs, index=features))
with tabs[3]:
    pca = PCA(n_components=2)
    proj = pca.fit_transform(X_scaled)
    proj_df = pd.DataFrame(proj, columns=["PC1","PC2"])
    proj_df["label"] = y
    fig, ax = plt.subplots()
    for lbl, col in zip(label_map.keys(), ["red","orange","green"]):
        subset = proj_df[proj_df["label"]==label_map[lbl]]
        ax.scatter(subset["PC1"], subset["PC2"], label=lbl, alpha=0.6)
    ax.legend()
    st.pyplot(fig)

# ---------------------- PREDICTION ----------------------
st.subheader("🎯 Try a Prediction")
input_vals = {}
cols = st.columns(len(features))
for i, f in enumerate(features):
    with cols[i % len(cols)]:
        input_vals[f] = st.text_input(f, "")

if st.button("Predict"):
    try:
        arr = np.array([float(input_vals[f]) for f in features]).reshape(1,-1)
        arr_scaled = scaler.transform(arr)
        pred = model.predict(arr_scaled)[0]
        st.success(f"Predicted class: {list(label_map.keys())[list(label_map.values()).index(pred)]}")
        st.write("Probabilities:", model.predict_proba(arr_scaled))
    except:
        st.error("Please enter valid numeric values for all features.")