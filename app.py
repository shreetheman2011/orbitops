import streamlit as st
import pandas as pd
import numpy as np
import io, pickle, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Orbit-Ops | Find a Planet",
    layout="wide",
    page_icon="🚀"
)

# ---------------------- CONSTANTS & MAPPINGS ----------------------
FRIENDLY_TITLE = "Can We Find a New Planet?"
FRIENDLY_SUB = (
    "When a planet outside of our solar system passes in front of a star, the star gets a little dimmer. "
    "Our AI learned how to spot those dips. Enter a few numbers and see what the AI thinks!"
)

# Universal nickname map across datasets
NICKNAME_MAP = {
    # Orbit period
    "koi_period": "Orbit Length (days)",
    "pl_orbper": "Orbit Length (days)",
    "period": "Orbit Length (days)",

    # Transit duration
    "koi_duration": "Transit Time (hours)",
    "pl_trandur": "Transit Time (hours)",
    "duration": "Transit Time (hours)",

    # Depth
    "koi_depth": "Dip in Brightness (ppm)",
    "depth": "Dip in Brightness (ppm)",

    # Radius
    "koi_prad": "Planet Size(Relative to Earth)",
    "pl_rade": "Planet Size(Relative to Earth)",
    "pl_radj": "Planet Size (Jupiter radii)",
    "prad": "Planet Size(Relative to Earth)",

    # Temperature
    "koi_teq": "Temperature (K)",
    "pl_eqt": "Temperature (K)",
    "teq": "Temperature (K)",

    # Semi-major axis / distance
    "koi_sma": "Distance from Star (AU)",
    "pl_orbsmax": "Distance from Star (AU)",
    "sma": "Distance from Star (AU)"
}

# Minimal default features we try to use (in friendly order)
DEFAULT_FRIENDLY_FEATURES = [
    "Orbit Length (days)",
    "Transit Time (hours)",
    "Planet Size(Relative to Earth)",
    "Dip in Brightness (ppm)",
]

# ---------------------- UTILS ----------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path, comment="#", low_memory=False)


def find_label_column(df):
    for c in df.columns:
        if "disp" in c.lower() or "disposition" in c.lower():
            return c
    return None


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


def find_matching_columns(cols):
    """Return a dict mapping canonical column -> original dataset column name (if present)."""
    mapping = {}
    lower_to_col = {c.lower(): c for c in cols}
    for key in NICKNAME_MAP:
        if key.lower() in lower_to_col:
            mapping[key] = lower_to_col[key.lower()]
    return mapping


def inverse_nickname_map(cols):
    """Given a list of dataset columns, return friendly names (unique) and ordered list."""
    friendly = []
    used = set()
    # prefer keys in NICKNAME_MAP (dataset style)
    for k, v in NICKNAME_MAP.items():
        if k in cols and v not in used:
            friendly.append((k, v))
            used.add(v)
    # also accept direct columns that match friendly words
    for c in cols:
        if c in NICKNAME_MAP and NICKNAME_MAP[c] not in used:
            friendly.append((c, NICKNAME_MAP[c]))
            used.add(NICKNAME_MAP[c])
    return friendly

# ---------------------- LAYOUT: Super simple UI ----------------------
st.markdown(f"<div style='padding:20px;border-radius:10px;background:linear-gradient(90deg,#001f3f,#0074D9);'>"
            f"<h1 style='color:white;text-align:center;margin:0;padding:6px'>{FRIENDLY_TITLE}</h1>"
            f"</div>", unsafe_allow_html=True)

st.markdown(f"<p style='font-size:16px;margin-top:10px'>{FRIENDLY_SUB}</p>", unsafe_allow_html=True)

st.write("---")

# Sidebar controls (kept minimal but with an Advanced toggle)
st.sidebar.title("Pick how you want to predict")
use_kepler = st.sidebar.checkbox("Kepler (classic)", value=True)
use_tess = st.sidebar.checkbox("TESS (TOI)", value=False)
use_k2 = st.sidebar.checkbox("K2 (campaigns)", value=False)

uploaded_file = st.sidebar.file_uploader("Or upload your own CSV (optional)", type=["csv"])

advanced_mode = st.sidebar.checkbox("Show Scientist Mode (advanced)", value=False)

# Model hyperparams are hidden in advanced mode
if advanced_mode:
    st.sidebar.markdown("---")
    st.sidebar.header("Advanced: Model")
    model_choice = st.sidebar.selectbox("Model", ["Random Forest", "Logistic Regression"]) 
    test_size = st.sidebar.slider("Test split", 0.1, 0.5, 0.25, 0.05)
    if model_choice == "Random Forest":
        n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, 50)
        max_depth = st.sidebar.slider("max_depth", 0, 50, 10, 1)
    else:
        C = st.sidebar.slider("C (LogReg)", 0.01, 10.0, 1.0)
else:
    # sane defaults
    model_choice = "Random Forest"
    test_size = 0.25
    n_estimators = 200
    max_depth = 10
    C = 1.0

# ---------------------- Data loading ----------------------
selected_files = []
if use_kepler:
    selected_files.append(("Kepler (cumulative)", "cumulative_2025.09.29_18.45.43.csv"))
if use_tess:
    selected_files.append(("TESS (TOI)", "TOI_2025.09.29_18.46.06.csv"))
if use_k2:
    selected_files.append(("K2 (planet & candidate)", "k2pandc_2025.09.29_18.46.17.csv"))

# upload takes precedence and is treated as a single dataset
if uploaded_file is not None:
    try:
        df_up = pd.read_csv(uploaded_file, comment="#", low_memory=False)
        selected = [("Uploaded", None)]
        dfs = [df_up]
    except Exception as e:
        st.error("Could not read uploaded file: " + str(e))
        st.stop()
else:
    # load selected builtin files
    if not selected_files:
        st.info("Please pick at least one telescope data source from the left to proceed.")
        st.stop()
    dfs = []
    for name, path in selected_files:
        try:
            df = load_data(path)
            df["__source"] = name
            dfs.append(df)
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")

# combine
try:
    data = pd.concat(dfs, ignore_index=True, sort=False)
except Exception as e:
    st.error("Problem combining datasets: " + str(e))
    st.stop()

# find label column
label_col = find_label_column(data)
if not label_col:
    st.error("Sorry — the selected data does not include a column telling us if a signal was a planet or not. Upload a dataset that includes a disposition / disposition-like column.")
    st.stop()

data["label"] = data[label_col].apply(normalize_label)

# pick columns that match the nickname map or that are numeric
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
# lowercase map for matching
lower_map = {c.lower(): c for c in numeric_cols}

# Build a list of available friendly features based on dataset columns
available = []
for ds_key, friendly in NICKNAME_MAP.items():
    if ds_key in numeric_cols or ds_key.lower() in lower_map:
        colname = ds_key if ds_key in numeric_cols else lower_map.get(ds_key.lower())
        if colname and friendly not in [f for (_, f) in available]:
            available.append((colname, friendly))

# If nothing matched, fall back to some numeric columns (trim to 6)
if not available:
    for c in numeric_cols[:6]:
        available.append((c, c))

# Present simplified feature inputs (friendly names) — pick top 4
friendly_pairs = available[:4]

# Show a compact explanation of what each input means
st.markdown("### ✋ Step 1 — Tell me about your signal (just a few numbers)")
cols = st.columns(len(friendly_pairs))
user_inputs = {}
for i, (col, friendly) in enumerate(friendly_pairs):
    with cols[i]:
        st.write(f"**{friendly}**")
        tooltip = """Scientists use this for accurate measurements. If you are not sure, leave blank."""
        user_inputs[col] = st.text_input(label=friendly, help=tooltip)

st.markdown("""
<small style='color:gray'>Tip: You can leave fields empty to use the AI's average for that measurement.</small>
""", unsafe_allow_html=True)

# Teach / Train button
st.markdown("---")
if st.button("🚀 Teach the AI (Train Model)"):
    with st.spinner("Teaching the AI, this may take a few seconds..."):
        # Build dataset for training using whatever features we have
        features = [col for col, _ in friendly_pairs]
        # drop rows without label or without any of the features
        model_df = data[features + ["label"]].copy()
        # allow incomplete rows but drop rows where label is missing
        model_df = model_df.dropna(subset=["label"]) 
        # for simplicity drop rows that have all features NaN
        model_df = model_df.dropna(how="all", subset=features)
        # map labels
        label_map = {"false positive":0, "candidate":1, "confirmed":2}
        model_df = model_df[model_df["label"].isin(label_map.keys())]
        if model_df.empty:
            st.error("Not enough labeled data in the selected datasets to teach the AI.")
        else:
            # fill missing feature values with column mean (simple imputation)
            for f in features:
                if model_df[f].isna().all():
                    # if whole column is NA, fill with zeros
                    model_df[f] = model_df[f].fillna(0.0)
                else:
                    model_df[f] = model_df[f].fillna(model_df[f].mean())
            X = model_df[features].astype(float)
            y = model_df["label"].map(label_map).astype(int)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # train/test split
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, stratify=y)

            # choose model
            if model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=(max_depth if max_depth>0 else None), random_state=42)
            else:
                model = LogisticRegression(C=C, max_iter=2000)

            # simulate progress for kid-friendly feel
            progress = st.progress(0)
            for i in range(4):
                time.sleep(0.2)
                progress.progress((i+1)*25)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # store objects in session state for prediction
            st.session_state["orbit_model"] = model
            st.session_state["orbit_scaler"] = scaler
            st.session_state["orbit_features"] = features
            st.session_state["orbit_label_map"] = label_map

            st.balloons()
            st.success(f"All done! Our AI was right about planets **{acc*100:.1f}%** of the time on test signals.")
            # simple interpretation sentences
            st.write("- ✅ If you see 'Confirmed Planet' it means the AI thinks this signal looks very much like a real planet.")
            st.write("- 🕵️ If you see 'Candidate' it means the AI is unsure — more observations could help.")
            st.write("- ❌ If you see 'False Alarm' it means the AI thinks this signal is likely not a planet.")

# Prediction area
st.markdown("---")
st.markdown("### 🎯 Try a Prediction — what does the AI think about this signal?")
if "orbit_model" not in st.session_state:
    st.info("Teach the AI first by pressing 'Teach the AI (Train Model)'.")
else:
    model = st.session_state["orbit_model"]
    scaler = st.session_state["orbit_scaler"]
    features = st.session_state["orbit_features"]
    label_map = st.session_state["orbit_label_map"]

    # build input vector — use means for blanks
    input_vector = []
    for f, friendly in friendly_pairs:
        val = user_inputs.get(f, "")
        if val is None or val == "":
            # use mean from training data if available
            # (safe fallback: 0)
            input_vector.append(np.nan)
        else:
            try:
                input_vector.append(float(val))
            except:
                input_vector.append(np.nan)

    # allow button to predict
    if st.button("🔮 Predict" ):
        # impute missing with training means (from model_df used during training)
        # For simplicity, recompute training means from the stored scaler's mean via scaler.mean_
        # Note: StandardScaler stores mean_ for scaled features
        means = scaler.mean_
        imputed = [v if not np.isnan(v) else means[i] for i, v in enumerate(input_vector)]
        arr = np.array(imputed).reshape(1, -1)
        arr_scaled = scaler.transform(arr)
        pred = model.predict(arr_scaled)[0]
        probs = model.predict_proba(arr_scaled)[0]

        inv_map = {v:k for k,v in label_map.items()}
        label_str = inv_map[pred]

        # Big friendly card
        if label_str == "confirmed":
            st.markdown("<div style='padding:20px;border-radius:12px;background:#e6ffed;border:2px solid #2ecc71'>"
                        "<h2>✅ Confirmed Planet</h2>"
                        "<p>The AI is confident this looks like a real planet.</p>"
                        "</div>", unsafe_allow_html=True)
        elif label_str == "candidate":
            st.markdown("<div style='padding:20px;border-radius:12px;background:#fff8e6;border:2px solid #f0ad4e'>"
                        "<h2>🕵️ Candidate Planet</h2>"
                        "<p>The AI thinks this might be a planet but is not sure. More data could help.</p>"
                        "</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='padding:20px;border-radius:12px;background:#ffecec;border:2px solid #e74c3c'>"
                        "<h2>❌ False Alarm</h2>"
                        "<p>The AI thinks this signal is likely not a planet.</p>"
                        "</div>", unsafe_allow_html=True)

        # show simple probabilities (friendly)
        st.markdown("**AI confidence:**")
        prob_df = pd.DataFrame({"class": [inv_map[i] for i in range(len(probs))], "probability": probs})
        # reorder to meaningful order
        order = ["confirmed", "candidate", "false positive"]
        prob_df["class"] = pd.Categorical(prob_df["class"], categories=order, ordered=True)
        prob_df = prob_df.sort_values("class")
        st.table(prob_df.set_index("class"))

        # show a tiny explanation for kids
        st.write("**What this means (for everyone):**")
        st.write("- If the AI says **Confirmed Planet**, it's probably a real planet.")
        st.write("- If it says **Candidate**, telescopes should look again to be sure.")
        st.write("- If it says **False Alarm**, this signal probably came from something else (like noise).")

        if advanced_mode:
            st.markdown("---")
            st.subheader("Scientist Mode: Quick Debug Info")
            st.write("Accuracy on test set:" , f"{accuracy_score(y_test, y_pred)*100:.2f}%")
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))
            # feature importance if available
            try:
                if isinstance(model, RandomForestClassifier):
                    st.write("Feature importance:")
                    st.write(dict(zip(features, model.feature_importances_)))
                else:
                    coefs = np.mean(np.abs(model.coef_), axis=0)
                    st.write("Coefficient magnitudes:")
                    st.write(dict(zip(features, coefs)))
            except Exception as e:
                st.write("(could not show feature importances)")

# Footer
st.write("---")
st.markdown("<small style='color:gray'>Built for NASA Space Apps — Backend uses real exoplanet catalogs; frontend is simplified for learning and exploration.</small>", unsafe_allow_html=True)
