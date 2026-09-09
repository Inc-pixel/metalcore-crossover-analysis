import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Metalcore Crossover Analysis", layout="wide")
st.title("Metalcore Crossover Analysis")
st.markdown("*When a metalcore song goes mainstream, what musical features change and which ones predict it best?*")
st.markdown("**Roy — BASIS Shavano, Summer 2026**")

# --- Load & encode ---
df = pd.read_csv("metalcore_dataset_hand_annotated.csv")

df["breakdown_num"]  = df["breakdown_density"].map({"3+": 2, "1-2": 1, "0": 0})
df["vocal_num"]      = df["vocal_style"].map({"screaming_heavy": 2, "mixed": 1, "clean_only": 0})
df["electronic_num"] = df["electronic_production"].map({"none": 0, "subtle": 1, "prominent": 2})
df["complexity_num"] = df["guitar_complexity"].map({"high": 2, "medium": 1, "low": 0})
df["tuning_num"]     = (df["dropped_tuning"] == "yes").astype(int)

df["heaviness_index"]       = df["breakdown_num"] + df["vocal_num"] + df["complexity_num"] + df["tuning_num"]
df["polish_index"]          = df["electronic_num"]
df["heaviness_minus_polish"] = df["heaviness_index"] - df["polish_index"]
df["vocal_x_electronic"]    = df["vocal_num"] * df["electronic_num"]
df["softened_lyrics"]       = df["lyrical_theme"].isin(["personal", "pop_romance"]).astype(int)
df["log_streams"]           = np.log10(df["popularity_metric"] + 1)

ERA_ORDER = ["underground", "crossover", "mainstream"]

# --- Sidebar filter ---
st.sidebar.header("Filter")
all_bands = sorted(df["band"].unique())
selected_bands = st.sidebar.multiselect("Bands to include", all_bands, default=all_bands)
view = df[df["band"].isin(selected_bands)]

# ──────────────────────────────────────────────────────────────
# SECTION 1: ERA SUMMARY
# ──────────────────────────────────────────────────────────────
st.header("1. Feature shift across eras")

era_summary = (
    view.groupby("era")[["heaviness_index", "polish_index", "softened_lyrics", "log_streams"]]
    .mean()
    .reindex(ERA_ORDER)
    .dropna()
)

col1, col2, col3 = st.columns(3)

with col1:
    fig, ax = plt.subplots(figsize=(5, 4))
    era_summary[["heaviness_index", "polish_index"]].plot(
        kind="bar", ax=ax, color=["firebrick", "steelblue"], rot=0
    )
    ax.set_title("Heaviness vs. Production Polish")
    ax.set_ylabel("Average Index Score")
    ax.legend(["Heaviness Index", "Polish Index"])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(5, 4))
    era_summary["softened_lyrics"].plot(kind="bar", ax=ax, color="mediumpurple", rot=0)
    ax.set_title("Songs with Softened Lyrics")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col3:
    fig, ax = plt.subplots(figsize=(5, 4))
    era_summary["log_streams"].plot(kind="bar", ax=ax, color="seagreen", rot=0)
    ax.set_title("Avg Spotify Streams (log₁₀)")
    ax.set_ylabel("log₁₀(streams)")
    ax.set_ylim(0, 9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with st.expander("Raw era averages table"):
    st.dataframe(era_summary.round(2))

# ──────────────────────────────────────────────────────────────
# SECTION 2: BAND TRAJECTORY
# ──────────────────────────────────────────────────────────────
st.header("2. Band trajectory")

selected_band = st.selectbox("Pick a band", sorted(df["band"].unique()))
era_rank = {e: i for i, e in enumerate(ERA_ORDER)}
band_df = (
    df[df["band"] == selected_band]
    .sort_values("era", key=lambda x: x.map(era_rank))
    .reset_index(drop=True)
)

st.dataframe(
    band_df[[
        "song", "album", "era",
        "heaviness_index", "polish_index",
        "softened_lyrics", "popularity_metric", "metric_source"
    ]]
)

if len(band_df) >= 2:
    fig, ax = plt.subplots(figsize=(7, 3))
    eras = band_df["era"].tolist()
    ax.plot(eras, band_df["heaviness_index"], marker="o", color="firebrick", label="Heaviness")
    ax.plot(eras, band_df["polish_index"], marker="o", color="steelblue", label="Polish")
    ax.set_title(f"{selected_band}: Heaviness vs. Polish")
    ax.set_ylabel("Index score")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ──────────────────────────────────────────────────────────────
# SECTION 3: CORRELATION HEATMAP
# ──────────────────────────────────────────────────────────────
st.header("3. Feature correlations")

numeric_cols = [
    "breakdown_num", "vocal_num", "electronic_num",
    "complexity_num", "tuning_num",
    "heaviness_index", "polish_index", "log_streams"
]
corr = view[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax)
ax.set_title("Correlation Matrix")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ──────────────────────────────────────────────────────────────
# SECTION 4: LOGISTIC REGRESSION FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────────
st.header("4. Which features predict mainstream status?")

era_map = {"underground": 0, "crossover": 1, "mainstream": 1}
df["era_binary"] = df["era"].map(era_map)
X = df[["breakdown_num", "vocal_num", "electronic_num", "complexity_num", "tuning_num"]]
y = df["era_binary"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

coef_df = pd.DataFrame({
    "Feature": ["Breakdown Density", "Vocal Style", "Electronic Production",
                "Guitar Complexity", "Dropped Tuning"],
    "Coefficient": model.coef_[0]
}).sort_values("Coefficient")

fig, ax = plt.subplots(figsize=(7, 4))
colors = ["firebrick" if c < 0 else "steelblue" for c in coef_df["Coefficient"]]
ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Logistic Regression Coefficients\n(positive = predicts mainstream, negative = predicts underground)")
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.caption("N = 46 songs across 20 bands, hand-annotated. Popularity metric: Spotify streams (kworb.net).")
