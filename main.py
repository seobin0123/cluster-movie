import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")
st.title("🎬 영화 유형 나누기")
st.caption("영화의 흥행 규모와 롱런 특성을 바탕으로 비슷한 영화끼리 묶어 봅니다.")

FEATURES = {
    "스크린 수(로그)": "log_first_scrn",
    "누적 관객(로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "longrun_index",
}
FEATURE_COLUMNS = list(FEATURES.values())
FEATURE_LABELS = list(FEATURES.keys())

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    required = [
        "movieCd", "movieNm", "openDt", "genre", "nation", "first_scrn",
        "first_show", "first_date", "peak", "first_week_audi",
        "total_audi", "days_in_top10"
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("데이터에 필요한 열이 없습니다: " + ", ".join(missing))

    for col in ["first_scrn", "first_week_audi", "total_audi", "days_in_top10"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 네 가지 군집 속성에 필요한 값이 모두 있고, 첫 주 관객이 양수인 영화만 사용
    df = df.dropna(subset=["first_scrn", "total_audi", "days_in_top10", "first_week_audi"]).copy()
    df = df[(df["first_scrn"] > 0) & (df["total_audi"] >= 0) &
            (df["days_in_top10"] >= 0) & (df["first_week_audi"] > 0)].copy()

    df["log_first_scrn"] = df["first_scrn"].apply(lambda x: __import__("math").log10(x))
    df["log_total_audi"] = df["total_audi"].apply(lambda x: __import__("math").log10(x + 1))
    df["longrun_index"] = (df["total_audi"] / df["first_week_audi"]).clip(upper=20)
    return df

try:
    raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오거나 처리하지 못했습니다: {e}")
    st.stop()

total_count = len(raw)
st.write(f"**전체 편수:** {total_count:,}편　|　**묶은 편수:** 계산 전")

st.subheader("1. 묶는 데 사용할 속성 선택")
selected_labels = st.multiselect(
    "군집화에 사용할 속성 (2개 이상 선택)",
    options=FEATURE_LABELS,
    default=FEATURE_LABELS,
)
if len(selected_labels) < 2:
    st.warning("군집화를 위해 속성을 두 개 이상 선택해 주세요.")
    st.stop()

selected_features = [FEATURES[label] for label in selected_labels]
valid = raw.dropna(subset=selected_features).copy()
if len(valid) < 3:
    st.error("유효한 영화가 3편 미만이라 세 그룹으로 나눌 수 없습니다.")
    st.stop()

# 선택한 속성만 표준화하고, 고정된 난수와 n_init으로 결과를 재현한다.
X = valid[selected_features].astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
valid["cluster_raw"] = kmeans.fit_predict(X_scaled)

# 누적 관객 평균이 큰 군집부터 ㉮, ㉯, ㉰를 부여한다.
cluster_order = (
    valid.groupby("cluster_raw")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
cluster_names = {cluster_id: ["㉮", "㉯", "㉰"][i] for i, cluster_id in enumerate(cluster_order)}
valid["유형"] = valid["cluster_raw"].map(cluster_names)
cluster_color_order = ["㉮", "㉯", "㉰"]

st.write(f"**전체 편수:** {total_count:,}편　|　**묶은 편수:** {len(valid):,}편")

st.subheader("2. 2차원 산점도")
c1, c2 = st.columns(2)
with c1:
    x_label = st.selectbox("가로축 속성", selected_labels, index=0, key="scatter_x")
with c2:
    y_default = 1 if len(selected_labels) > 1 else 0
    y_label = st.selectbox("세로축 속성", selected_labels, index=y_default, key="scatter_y")

fig2 = px.scatter(
    valid,
    x=FEATURES[x_label],
    y=FEATURES[y_label],
    color="유형",
    category_orders={"유형": cluster_color_order},
    hover_name="movieNm",
    hover_data={"유형": True, FEATURES[x_label]: ":.3f", FEATURES[y_label]: ":.3f"},
    labels={FEATURES[x_label]: x_label, FEATURES[y_label]: y_label, "유형": "영화 유형"},
)
fig2.update_traces(marker={"size": 8, "opacity": 0.75})
fig2.update_layout(legend_title_text="영화 유형", margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.subheader("3. 3차원 산점도")
if len(selected_labels) < 3:
    st.info("3차원 산점도를 보려면 군집화 속성을 세 개 이상 선택해 주세요.")
else:
    c3, c4, c5 = st.columns(3)
    with c3:
        x3_label = st.selectbox("X축 속성", selected_labels, index=0, key="x3")
    with c4:
        y3_label = st.selectbox("Y축 속성", selected_labels, index=1, key="y3")
    with c5:
        z3_label = st.selectbox("Z축 속성", selected_labels, index=2, key="z3")

    fig3 = px.scatter_3d(
        valid,
        x=FEATURES[x3_label],
        y=FEATURES[y3_label],
        z=FEATURES[z3_label],
        color="유형",
        category_orders={"유형": cluster_color_order},
        hover_name="movieNm",
        hover_data={"유형": True},
        labels={
            FEATURES[x3_label]: x3_label,
            FEATURES[y3_label]: y3_label,
            FEATURES[z3_label]: z3_label,
            "유형": "영화 유형",
        },
    )
    fig3.update_traces(marker={"size": 3, "opacity": 0.8})
    fig3.update_layout(legend_title_text="영화 유형", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("4. 유형별 편수와 네 속성의 평균")
summary = (
    valid.groupby("유형")
    .agg(
        편수=("movieNm", "size"),
        스크린_수=("first_scrn", "mean"),
        누적_관객=("total_audi", "mean"),
        10위권_일수=("days_in_top10", "mean"),
        롱런_지수=("longrun_index", "mean"),
    )
    .reindex(cluster_color_order)
    .reset_index()
)
summary = summary.rename(columns={
    "유형": "영화 유형",
    "스크린_수": "스크린 수 평균",
    "누적_관객": "누적 관객 평균",
    "10위권_일수": "10위권 일수 평균",
    "롱런_지수": "롱런 지수 평균",
})
summary["편수"] = summary["편수"].map(lambda x: f"{x:,}")
for col in ["스크린 수 평균", "누적 관객 평균", "10위권 일수 평균", "롱런 지수 평균"]:
    summary[col] = summary[col].map(lambda x: f"{x:,.2f}")
st.dataframe(summary, use_container_width=True, hide_index=True)

st.subheader("5. 유형별 누적 관객 상위 영화 5편")
for cluster_name in cluster_color_order:
    group = valid[valid["유형"] == cluster_name].sort_values("total_audi", ascending=False).head(5)
    with st.expander(f"{cluster_name} 유형 — 상위 {len(group)}편", expanded=True):
        if group.empty:
            st.write("해당 유형에 영화가 없습니다.")
        else:
            top = group[["movieNm", "total_audi", "first_week_audi", "days_in_top10"]].copy()
            top = top.rename(columns={
                "movieNm": "영화 제목",
                "total_audi": "누적 관객",
                "first_week_audi": "첫 주 관객",
                "days_in_top10": "10위권 일수",
            })
            for col in ["누적 관객", "첫 주 관객", "10위권 일수"]:
                top[col] = top[col].map(lambda x: f"{x:,.0f}")
            st.dataframe(top, use_container_width=True, hide_index=True)

