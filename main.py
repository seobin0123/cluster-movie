
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==================================================
# 1. 페이지 설정
# ==================================================
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 영화 유형 나누기")
st.write(
    "영화의 여러 특성을 분석하여 비슷한 영화끼리 묶어 봅니다. "
    "군집 수와 분석에 사용할 속성을 직접 선택할 수 있습니다."
)


# ==================================================
# 2. 데이터 불러오기
# ==================================================
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "it-teacher-go/data-share-/main/kobis_movies.csv"
)

CLUSTER_SYMBOLS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]

FEATURE_LABELS = {
    "log_first_scrn": "개봉 스크린 수 (로그 변환)",
    "log_total_audi": "누적 관객 수 (로그 변환)",
    "days_in_top10": "10위권에 머문 일수",
    "long_run_index": "롱런 지수",
}


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "movieCd",
        "movieNm",
        "openDt",
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            "데이터에 필요한 열이 없습니다: "
            + ", ".join(missing)
        )

    numeric_columns = [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
        "first_show",
        "peak",
    ]

    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce",
            )

    # 로그 변환은 양수인 값에만 적용한다.
    data["log_first_scrn"] = np.nan
    mask_screen = (
        data["first_scrn"].notna()
        & (data["first_scrn"] > 0)
    )
    data.loc[mask_screen, "log_first_scrn"] = np.log10(
        data.loc[mask_screen, "first_scrn"]
    )

    data["log_total_audi"] = np.nan
    mask_audience = (
        data["total_audi"].notna()
        & (data["total_audi"] > 0)
    )
    data.loc[mask_audience, "log_total_audi"] = np.log10(
        data.loc[mask_audience, "total_audi"]
    )

    # 롱런 지수 = 누적 관객 / 첫 주 관객
    data["long_run_index"] = np.nan

    mask_long = (
        data["first_week_audi"].notna()
        & (data["first_week_audi"] > 0)
        & data["total_audi"].notna()
        & (data["total_audi"] >= 0)
    )

    data.loc[mask_long, "long_run_index"] = (
        data.loc[mask_long, "total_audi"]
        / data.loc[mask_long, "first_week_audi"]
    ).clip(upper=20)

    return data


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


st.success(f"데이터를 불러왔습니다. 전체 영화 수: {len(df):,}편")


# ==================================================
# 3. 분석 속성 선택
# ==================================================
st.subheader("1. 분석 설정")

st.write("군집화에 사용할 속성을 두 개 이상 선택하세요.")

default_features = [
    "log_first_scrn",
    "log_total_audi",
    "days_in_top10",
    "long_run_index",
]

feature_cols = st.columns(2)
selected_features = []

for i, feature in enumerate(FEATURE_LABELS):
    with feature_cols[i % 2]:
        if st.checkbox(
            FEATURE_LABELS[feature],
            value=feature in default_features,
            key=f"feature_{feature}",
        ):
            selected_features.append(feature)

if len(selected_features) < 2:
    st.warning("군집화를 위해 속성을 두 개 이상 선택해 주세요.")
    st.stop()


# ==================================================
# 4. 군집 수 선택
# ==================================================
cluster_count = st.selectbox(
    "묶음 수",
    options=list(range(2, 8)),
    index=1,
    format_func=lambda x: f"{x}개",
    key="cluster_count",
)

st.caption(
    "묶음 수는 2~7개 중 선택할 수 있습니다. "
    "기본값은 3개입니다."
)


# ==================================================
# 5. 분석에 사용할 데이터 정리
# ==================================================
valid_df = df.copy()

for col in selected_features:
    valid_df[col] = pd.to_numeric(
        valid_df[col],
        errors="coerce",
    )

valid_df = valid_df.replace(
    [np.inf, -np.inf],
    np.nan,
)

valid_df = valid_df.dropna(
    subset=selected_features + ["total_audi"]
).copy()

valid_df = valid_df[
    np.isfinite(valid_df["total_audi"])
].copy()

if len(valid_df) < 8:
    st.error(
        "분석에 사용할 수 있는 영화가 너무 적습니다. "
        "속성을 바꾸거나 원본 데이터를 확인해 주세요."
    )
    st.stop()

if valid_df[selected_features].nunique().min() == 0:
    st.error("선택한 속성 중 유효한 값이 없는 속성이 있습니다.")
    st.stop()

# 선택한 속성을 표준화한다.
scaler = StandardScaler()
X = scaler.fit_transform(valid_df[selected_features])

if not np.isfinite(X).all():
    st.error(
        "표준화 과정에서 유효하지 않은 값이 발생했습니다. "
        "다른 속성을 선택해 주세요."
    )
    st.stop()


# ==================================================
# 6. 군집 수별 거리 제곱 합 계산
# ==================================================
st.subheader("2. 군집 수에 따른 거리 제곱 합")

# KMeans의 inertia_는 군집 중심과 각 데이터 점 사이의
# 유클리드 거리 제곱을 모두 더한 값이다.
inertia_rows = []

for k in range(1, 8):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    model.fit(X)

    inertia_rows.append({
        "묶음 수": k,
        "거리 제곱 합": float(model.inertia_),
    })

inertia_df = pd.DataFrame(inertia_rows)


# ==================================================
# 7. 거리 제곱 합 그래프
# ==================================================
fig_inertia = go.Figure()

fig_inertia.add_trace(
    go.Scatter(
        x=inertia_df["묶음 수"],
        y=inertia_df["거리 제곱 합"],
        mode="lines+markers",
        name="거리 제곱 합",
        hovertemplate=(
            "묶음 수: %{x}개"
            "<br>거리 제곱 합: %{y:,.2f}"
            "<extra></extra>"
        ),
    )
)

fig_inertia.add_vline(
    x=cluster_count,
    line_dash="dash",
    line_width=2,
    annotation_text=f"현재 선택: {cluster_count}개",
    annotation_position="top right",
)

fig_inertia.update_layout(
    title="군집 수에 따른 거리 제곱 합",
    xaxis_title="묶음 수",
    yaxis_title="거리 제곱 합",
    xaxis=dict(
        tickmode="linear",
        dtick=1,
        range=[0.5, 7.5],
    ),
    hovermode="x unified",
)

st.plotly_chart(
    fig_inertia,
    use_container_width=True,
)

st.caption(
    "값이 작을수록 각 영화가 소속 군집의 중심에 가깝다는 뜻입니다. "
    "군집 수가 늘면 값이 감소하므로, 감소 폭이 완만해지는 지점도 살펴보세요."
)


# ==================================================
# 8. 거리 제곱 합 표
# ==================================================
st.subheader("3. 군집 수별 거리 제곱 합 비교")

inertia_table = inertia_df.copy()

inertia_table["직전 값에서 감소한 양"] = (
    inertia_table["거리 제곱 합"].shift(1)
    - inertia_table["거리 제곱 합"]
)

inertia_table["묶음 수"] = (
    inertia_table["묶음 수"].astype(str) + "개"
)

inertia_table["거리 제곱 합"] = (
    inertia_table["거리 제곱 합"].map(
        lambda x: f"{x:,.2f}"
    )
)

inertia_table["직전 값에서 감소한 양"] = (
    inertia_table["직전 값에서 감소한 양"].map(
        lambda x: "" if pd.isna(x) else f"{x:,.2f}"
    )
)

st.dataframe(
    inertia_table,
    hide_index=True,
    use_container_width=True,
)


# ==================================================
# 9. 선택한 군집 수로 K-means 실행
# ==================================================
st.subheader("4. 영화 군집화")

kmeans = KMeans(
    n_clusters=cluster_count,
    random_state=42,
    n_init=10,
)

cluster_raw = kmeans.fit_predict(X)

valid_df["cluster_raw"] = cluster_raw

# 누적 관객 평균이 높은 군집부터 기호를 부여한다.
cluster_order = (
    valid_df.groupby("cluster_raw")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

cluster_labels = {
    cluster_id: CLUSTER_SYMBOLS[i]
    for i, cluster_id in enumerate(cluster_order)
}

valid_df["묶음"] = (
    valid_df["cluster_raw"].map(cluster_labels)
)

valid_df["영화 제목"] = (
    valid_df["movieNm"]
    .fillna("(제목 없음)")
    .astype(str)
)

active_symbols = CLUSTER_SYMBOLS[:cluster_count]


# ==================================================
# 10. 실루엣 점수
# ==================================================
st.subheader("5. 실루엣 점수")

actual_cluster_count = len(
    np.unique(valid_df["cluster_raw"])
)

if 2 <= actual_cluster_count < len(valid_df):
    try:
        score = silhouette_score(
            X,
            valid_df["cluster_raw"],
        )

        st.markdown(
            f"**현재 {cluster_count}개 묶음의 실루엣 점수: "
            f"{score:.3f}**"
        )

        st.caption(
            "실루엣 점수는 -1부터 1까지입니다. "
            "1에 가까울수록 군집이 뚜렷하게 구분됩니다. "
            "0에 가까우면 군집이 겹칠 수 있고, "
            "음수이면 일부 영화가 다른 군집에 더 가까울 수 있습니다."
        )

    except Exception as e:
        st.warning("실루엣 점수를 계산하지 못했습니다.")
        st.exception(e)

else:
    st.warning("실루엣 점수를 계산할 수 없습니다.")


# ==================================================
# 11. 2차원 산점도
# ==================================================
st.subheader("6. 2차원 산점도")

col_x, col_y = st.columns(2)

with col_x:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        format_func=lambda x: FEATURE_LABELS[x],
        key="scatter_x",
    )

with col_y:
    y_feature = st.selectbox(
        "세로축 속성",
        options=selected_features,
        index=1,
        format_func=lambda x: FEATURE_LABELS[x],
        key="scatter_y",
    )

fig2 = px.scatter(
    valid_df,
    x=x_feature,
    y=y_feature,
    color="묶음",
    hover_name="영화 제목",
    hover_data={
        x_feature: ":.3f",
        y_feature: ":.3f",
        "묶음": True,
    },
    category_orders={
        "묶음": active_symbols,
    },
    labels={
        **FEATURE_LABELS,
        "묶음": "영화 유형",
    },
    title=f"선택한 두 속성으로 본 영화 유형 ({cluster_count}개 묶음)",
)

fig2.update_traces(marker={"size": 7})
fig2.update_layout(legend_title_text="영화 유형")

st.plotly_chart(
    fig2,
    use_container_width=True,
)


# ==================================================
# 12. 3차원 산점도
# ==================================================
st.subheader("7. 3차원 산점도")

if len(selected_features) < 3:
    st.info(
        "3차원 산점도를 보려면 속성을 세 개 이상 선택해 주세요."
    )

else:
    axis_cols = st.columns(3)

    with axis_cols[0]:
        x3 = st.selectbox(
            "X축 속성",
            selected_features,
            index=0,
            format_func=lambda x: FEATURE_LABELS[x],
            key="x3",
        )

    with axis_cols[1]:
        y3 = st.selectbox(
            "Y축 속성",
            selected_features,
            index=1,
            format_func=lambda x: FEATURE_LABELS[x],
            key="y3",
        )

    with axis_cols[2]:
        z3 = st.selectbox(
            "Z축 속성",
            selected_features,
            index=2,
            format_func=lambda x: FEATURE_LABELS[x],
            key="z3",
        )

    fig3 = px.scatter_3d(
        valid_df,
        x=x3,
        y=y3,
        z=z3,
        color="묶음",
        hover_name="영화 제목",
        hover_data={
            x3: ":.3f",
            y3: ":.3f",
            z3: ":.3f",
            "묶음": True,
        },
        category_orders={
            "묶음": active_symbols,
        },
        labels={
            **FEATURE_LABELS,
            "묶음": "영화 유형",
        },
        title=f"선택한 세 속성으로 본 영화 유형 ({cluster_count}개 묶음)",
    )

    fig3.update_traces(marker={"size": 3})
    fig3.update_layout(legend_title_text="영화 유형")

    st.plotly_chart(
        fig3,
        use_container_width=True,
    )


# ==================================================
# 13. 군집별 편수와 평균
# ==================================================
st.subheader("8. 묶음별 편수와 평균")

summary = (
    valid_df.groupby("묶음")
    .agg(
        편수=("movieNm", "size"),
        스크린수_평균=("first_scrn", "mean"),
        누적관객_평균=("total_audi", "mean"),
        십위권일수_평균=("days_in_top10", "mean"),
        롱런지수_평균=("long_run_index", "mean"),
    )
    .reindex(active_symbols)
    .rename_axis("영화 유형")
    .reset_index()
)

summary = summary.rename(
    columns={
        "스크린수_평균": "스크린 수 평균",
        "누적관객_평균": "누적 관객 평균",
        "십위권일수_평균": "10위권 일수 평균",
        "롱런지수_평균": "롱런 지수 평균",
    }
)

summary_display = summary.copy()

for col in [
    "스크린 수 평균",
    "누적 관객 평균",
    "10위권 일수 평균",
    "롱런 지수 평균",
]:
    summary_display[col] = summary_display[col].map(
        lambda x: "" if pd.isna(x) else f"{x:,.2f}"
    )

summary_display["편수"] = summary_display["편수"].map(
    lambda x: "" if pd.isna(x) else f"{int(x):,}"
)

st.dataframe(
    summary_display,
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "평균은 로그 변환 전 원래 단위로 계산했습니다. "
    "롱런 지수는 누적 관객을 첫 주 관객으로 나눈 값이며 "
    "20을 상한으로 적용했습니다."
)


# ==================================================
# 14. 묶음별 누적 관객 상위 영화 5편
# ==================================================
st.subheader("9. 묶음별 누적 관객 상위 영화 5편")

for label in active_symbols:
    group = (
        valid_df[valid_df["묶음"] == label]
        .sort_values(
            "total_audi",
            ascending=False,
        )
        .head(5)
    )

    with st.expander(
        f"{label} 묶음 — 누적 관객 상위 {len(group)}편",
        expanded=True,
    ):
        if group.empty:
            st.write("이 묶음에 해당하는 영화가 없습니다.")

        else:
            top_movies = group[
                ["movieNm", "total_audi"]
            ].rename(
                columns={
                    "movieNm": "영화 제목",
                    "total_audi": "누적 관객",
                }
            ).copy()

            top_movies["누적 관객"] = (
                top_movies["누적 관객"].map(
                    lambda x: f"{x:,.0f}"
                )
            )

            st.dataframe(
                top_movies,
                hide_index=True,
                use_container_width=True,
            )


# ==================================================
# 15. 마무리 안내
# ==================================================
st.divider()

st.caption(
    "군집 기호는 누적 관객 평균이 높은 순서대로 부여합니다. "
    "군집 수 또는 분석 속성을 변경하면 거리 제곱 합, "
    "실루엣 점수, 산점도, 통계표와 영화 목록이 다시 계산됩니다."
)
