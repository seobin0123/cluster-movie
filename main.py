
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 영화 유형 나누기")
st.write(
    "영화의 개봉 규모와 관객 수, 흥행 지속 기간 등을 이용해 "
    "비슷한 특징을 가진 영화끼리 묶어 봅니다."
)


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "it-teacher-go/data-share-/main/kobis_movies.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    numeric_columns = [
        "first_scrn",
        "first_show",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
        "peak",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 롱런 지수 = 누적 관객 ÷ 첫 주 관객
    # 첫 주 관객이 0 이하이거나 결측이면 계산할 수 없다.
    df["long_run_index"] = np.where(
        df["first_week_audi"] > 0,
        df["total_audi"] / df["first_week_audi"],
        np.nan,
    )

    # 지나치게 큰 값의 영향을 줄이기 위해 20을 상한으로 적용
    df["long_run_index"] = df["long_run_index"].clip(upper=20)

    # 로그 변환은 양수인 값에만 적용
    df["log_first_scrn"] = np.where(
        df["first_scrn"] > 0,
        np.log10(df["first_scrn"]),
        np.nan,
    )

    df["log_total_audi"] = np.where(
        df["total_audi"] > 0,
        np.log10(df["total_audi"]),
        np.nan,
    )

    return df


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 3. 사용할 속성 설정
# --------------------------------------------------
FEATURE_LABELS = {
    "log_first_scrn": "개봉 스크린 수 (log₁₀)",
    "log_total_audi": "누적 관객 수 (log₁₀)",
    "days_in_top10": "10위권에 머문 일수",
    "long_run_index": "롱런 지수",
}

RAW_FEATURES = {
    "log_first_scrn": "first_scrn",
    "log_total_audi": "total_audi",
    "days_in_top10": "days_in_top10",
    "long_run_index": "long_run_index",
}

st.subheader("1. 군집화 설정")

st.write(
    "군집화에 사용할 속성과 묶음 수를 선택하세요. "
    "선택한 속성과 묶음 수에 따라 분석 결과가 다시 계산됩니다."
)

feature_columns = st.columns(2)

default_features = [
    "log_first_scrn",
    "log_total_audi",
    "days_in_top10",
    "long_run_index",
]

selected_features = []

for i, feature in enumerate(FEATURE_LABELS):
    with feature_columns[i % 2]:
        if st.checkbox(
            FEATURE_LABELS[feature],
            value=feature in default_features,
            key=f"feature_{feature}",
        ):
            selected_features.append(feature)

if len(selected_features) < 2:
    st.warning("군집화를 위해 속성을 두 개 이상 선택해 주세요.")
    st.stop()


# --------------------------------------------------
# 4. 군집 수 선택
# --------------------------------------------------
cluster_count = st.selectbox(
    "묶음 수 선택",
    options=list(range(2, 8)),
    index=1,
    format_func=lambda k: f"{k}개",
    key="cluster_count",
)

CLUSTER_SYMBOLS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]

st.info(
    f"현재 선택한 묶음 수는 **{cluster_count}개**입니다. "
    "묶음 기호는 누적 관객 평균이 높은 순서대로 부여합니다."
)


# --------------------------------------------------
# 5. 군집화용 데이터 준비
# --------------------------------------------------
# 실제 군집화에는 사용자가 선택한 속성만 사용한다.
work_df = df.copy()

for col in selected_features:
    work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

valid_mask = (
    work_df[selected_features].notna().all(axis=1)
    & np.isfinite(work_df[selected_features]).all(axis=1)
)

# 군집별 누적 관객 평균을 비교할 수 있도록 누적 관객도 확인
valid_mask &= work_df["total_audi"].notna()
valid_mask &= np.isfinite(work_df["total_audi"])

valid_df = work_df.loc[valid_mask].copy()

if len(valid_df) < max(3, cluster_count + 1):
    st.error(
        f"분석 가능한 영화가 {len(valid_df)}편입니다. "
        f"{cluster_count}개 군집을 분석하려면 영화 수가 더 필요합니다."
    )
    st.stop()

# 선택한 속성들을 표준화한다.
# 단위와 값의 범위가 다른 속성들을 비슷한 기준으로 비교하기 위해서다.
scaler = StandardScaler()
X = scaler.fit_transform(valid_df[selected_features])

# --------------------------------------------------
# 6. 군집 수별 거리 제곱 합 계산
# --------------------------------------------------
# inertia_는 각 영화와 소속 군집 중심 사이의 거리 제곱 합이다.
# 군집 수 1~7을 각각 계산한다.
max_k = min(7, len(valid_df) - 1)
inertia_rows = []

for k in range(1, max_k + 1):
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


# --------------------------------------------------
# 7. 거리 제곱 합 꺾은선 그래프
# --------------------------------------------------
st.subheader("2. 묶음 수에 따른 거리 제곱 합")

st.write(
    "가로축은 묶음 수, 세로축은 각 영화와 소속 군집 중심 사이의 "
    "거리 제곱을 모두 더한 값입니다. "
    "일반적으로 묶음 수가 늘어날수록 이 값은 작아집니다."
)

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

# 현재 선택한 묶음 수 위치에 세로선 표시
fig_inertia.add_vline(
    x=cluster_count,
    line_dash="dash",
    line_width=2,
    annotation_text=f"현재 선택: {cluster_count}개",
    annotation_position="top right",
)

fig_inertia.update_layout(
    xaxis_title="묶음 수",
    yaxis_title="거리 제곱 합",
    xaxis=dict(
        tickmode="linear",
        dtick=1,
        range=[0.8, max_k + 0.2],
    ),
    hovermode="x unified",
)

st.plotly_chart(fig_inertia, use_container_width=True)


# --------------------------------------------------
# 8. 거리 제곱 합과 감소량 표
# --------------------------------------------------
st.subheader("3. 묶음 수별 거리 제곱 합과 감소량")

inertia_table = inertia_df.copy()

inertia_table["직전 값에서 감소한 양"] = (
    inertia_table["거리 제곱 합"].shift(1)
    - inertia_table["거리 제곱 합"]
)

# 첫 번째 행은 비교할 이전 값이 없으므로 빈칸으로 표시
inertia_table["직전 값에서 감소한 양"] = (
    inertia_table["직전 값에서 감소한 양"].map(
        lambda x: "" if pd.isna(x) else f"{x:,.2f}"
    )
)

inertia_table["거리 제곱 합"] = inertia_table["거리 제곱 합"].map(
    lambda x: f"{x:,.2f}"
)

inertia_table["묶음 수"] = inertia_table["묶음 수"].map(
    lambda x: f"{x}개"
)

inertia_table = inertia_table.rename(
    columns={
        "묶음 수": "묶음 수",
        "거리 제곱 합": "거리 제곱 합",
        "직전 값에서 감소한 양": "직전 값에서 감소한 양",
    }
)

st.dataframe(
    inertia_table,
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "거리 제곱 합은 선택한 속성을 표준화한 공간에서 계산했습니다. "
    "감소량은 직전 묶음 수의 거리 제곱 합에서 현재 값을 뺀 결과입니다."
)


# --------------------------------------------------
# 9. 현재 묶음 수의 K-means 군집화
# --------------------------------------------------
kmeans = KMeans(
    n_clusters=cluster_count,
    random_state=42,
    n_init=10,
)

cluster_raw = kmeans.fit_predict(X)

valid_df["cluster_raw"] = cluster_raw

# 누적 관객 평균이 큰 군집부터 기호 부여
cluster_order = (
    valid_df.groupby("cluster_raw")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

cluster_labels = {
    cluster_id: CLUSTER_SYMBOLS[rank]
    for rank, cluster_id in enumerate(cluster_order)
}

valid_df["묶음"] = valid_df["cluster_raw"].map(cluster_labels)

# 영화 제목 결측치 처리
valid_df["영화 제목"] = (
    valid_df["movieNm"].fillna("(제목 없음)").astype(str)
)


# --------------------------------------------------
# 10. 실루엣 점수
# --------------------------------------------------
st.subheader("4. 현재 묶음 수의 실루엣 점수")

# 실루엣 점수는 군집 수가 2개 이상이고
# 실제 군집 라벨 수가 2개 이상이며 영화 수가 충분할 때 계산한다.
actual_cluster_count = len(np.unique(cluster_raw))

if 2 <= actual_cluster_count < len(valid_df):
    score = silhouette_score(X, cluster_raw)

    st.markdown(
        f"**현재 {cluster_count}개 묶음의 실루엣 점수: "
        f"{score:.3f}**"
    )

    st.caption(
        "실루엣 점수는 -1부터 1까지입니다. "
        "1에 가까울수록 각 영화가 자기 군집에 잘 속하고 "
        "다른 군집과도 잘 구분된다는 뜻입니다. "
        "0에 가까우면 군집 경계가 겹칠 수 있으며, "
        "음수이면 일부 영화가 다른 군집에 더 가까울 수 있습니다."
    )
else:
    st.warning(
        "실루엣 점수를 계산할 수 없습니다. "
        "실제로 만들어진 군집이 두 개 이상인지 확인해 주세요."
    )


# --------------------------------------------------
# 11. 2차원 산점도
# --------------------------------------------------
st.subheader("5. 2차원 산점도")

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
    y_default = 1 if len(selected_features) > 1 else 0

    y_feature = st.selectbox(
        "세로축 속성",
        options=selected_features,
        index=y_default,
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
        "묶음": CLUSTER_SYMBOLS[:cluster_count],
    },
    labels={
        **FEATURE_LABELS,
        "묶음": "영화 유형",
    },
    title=f"선택한 두 속성으로 본 영화 유형 ({cluster_count}개 묶음)",
)

fig2.update_traces(marker={"size": 7})
fig2.update_layout(legend_title_text="영화 유형")

st.plotly_chart(fig2, use_container_width=True)


# --------------------------------------------------
# 12. 3차원 산점도
# --------------------------------------------------
st.subheader("6. 3차원 산점도")

if len(selected_features) < 3:
    st.info(
        "3차원 산점도를 보려면 군집화에 사용할 속성을 "
        "세 개 이상 선택해 주세요."
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
            "묶음": CLUSTER_SYMBOLS[:cluster_count],
        },
        labels={
            **FEATURE_LABELS,
            "묶음": "영화 유형",
        },
        title=f"선택한 세 속성으로 본 영화 유형 ({cluster_count}개 묶음)",
    )

    fig3.update_traces(marker={"size": 3})
    fig3.update_layout(legend_title_text="영화 유형")

    st.plotly_chart(fig3, use_container_width=True)


# --------------------------------------------------
# 13. 묶음별 편수와 평균
# --------------------------------------------------
st.subheader("7. 묶음별 편수와 평균")

summary = (
    valid_df.groupby("묶음")
    .agg(
        편수=("movieNm", "size"),
        스크린수_평균=("first_scrn", "mean"),
        누적관객_평균=("total_audi", "mean"),
        십위권일수_평균=("days_in_top10", "mean"),
        롱런지수_평균=("long_run_index", "mean"),
    )
    .reindex(CLUSTER_SYMBOLS[:cluster_count])
    .reset_index()
    .rename(
        columns={
            "묶음": "영화 유형",
            "스크린수_평균": "스크린 수 평균",
            "누적관객_평균": "누적 관객 평균",
            "십위권일수_평균": "10위권 일수 평균",
            "롱런지수_평균": "롱런 지수 평균",
        }
    )
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
    "평균은 로그 변환 전의 원래 단위로 계산했습니다. "
    "롱런 지수는 누적 관객 ÷ 첫 주 관객이며 20을 상한으로 적용했습니다."
)


# --------------------------------------------------
# 14. 묶음별 누적 관객 상위 영화 5편
# --------------------------------------------------
st.subheader("8. 묶음별 누적 관객 상위 영화 5편")

for label in CLUSTER_SYMBOLS[:cluster_count]:
    group = (
        valid_df[valid_df["묶음"] == label]
        .sort_values("total_audi", ascending=False)
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

            top_movies["누적 관객"] = top_movies["누적 관객"].map(
                lambda x: f"{x:,.0f}"
            )

            st.dataframe(
                top_movies,
                hide_index=True,
                use_container_width=True,
            )


# --------------------------------------------------
# 15. 분석 안내
# --------------------------------------------------
st.divider()

st.caption(
    "군집화에는 선택한 속성만 사용하며, 각 속성은 표준화한 뒤 K-means에 "
    "입력합니다. 군집 기호는 누적 관객 평균이 높은 순서대로 부여합니다. "
    "군집 수를 바꾸면 군집화 결과, 산점도, 실루엣 점수, "
    "요약 통계와 영화 목록이 함께 갱신됩니다."
)
