
# 누적 관객 평균이 큰 군집부터 ㉮, ㉯, ㉰를 부여합니다.
cluster_order = (
    valid_df.groupby("cluster_raw")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
cluster_labels = {cluster_id: ["㉮", "㉯", "㉰"][rank] for rank, cluster_id in enumerate(cluster_order)}
valid_df["묶음"] = valid_df["cluster_raw"].map(cluster_labels)

st.subheader("2. 2차원 산점도")
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

plot_df = valid_df.copy()
plot_df["영화 제목"] = plot_df["movieNm"].fillna("(제목 없음)").astype(str)
fig2 = px.scatter(
    plot_df,
    x=x_feature,
    y=y_feature,
    color="묶음",
    hover_name="영화 제목",
    hover_data={x_feature: ":.3f", y_feature: ":.3f", "묶음": True},
    category_orders={"묶음": ["㉮", "㉯", "㉰"]},
    labels={**FEATURE_LABELS, "묶음": "영화 유형"},
    title="선택한 두 속성으로 본 영화 유형",
)
fig2.update_traces(marker={"size": 7})
fig2.update_layout(legend_title_text="영화 유형")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("3. 3차원 산점도")
if len(selected_features) < 3:
    st.info("3차원 산점도를 보려면 군집화에 사용할 속성을 세 개 이상 선택해 주세요.")
else:
    axis_cols = st.columns(3)
    with axis_cols[0]:
        x3 = st.selectbox(
            "X축 속성", selected_features, index=0,
            format_func=lambda x: FEATURE_LABELS[x], key="x3",
        )
    with axis_cols[1]:
        y3 = st.selectbox(
            "Y축 속성", selected_features, index=1,
            format_func=lambda x: FEATURE_LABELS[x], key="y3",
        )
    with axis_cols[2]:
        z3 = st.selectbox(
            "Z축 속성", selected_features, index=2,
            format_func=lambda x: FEATURE_LABELS[x], key="z3",
        )

    fig3 = px.scatter_3d(
        plot_df,
        x=x3,
        y=y3,
        z=z3,
        color="묶음",
        hover_name="영화 제목",
        hover_data={x3: ":.3f", y3: ":.3f", z3: ":.3f", "묶음": True},
        category_orders={"묶음": ["㉮", "㉯", "㉰"]},
        labels={**FEATURE_LABELS, "묶음": "영화 유형"},
        title="선택한 세 속성으로 본 영화 유형",
    )
    fig3.update_traces(marker={"size": 3})
    fig3.update_layout(legend_title_text="영화 유형")
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("4. 묶음별 편수와 평균")
summary = (
    valid_df.groupby("묶음")
    .agg(


        편수=("movieNm", "size"),
        스크린수_평균=("first_scrn", "mean"),
        누적관객_평균=("total_audi", "mean"),
        십위권일수_평균=("days_in_top10", "mean"),
        롱런지수_평균=("long_run_index", "mean"),
    )
    .reindex(["㉮", "㉯", "㉰"])
    .reset_index()
    .rename(columns={
        "묶음": "영화 유형",
        "스크린수_평균": "스크린 수 평균",
        "누적관객_평균": "누적 관객 평균",
        "십위권일수_평균": "10위권 일수 평균",
        "롱런지수_평균": "롱런 지수 평균",
    })
)
summary_display = summary.copy()
for col in ["스크린 수 평균", "누적 관객 평균", "10위권 일수 평균", "롱런 지수 평균"]:
    summary_display[col] = summary_display[col].map(lambda x: f"{x:,.2f}")
summary_display["편수"] = summary_display["편수"].map(lambda x: f"{x:,}")
st.dataframe(summary_display, hide_index=True, use_container_width=True)
st.caption("평균은 로그 변환 전의 원래 단위로 계산했습니다. 롱런 지수는 누적 관객 ÷ 첫 주 관객이며 20을 상한으로 적용했습니다.")

st.subheader("5. 묶음별 누적 관객 상위 영화 5편")
for label in ["㉮", "㉯", "㉰"]:
    group = valid_df[valid_df["묶음"] == label].sort_values("total_audi", ascending=False).head(5)
    with st.expander(f"{label} 묶음 — 누적 관객 상위 {len(group)}편", expanded=True):
        if group.empty:
            st.write("이 묶음에 해당하는 영화가 없습니다.")
        else:
            top_movies = group[["movieNm", "total_audi"]].rename(
                columns={"movieNm": "영화 제목", "total_audi": "누적 관객"}
            ).copy()
            top_movies["누적 관객"] = top_movies["누적 관객"].map(lambda x: f"{x:,.0f}")
            st.dataframe(top_movies, hide_index=True, use_container_width=True)
