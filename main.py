if len(selected_features) < 2:
    st.warning("군집화를 위해 속성을 두 개 이상 선택해 주세요.")
    st.stop()
if len(valid) < 3:
    st.error("유효한 영화가 3편 미만이라 세 그룹으로 나눌 수 없습니다.")
    st.stop()

# 선택한 속성만 표준화해 K-means 적용. 고정된 난수와 n_init으로 결과 재현성을 확보한다.
X = valid[selected_features].astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
valid["cluster_raw"] = model.fit_predict(X_scaled)

# 누적 관객의 원래 값 평균이 큰 군집부터 ㉮, ㉯, ㉰로 재명명한다.
cluster_order = (
    valid.groupby("cluster_raw")["total_audi_raw"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
cluster_to_label = {cluster_id: GROUP_LABELS[i] for i, cluster_id in enumerate(cluster_order)}
valid["유형"] = valid["cluster_raw"].map(cluster_to_label)

st.subheader("2. 2차원 산점도")
col_x, col_y = st.columns(2)
with col_x:
    x_label = st.selectbox("가로축 속성", options=list(feature_options.keys()), index=0, key="scatter_x")
with col_y:
    y_default = 1 if len(feature_options) > 1 else 0
    y_label = st.selectbox("세로축 속성", options=list(feature_options.keys()), index=y_default, key="scatter_y")

plot_df = valid.copy()
plot_df["제목"] = plot_df["movieNm"].fillna("(제목 없음)").astype(str)
fig2 = px.scatter(
    plot_df,
    x=feature_options[x_label],
    y=feature_options[y_label],
    color="유형",
    category_orders={"유형": ["㉮", "㉯", "㉰"]},
    hover_name="제목",
    hover_data={"유형": True, "제목": False, "cluster_raw": False},
    labels={
        feature_options[x_label]: x_label,
        feature_options[y_label]: y_label,
        "유형": "영화 유형",
    },
    title=f"{x_label}과(와) {y_label}에 따른 영화 유형",
)
fig2.update_traces(marker={"size": 7, "opacity": 0.78})
fig2.update_layout(legend_title_text="영화 유형", margin=dict(l=10, r=10, t=55, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.subheader("3. 3차원 산점도")
if len(selected_features) < 3:
    st.info("군집화에 선택한 속성이 세 개보다 적어서 3차원 산점도는 표시하지 않습니다. 위에서 속성을 세 개 이상 선택해 주세요.")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        x3_label = st.selectbox("X축 속성", options=list(feature_options.keys()), index=0, key="x3")
    with c2:
        y3_label = st.selectbox("Y축 속성", options=list(feature_options.keys()), index=1, key="y3")
    with c3:
        z3_label = st.selectbox("Z축 속성", options=list(feature_options.keys()), index=2, key="z3")
    fig3 = px.scatter_3d(
        plot_df,
        x=feature_options[x3_label],
        y=feature_options[y3_label],
        z=feature_options[z3_label],
        color="유형",
        category_orders={"유형": ["㉮", "㉯", "㉰"]},
        hover_name="제목",
        hover_data={"유형": True, "제목": False, "cluster_raw": False},
        labels={
            feature_options[x3_label]: x3_label,
            feature_options[y3_label]: y3_label,
            feature_options[z3_label]: z3_label,
            "유형": "영화 유형",
        },
        title="선택한 속성으로 본 영화 유형",
    )
    fig3.update_traces(marker={"size": 3, "opacity": 0.8})
    fig3.update_layout(legend_title_text="영화 유형", margin=dict(l=0, r=0, t=55, b=0))
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("4. 유형별 편수와 속성 평균")
summary = valid.groupby("유형").agg(
    편수=("movieNm", "size"),
    스크린수_평균=("first_scrn_raw", "mean"),
    누적관객_평균=("total_audi_raw", "mean"),
    십위권일수_평균=("days_in_top10_raw", "mean"),
    롱런지수_평균=("long_run_index_raw", "mean"),
).reindex(["㉮", "㉯", "㉰"])
summary = summary.rename(columns={
    "스크린수_평균": "스크린 수 평균",
    "누적관객_평균": "누적 관객 평균",
    "십위권일수_평균": "10위권 일수 평균",
    "롱런지수_평균": "롱런 지수 평균",
})
summary_display = summary.copy()
summary_display["편수"] = summary_display["편수"].map(lambda x: f"{int(x):,}")
summary_display["스크린 수 평균"] = summary_display["스크린 수 평균"].map(lambda x: f"{x:,.1f}")
summary_display["누적 관객 평균"] = summary_display["누적 관객 평균"].map(lambda x: f"{x:,.0f}")
summary_display["10위권 일수 평균"] = summary_display["10위권 일수 평균"].map(lambda x: f"{x:,.1f}")
summary_display["롱런 지수 평균"] = summary_display["롱런 지수 평균"].map(lambda x: f"{x:,.2f}")
st.dataframe(summary_display, use_container_width=True)

st.subheader("5. 유형별 누적 관객 상위 영화 5편")
for label in ["㉮", "㉯", "㉰"]:
    group = valid.loc[valid["유형"] == label].sort_values("total_audi_raw", ascending=False).head(5)
    with st.expander(f"{label} 유형 — 누적 관객 상위 {len(group)}편", expanded=True):
        if group.empty:
            st.write("이 유형에 해당하는 영화가 없습니다.")
        else:
            top = group[["movieNm", "total_audi_raw", "openDt", "genre"]].copy()
            top.columns = ["영화 제목", "누적 관객", "개봉일", "장르"]
            top["누적 관객"] = top["누적 관객"].map(lambda x: f"{x:,.0f}")
            st.dataframe(top.reset_index(drop=True), use_container_width=True, hide_index=True)

st.caption("참고: 스크린 수와 누적 관객은 군집화 전에 log10(값)로 변환했습니다. 그룹별 평균 표에는 원래 단위 값을 사용합니다. 롱런 지수는 누적 관객 ÷ 첫 주 관객이며 최대 20으로 제한합니다.")

