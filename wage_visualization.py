import streamlit as st

std_hours = 176

st.markdown(
    "기본급이 **220만원 → 330만원**으로 바뀌고, "
    "소정근로시간이 176시간으로 고정일 때의 변화를 봅시다,<br>"
    "**시급과 수당이 어떻게 달라질까요?**",
    unsafe_allow_html=True
)

col1, spacer, col2 = st.columns([2, 0.2, 1])

with col1:
    selected_salary = st.radio("기본급을 선택하세요", [220, 330], horizontal=True, key="salary_case1")

    extra_hours = st.slider("추가근로시간 (시간)", min_value=0, max_value=100, step=5, value=2,key="hours_case1")

hourly = (selected_salary * 10000) / std_hours
extra_pay = hourly * extra_hours

with col2:
    st.markdown("#### 계산 결과")
    st.markdown(f"""
    <div style='font-size: 16px; line-height: 2'>
        <b>기본급:</b> {selected_salary}만원<br>
        <b>시급:</b> {int(hourly):,} 원<br>
        <b>수당:</b> {int(extra_pay):,} 원
    </div>
    """, unsafe_allow_html=True)


 # 두 번째 계산: 같은 추가근로시간으로 수당이 얼마나 달라질까?
import streamlit as st

# 고정값
std_hours = 176
target_allowance = st.slider("받고자 하는 수당 (원)", min_value=100_000, max_value=600_000, step=10_000, value=300_000)

#st.markdown("### 💼 기본급별 추가근로시간 비교")
st.markdown(f"""
- **소정근로시간:** {std_hours}시간 (고정)  
- **수당:** {target_allowance:,}원 (고정)
""")

# 비교용 표 그리기
cols = st.columns(2)

for idx, salary in enumerate([220, 330]):
    with cols[idx]:
        st.markdown(f"#### 기본급: {salary}만원")

        hourly = (salary * 10000) / std_hours
        required_hours = target_allowance / hourly

        st.markdown(f"""
        - **시급:** {int(hourly):,} 원  
        - **필요 추가근로시간:**  
          <span style='font-size:22px; color:crimson'><b>{required_hours:.1f}시간</b></span>
        """, unsafe_allow_html=True)
