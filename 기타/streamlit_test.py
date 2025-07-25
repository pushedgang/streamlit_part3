
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

#############################기본급 + 수당 변화 그래프 (노조)###################################


import streamlit as st
import plotly.graph_objects as go

st.markdown("""
## 3️⃣ 서울시의 주장, 무엇이 문제일까?

그렇다면 서울시가 주장하는 통상임금 개편안은 어떠한 문제가 있는 것일까요?

서울시는 상여금을 기본급에 포함하되, 총 임금을 ‘동결’할 것을 주장하였습니다. 반대로 버스 노조 측은 상여금이 기본급에 포함되는데 임금이 동결되는 것은 곧 인정근로시간을 줄이는 것과 마찬가지라고, 기본급이 늘어났는데 총 임금이 같은 건 말이 안 된다고 하고 있죠. 

이를 이해하기 위해선 우선 ‘간주근로시간제’가 무엇인지 이해해야 합니다

""")
###############################################################
st.markdown("""
### 간주근로시간제란?

간주근로시간제는 **사업장 밖에서 근로 시간을 실제적으로 측정하기 어려운 경우, 근로 시간을 인정하는 제도**입니다. 대표적으로 재택근무, AS, 영업직 등이 있고 버스기사도 역시 대표적인 간주근로시간제로 임금을 받는 직업입니다.


얼마나 일했는지가 때에 따라 다르고, 기준도 애매하니 ‘**이만큼을 일했다**’라고 회사랑 합의하고 일한 시간과 무관하게 합의한 시간에 맞는 급여를 받는 방식이죠.  


버스기사님들은 기사님별로 일하는 날짜 (주간, 주말)도 다르고, 어떤 날은 밤에, 어떤날은 낮에 일하곤 합니다. 고정된 근무 시간을 측정하기 어려운 직업이죠.

따라서 버스기사의 경우에는 주휴수당 제외 176시간, 주휴수당 포함 시 209시간을 일하는 것으로 정해져 있습니다. 월 22일 기준, 하루 9시간 (8시간 + 1시간 연장)을 모두 일해야 ‘만근’으로 인정됩니다. 


그러면 간주근로시간제에 따라 일한다는 것을 이해했으니, 다시 서울시와 노조의 입장 차이를 봐볼까요?
""", unsafe_allow_html=True)




############################수당 산정 식 ##################################

st.markdown("""
### 서울시는 무엇을 주장하나?
우선 간단한 시급과 수당 간의 관계를 살펴봅시다.
""")


import streamlit as st

st.latex(r'''
\frac{\text{기본급}}{\text{소정근로시간}} = \text{시급}
''')

st.latex(r'''
\text{시급} \times \text{연장근로시간} = \text{수당}
''')
###########################끝 수당 산정 식#################################
st.write("")

st.markdown("""
            

* ‘시급’은 기본급과 근로시간을 기준으로 산정됩니다. 즉 **시급 = 기본급/근로시간** 인 것이죠. 

* ‘수당’은 이렇게 산정된 시급과 추가근로시간의 곱으로 정해집니다. 즉 **수당 = 시급 * 추가근로시간** 입니다,


이제 시급과 수당이 무엇인지 이해한 채로, 서울시와 노조 각각의 주장을 살펴봅시다.

""")

st.write("")
st.write("")

st.markdown("""
            #### 노조의 주장
            """)

st.markdown("""
            
            노조가 주장하는 것은 다음과 같습니다.""")

import streamlit as st
import plotly.graph_objects as go

# 🎨 색상 함수
def get_allow_color(val):
    norm_val = (val - 110) / 55
    norm_val = max(0, min(norm_val, 1))
    rg_value = int(190 - 130 * norm_val)
    return f'rgb({rg_value}, {rg_value}, 255)'

# 🎁 왼쪽 여백을 위한 columns 활용
left_space, main, right_space = st.columns([0.2, 5,0.2])  # 왼쪽에 1 비율의 여백, 본문은 5

with main:
    # 🎚️ 슬라이더 입력
    base_salary = st.slider("기본급 입력 (만원)", 220, 330, 220, step=10)
    allowance = base_salary * 0.5

    # 색상 설정
    base_color = 'rgb(100, 180, 255)'
    allow_color = get_allow_color(allowance)

    # 📊 그래프 생성
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=["급여"],
        x=[base_salary],
        name="기본급",
        orientation="h",
        marker=dict(color=base_color, line=dict(color='black', width=2)),
        text=[f"{int(base_salary)}만원"],
        textposition='inside',
        hovertemplate='기본급<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        y=["급여"],
        x=[allowance],
        name="수당",
        orientation="h",
        marker=dict(color=allow_color, line=dict(color='black', width=2)),
        text=[f"{int(allowance)}만원"],
        textposition='inside',
        hovertemplate='수당<extra></extra>'
    ))

    fig.update_layout(
        barmode='stack',
        title=f"임금 총액: {int(base_salary + allowance)}만원",
        xaxis=dict(title='총 금액 (만원)', range=[0, 600]),
        yaxis=dict(title=''),
        transition_duration=500,
        height=300
    )

    # 📈 출력
    st.plotly_chart(fig, use_container_width=True)




st.markdown("""
 수당은 기본급과 비례하니, 기본급이 늘면 수당도 늘어야한다라는 것이죠. 사실상 맞는 말이고요.
""")
st.write("")
st.write("")



#############################기본급 + 수당 고정 그래프 (서울시)###################################

st.markdown("""
            #### 서울시의 주장
            """)

st.markdown("""
**그러나 서울시는 이렇게 주장합니다.** """)


image = Image.open("/Users/min/Documents/대학교/강의/25-1/데이터 저널리즘/팀플/이미지/서울시임금체계개편안.webp")  # 예: "img/salary_chart.png"
st.image(image, caption="임금 구조 차트 예시", use_container_width=True)

st.markdown("""

“상여금을 기본급에 포함해서 기본급을 늘리지만, 임금 총액은 유지하겠다. 일단 기존임금을 보전한 뒤 임금인상률은 추후 논의하자 “ 


이는 아래 표를 보면 '임금 총액을 유지한다'라는 뜻이 무엇인지 이해해볼 수 있습니다       
           """)

st.write("")
st.write("")



import streamlit as st
import plotly.graph_objects as go

# 총 급여 고정
total_salary = 440

# 여백 + 본문 구조
left, main, right = st.columns([0.2, 5, 0.2])

with main:
    # 기본급 슬라이더: 200 ~ 300
    base_salary = st.slider("기본급 (만원)", 220, 330, 220, step=10)

    # 수당 자동 계산
    allowance = total_salary - base_salary

    # 색상 함수 (수당만 색상 변화)
    def get_allow_color(val):
        norm_val = val / 200  # 수당 최대 200 기준 정규화
        norm_val = max(0, min(norm_val, 1))
        rg_value = int(190 - 130 * norm_val)
        return f'rgb({rg_value}, {rg_value}, 255)'

    base_color = 'rgb(180, 180, 255)'
    allow_color = get_allow_color(allowance)

    # 시각화 시작
    fig = go.Figure()

    # 기본급 바
    fig.add_trace(go.Bar(
        y=["급여"],
        x=[base_salary],
        name="기본급",
        orientation="h",
        marker=dict(
            color=base_color,
            line=dict(color='black', width=2)
        ),
        text=[f"{int(base_salary)}만원"],
        textposition='inside',
        hovertemplate='기본급<extra></extra>'
    ))

    # 수당 바
    fig.add_trace(go.Bar(
        y=["급여"],
        x=[allowance],
        name="수당",
        orientation="h",
        marker=dict(
            color=allow_color,
            line=dict(color='black', width=2)
        ),
        text=[f"{int(allowance)}만원"],
        textposition='inside',
        hovertemplate='수당<extra></extra>'
    ))

    # 레이아웃
    fig.update_layout(
        barmode='stack',
        title=f"임금 총액: {total_salary}만원 (기본급 {int(base_salary)} / 수당 {int(allowance)})",
        xaxis=dict(title="총 금액 (만원)", range=[0, 600]),
        yaxis=dict(title=""),
        height=300,
        transition_duration=500
    )

    # 출력
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
            즉, 일단 상여금이 기본급화 된다고 하더라도, 이에 따른 수당의 변화분은 생기지 않는다는 것입니다.
            
            그러나 생각을 한번 해봅시다. 기본급이 올랐으면, 근로시간이 같은 이상 시급은 자연스럽게 오르게 되어 있습니다.
            """)


image = Image.open("/Users/min/Documents/대학교/강의/25-1/데이터 저널리즘/팀플/이미지/기본급_시급.png")  # 예: "img/salary_chart.png"
st.image(image, caption="4시간 근로 시 기본급 총액이 인상되면, 하나의 덩어리인 시급도 함께 오릅니다", use_container_width=True)

st.markdown("""
            그러나 기본급이 올랐는데 임금 총액을 유지한다는 말은 곧
            **추가 근로로 인정되는 시간**을 줄이겠다는 말과 같습니다.  
            """)

#############################이미지#################################



image = Image.open("/Users/min/Documents/대학교/강의/25-1/데이터 저널리즘/팀플/이미지/수당_근로시간.png")  # 예: "img/salary_chart.png"
st.image(image, caption="수당의 변화분이 없게 된다면, 이는 곧 인정되는 시간을 줄이겠다는 뜻입니다", use_container_width=True)

st.markdown("""
            시급이 올랐지만, 원래 4시간 일한 것을 3시간 일한 것과 같이 치겠다는 뜻이죠. 
            
            대학생 입장에서 이야기를 해보면, 알바 시급은 올려주지만, 8시간 일하던 걸 7시간 일한 걸로 퉁치자 라고 하는 말과 같은 겁니다.
            """)

###########################인용문 style###########################
st.markdown("""
<div style='
    background-color: #f9f9f9;
    padding: 30px;
    margin: 20px 0;
    font-style: normal;
    color: #333;
    line-height: 1.6;
    font-size : 0.9rem;
'>
실제 근로시간이 지금 1일 9시간을 기준으로 주는데  
한 7시간 반, 8시간 이렇게 일해도 9시간 임금을 주고,  
오후 근무 같은 경우는 11시간 넘기고 12시간 가까이 돼도  
9시간 임금을 받아요. 그러니까 퉁쳐서 그렇게 받는 시스템이에요.  

그런데 이제 실제 근무하는 시간이 줄어드는 건 아닌데 
**임금 산정 시간만 줄이겠다는 게 사측과 서울시의 시도**예요.  
저희로서는 그런 제안을 받아들이기 힘든 거죠.

<div style='text-align: right; font-size: 0.85em; margin-top: 8px; color: #666;'>
— 유재호 노무사님

</div>
""", unsafe_allow_html=True)


st.markdown("""
            간주근로제에 따라 '이만큼 일한다고 치자' 라고 합의한 시간이 줄어드는 것과 마찬가지입니다. 
            
            **실상 일하는 시간은 줄어들지 않는데도 말이죠.**
            
            노조 측은 부산의 사례를 모범 사례로 제시합니다. 
            
            비록 통상임금 개편안 적용 외의 추가적인 기본급 인상은 미미하였지만, 그럼에도 교섭 테이블에서 진솔한 태도로 임하며 대법원 판결을 존중하였다는 점을 노조 관계자는 긍정적으로 받아들였습니다.
            """)


st.markdown("""
<div style='
    background-color: #f9f9f9;
    padding: 30px;
    margin: 20px 0;
    font-style: normal;
    color: #333;
    line-height: 1.6;
    font-size : 0.9rem;
'>
 부산 시 교통혁신국장이, 서울시로 하면 도시교통실인데 그 사람이 시의회에서 나와서 한 얘기가 있어요. 서울시 의원이 "왜 재정 부담이 이렇게 많이 되는 거를 섣불리 합의를 했냐"라고 하니까 그 교통 혁신 국장, 부산시 책임 공무원이 "법에 따라서 올라온 걸 어떻게 안 지키냐", "그건 지키고 그 대신 **노조의 양보를 받아서** 임금 동결을 했다" (라고 말했다)


<div style='text-align: right; font-size: 0.85em; margin-top: 8px; color: #666;'>
— 유재호 노무사님

</div>
""", unsafe_allow_html=True)

st.markdown("""
            말한 김에, 버스 기사님들은 얼마나 일하고 계신건지 한번 살펴볼까요? 연봉 5400만원, 이거는 어떻게 산정된 것일까요?
            """)

###################################버스기사 임금 현실#####################################
st.markdown("""
            ### 버스 기사 임금 현실, 6000만원은 어디서?
            """)

import streamlit as st
import plotly.graph_objects as go

# 페이지 설정
#st.set_page_config(page_title="버스기사 임금 시각화", layout="centered")

# 데이터
components = {
    "기본급": 220,
    "상여금": 110,
    "수당": 80,
    "주휴수당": 40,
    "선택추가근무": 30,
    "무사고": 20
}

descriptions = {
    "기본급": "월 기준 정해진 고정 급여",
    "상여금": "일정한 기간(주로 월)을 기준으로 반복적으로 지급되는 급여 외에 추가적으로 지급되는 금품",
    "수당": "연장·야간·휴일 근무에 따른 추가 지급",
    "주휴수당": "유급 주휴일 보장 수당",
    "선택추가근무": "자발적 추가근무에 대한 보상",
    "무사고": "무사고 근무에 대한 인센티브"
}

colors = ["#c7d4e2", "#a5bdd5", "#80a4c2", "#678db5", "#4f76a5", "#3b5c8a"]


# 📊 그래프
fig = go.Figure()
for i, (label, value) in enumerate(components.items()):
    fig.add_trace(go.Bar(
        y=["임금 구성"],
        x=[value],
        name=label,
        orientation='h',
        marker=dict(
            color=colors[i % len(colors)],
            line=dict(color='black', width=1)
        ),
        text=[label],
        textposition='inside',
        insidetextanchor='middle',
        hovertemplate=f"{label}: {value}만원<extra></extra>"
    ))

fig.update_layout(
    barmode='stack',
    height=250,
    title="버스기사 임금구조표(5호봉 기준)",
    xaxis=dict(title="총 임금 (만원)", range=[0, sum(components.values()) + 100]),
    yaxis=dict(showticklabels=False),
    margin=dict(l=20, r=20, t=40, b=20),
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# 구분선
#st.divider()

# 항목 설명 선택
#st.markdown("### ℹ️ 항목 설명 (선택 시 나타납니다)", help="선택 시 간단한 설명을 보여줘요.")

selected_item = st.selectbox(
    "항목 선택", 
    list(components.keys()),
    label_visibility="collapsed"  # ✅ 레이블 숨기기!
)

# 설명을 작고 흐리게 표현
st.markdown(f"""
<div style='
    font-size: 0.9rem;
    background-color: #f8f9fa;
    padding: 12px 16px;
    border-left: 4px solid #dee2e6;
    margin-top: 10px;
    color: #444;
'>
<strong>{selected_item}</strong><br>
{descriptions[selected_item]}
</div>
""", unsafe_allow_html=True)



st.write("")

st.markdown("""
            
            하루 9시간씩, 때론 새벽시간까지도 22일동안 무사고 운전을 하고, 그 22일에 더해서 추가근무까지 해야 월 500만원이 겨우 채워집니다.
            
            연장근로, 야간근로가 많다보니 애초에 임금구조가 기형적으로 잡혀있고, 기본급으로 포함을 시키면 수당이 높아지니, 받아야 할 돈을 다 상여금으로 빼놓고 수당은 기본급을 기준으로 주고 있던 거죠.
            """)
st.markdown("""
<div style='
    background-color: #f9f9f9;
    padding: 30px;
    margin: 20px 0;
    font-style: normal;
    color: #333;
    line-height: 1.6;
    font-size : 0.9rem;
'>
국내 봉급자들이 대부분 그렇듯, **버스기사들의 봉급도 기본급, 상여금, 수당으로 구성**됩니다.  
서울시 버스정책과에 따르면, 기본급과 상여금이 전체 임금의 60%, 수당이 40%를 차지합니다.  

이 중 **2개월마다 동일 금액으로 지급되는 상여금**은,  
수당 중심 임금 체계에 대한 노동자의 반발을 무디게 하기 위한 장치입니다.  
즉, 기본급을 낮게 묶고, 그를 바탕으로 산정되는 **수당 인상 폭을 제한**하려는 전략이죠.  

이에 따라 **서울의 버스기사들을 대표한 동아운수 노조는**  
2015년 상여금을 통상임금에 산입해달라는 소송을 제기했습니다.  
해당 소송은 1심에서 패소했으며, 현재 2심에 계류 중입니다.

<div style='text-align: right; font-size: 0.85em; margin-top: 8px; color: #666;'>
— 오마이뉴스 기사 인용  
<a href="https://n.news.naver.com/article/079/0004026584?sid=102" target="_blank">[원문 보기]</a>
</div>
""", unsafe_allow_html=True)




st.markdown("""
            **그러나, 이제 대법원 판결이 적용되면?**
            
            """)

import streamlit as st
import plotly.graph_objects as go

# 페이지 설정
#st.set_page_config(page_title="버스기사 임금 시각화", layout="centered")

# 데이터
components = {
    "기본급(상여금 포함)": 330,
    "수당": 120,
    "주휴수당": 60,
    "선택추가근무": 30,
    "무사고": 20
}

descriptions = {
    "기본급": "월 기준 정해진 고정 급여",
    "상여금": "일정한 기간(주로 월)을 기준으로 반복적으로 지급되는 급여 외에 추가적으로 지급되는 금품",
    "수당": "연장·야간·휴일 근무에 따른 추가 지급",
    "주휴수당": "유급 주휴일 보장 수당",
    "선택추가근무": "자발적 추가근무에 대한 보상",
    "무사고": "무사고 근무에 대한 인센티브"
}
colors = [
    "#F2C94C",  # 부드러운 골드 (기본급)
    "#F2998D",  # 연코랄 (상여금)
    "#6BB5D9",  # 맑은 하늘블루 (수당)
    "#4F8FBF",  # 차분한 블루 (추가근무)
    "#3D5A80"   # 딥블루 (무사고)
]


# 📊 그래프
fig = go.Figure()
for i, (label, value) in enumerate(components.items()):
    fig.add_trace(go.Bar(
        y=["임금 구성"],
        x=[value],
        name=label,
        orientation='h',
        marker=dict(
            color=colors[i % len(colors)],
            line=dict(color='black', width=1)
        ),
        text=[label],
        textposition='inside',
        insidetextanchor='middle',
        hovertemplate=f"{label}: {value}만원<extra></extra>"
    ))

fig.update_layout(
    barmode='stack',
    height=250,
    title="버스기사 임금구조표(5호봉 기준)",
    xaxis=dict(title="총 임금 (만원)", range=[0, sum(components.values()) + 100]),
    yaxis=dict(showticklabels=False),
    margin=dict(l=20, r=20, t=40, b=20),
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""

100%에 해당하는 상여금이 모두 통상임금에 포함이 되면서 수당 역시 상여금 포함분을 기준으로 책정되게 되는 것입니다. 

버스기사는  연장근로와 야간근로가 많아 수당으로 지급받는 임금이 많고, 기본급에 비해 상여금이 높아 상여금이 기본급에 포함될 시 수당 증가분이 가파르게 오르는 직업이 바로 버스기사입니다.
            """)


# tab1, tab2, tab3 = st.tabs(["📌 인트로", "📊 분석", "🗣 인터뷰"])

# with tab1:
#     st.write("인트로 내용")

# with tab2:
#     st.write("그래프와 분석 내용")

# with tab3:
#     st.write("노무사 인터뷰 내용")

#####################################신의성실 원칙#####################################

st.markdown("""
            ### 왜 서울시는 이런 안을 주장하는 걸까?

서울시가 이러한 안을 내세우는 데에는 법적 수싸움도 있다고 노조측에서는 이야기합니다. 

기존에 진행중인 소송에서 유리한 위치를 가져가고자 하는 것인데요, 상여금이 통상임금에 해당되기에 미지급 임금 청구소송을 2012년부터 노조는 제기해 왔고, 2015년에는 서울시 시내버스 회사인 동아운수를 상대로 노조 조합원들이 10년째 소송을 이어가고 있습니다. 

상여금이 통상임금에 해당된다는 판결이 나온 지금, 서울시의 입장은 상당히 불리해졌는데요. 여기서 서울시가 재판을 뒤집을 수 있는 방법은 바로 **신의성실의 원칙**을 이용하는 것입니다.
            """)

st.markdown("""
            **신의성실의 원칙이란?**

```
민법 제2조 (신의성실)
①권리의 행사와 의무의 이행은 신의에 좇아 성실히 하여야 한다.
②권리는 남용하지 못한다.
```
""")

st.markdown("""
서울시는 버스노조를 상대로 두 가지 요건이 만족되어야  
신의성실의 원칙 위반을 주장할 수 있게 됩니다.
""")

st.markdown("""
<div style='
    background-color: #f0f2f6;
    padding: 20px 25px;
    border-left: 5px solid #7f8fa6;
    border-radius: 5px;
    font-size: 0.92rem;
    color: #333;
    line-height: 1.7;
'>
<b>1.</b> 노동조합이 그동안 통상임금이 상여금에 포함되지 않음을 동의해왔다<br>
<b>2.</b> 회사가 망할 위기에 처해있다
</div>
""", unsafe_allow_html=True)


st.markdown("""

현재 재정 상태를 고려했을 때 서울시는 2번은 주장할 수 있지만, 노조는 지속적으로 통상임금이 상여금에 포함된다는 점을 여러 소송을 통해 주장해왔습니다. 또한 현재 대법원 판결이 통상임금에 상여금을 포함하도록 나온 만큼 노조에 더욱 유리해졌습니다. 

그러나? 만약 상여금의 통상임금 포함을 반려하는 서울시 안을 이번에 노조에서 받아들이게 된다면, 신의성실 원칙을 위반하게 되어 이후 소송에서 불리해질 여지가 존재합니다.
""")

st.markdown("""
<div style='
    background-color: #f9f9f9;
    padding: 30px;
    margin: 4px 0;
    font-style: normal;
    color: #333;
    line-height: 1.6;
    font-size : 0.9rem;
'>
저희가 그 신의성실 원칙을 주장하기 위한 첫 번째 요건에서 저희가 동의를 해 준 적이 없잖아요. 지금까지 노조가 오히려 2015년부터 소송을 제기해 왔으니까 그리고 노동조합이 상여금이 통상임금이 아니라는 그런 합의가 있는 것도 아니고. 그러니까 그 동의가 필요한 거예요. 동의 그러니까 그 임금 체계 서울시가 얘기하는 임금 체계 개편에 동의를 하면 이거 봐라. 대법원 판결에 따라서 당연하게 그냥 인정되는 부분을 노조가 포기를 하고 상여금을 그냥 날려버리는 합의안에 동의를 했다.이거를 써먹으려고 하는 거예요. 실제 소송에서 그냥 그런 꼼수가 있는 거고 그래서 뭐 그런 이유로다가 지금 임금 체계 개편을 계속 주장...


<div style='text-align: right; font-size: 0.85em; margin-top: 8px; color: #666;'>
— 유재호 노무사님

</div>
""", unsafe_allow_html=True)