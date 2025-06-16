
###################################유림 파트 #############################
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Webpage Title
st.title("서울시 버스파업, 그 이면에 대하여")
st.markdown("<h5>25-1 데이터저널리즘 3조 : 김경민, 김명원, 이유림</h5>", unsafe_allow_html=True)  # subheader보다 작음


# ================================================================================
# ==========================이런 문자, 본 적 있나요?=================================
# ================================================================================

st.image("이미지/message.png", use_container_width=True)
st.markdown(
    """
<div style='text-align: center; white-space: pre-wrap; line-height: 2;'>
이런 문자, 본 적 있나요?
2025년 5월 27일 밤 9시 경, 서울 시민들에게 재난 문자가 도착했습니다.
서울을 오가는 시내버스들이 파업에 돌입한 상황.
</div>
""",
    unsafe_allow_html=True
)
st.image("이미지/many_articles.png",use_container_width=True)

st.markdown(
    """
<div style='text-align: center; white-space: pre-wrap; line-height: 2;'>
다음 날 아침 수 백만 명의 출근길과 등굣길에 극심한 혼란이 예고되었습니다.
</div>
""",
    unsafe_allow_html=True
)

st.image("이미지/thinking.png", use_container_width=True)

content = """
여러분은 아무 일도 없었다고요?
그럴 겁니다.

총파업 예고 후 약 4시간도 채 지나지 않아,
서울시 버스 노조가 총파업 유보를 선언했기 때문입니다.
덕분에 시민들의 평화로운 하루는 지켜졌죠.

하지만 이게 끝은 아닙니다.
이번 파업은 단순한 해프닝이 아니었습니다.

그 뒤에는 수개월, 나아가 수년 간 쌓여 온 갈등이 자리하고 있습니다.

**2023년 3월 파업부터 시작해
노사 간의 9차례 교섭,
반복된 조정 회의와 준법 투쟁,
그리고 또 다른 파업 예고와 유보까지.**

서울 시내버스 파업은 법적 해석과 임금 협상 문제가 맞물려 장기간 이어진 사회적 쟁점이었습니다.
"""

st.markdown(
    f"""
<div style="text-align: center; white-space: pre-wrap;line-height: 2;">
{content}
</div>
""",
    unsafe_allow_html=True
)



# ================================================================================
# ==========================그동안 무슨 일이 있었던 걸까요?=================================
# ============================================================================================
st.markdown(
    """
<div style='white-space: pre-wrap; line-height: 2;'>
<h4>그동안 무슨 일이 있었던 걸까요?</h4>아래 타임라인에서 그 흐름을 정리해보았습니다.
</div>
""",
    unsafe_allow_html=True
)

# 타임라인 표 시작************************************************************************************
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, DateFormatter
import numpy as np

st.markdown(
"""
<style>
/* 1) 맨 왼쪽 인덱스(0,1,2…) 열 감추기 */
.stTable table th:first-child,
.stTable table td:first-child {
display: none;
}

/* 2) 헤더 스타일 (초록 배경, 흰 글씨, 크기) */
.stTable table th {
    background-color: #53b332 !important;
    color: white !important;
    font-size: 16px !important;
    text-align: center;
    padding: 8px !important;
}

/* 3) 데이터 셀 스타일 (글씨 크기, 여백) */
.stTable table td {
    font-size: 14px !important;
    padding: 8px !important;
    vertical-align: top;
}
</style>
""",
    unsafe_allow_html=True
)



# ————————————————————————————————

# 2) 데이터 준비
events = [
    {"시점": "2024-03-27",
     "사건": "노사 조정 회의 → **협상 결렬**"},
    {"시점": "2024-03-28",
     "사건": "**오전 4시 30분, 첫 파업 돌입** → 서울 시내버스 97.6% 운행 중단"},
    {"시점": "2024-03-28",
     "사건": "**오후 3시**, 임금 4.48% 인상+명절 수당 65만 원 합의 → **정상 운행 복귀**"},
    {"시점": "2024-12 ~ 2025-05",
     "사건": "**9차례 교섭** 이어지나 **최종 합의 실패**"},
    {"시점": "2025-04-30",
     "사건": "**준법투쟁 전개** → **연휴 기간 정상 복귀**"},
    {"시점": "2025-05-07",
     "사건": "**준법 투쟁 재개**"},
    {"시점": "2025-05-27",
     "사건": "**노사 교섭 최종 결렬**"},
    {"시점": "2025-05-28",
     "사건": "**오전 12시, 총파업 돌입 예고**"},
    {"시점": "2025-05-28",
     "사건": "**오전 4시, 총파업 즉시 유보**"},
]

df = pd.DataFrame(events)

# 3) 인덱스 없이 테이블 출력
st.table(df)

# 타임라인 표 끝************************************************************************************


content = """

불과 몇 시간 안에 파업을 예고했다가 유보했다가…
노조 관계자는 이렇게 말합니다.
"""

st.markdown(
    f"""
<div style="white-space: pre-wrap;line-height: 2;">
{content}
</div>
""",
    unsafe_allow_html=True
)


quote = """
*“파업을 해도 서울시와 사측의 입장이 달라지지 않을 것이란 확신이 들어
무의미한 파업이 될 것 같았다”*
"""

st.markdown(
    f"""
<blockquote style="margin: 1em 2em; padding: 0.5em 1em; border-left: 4px solid #bbb; color: #555;">
{quote}
</blockquote>
""",
    unsafe_allow_html=True
)

content = """
양측의 의견은 평행선을 달리고 있고,
협상은 좁혀질 기미가 보이지 않습니다.
"""

st.markdown(
    f"""
<div style="white-space: pre-wrap;line-height: 2;">
{content}
</div>
""",
    unsafe_allow_html=True
)


# ================================================================================
# =========================서울시와 사측, 그리고 노조는 무엇을 두고 싸우고 있을까요?=================================
# ============================================================================================

content = """
<h4>서울시와 사측, 그리고 노조는 무엇을 두고 싸우고 있을까요?</h4> 그 중심에는 <strong>통상임금</strong>이라는 조금은 생소한 단어가 있습니다.
지금부터 차근차근 짚어보겠습니다
"""

st.markdown(
    f"""
<div style="white-space: pre-wrap;line-height: 2;">
{content}
</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    f"""
<div style="white-space: pre-wrap;line-height: 2;">
<h5>임금구성</h5>임금은 기본급, 상여금, 수당으로 구성됩니다.
</div>
""",
    unsafe_allow_html=True
)


# 임금 구조 그래프 시작************************************************************************************

import os
import sys
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

# ——— 한글 폰트 설정 ———
# Windows 경로 예시 (malgun.ttf); macOS는 '/Library/Fonts/AppleGothic.ttf'
#font_path = 'C:/Windows/Fonts/malgun.ttf'
#font_path = os.path.join(proj_dir, "KoPub Batang Medium.ttf") #상대경로입니다!!
# font_prop = fm.FontProperties(fname=font_path)
# plt.rc('font', family=font_prop.get_name())
# plt.rcParams['axes.unicode_minus'] = False  # 음수 부호 깨짐 방지

proj_dir = os.path.dirname(__file__)
if sys.platform.startswith("win"):
    font_path = os.path.join(proj_dir, "KoPub Batang Medium.ttf")
else:
    font_path = "fonts/KoPub Batang Medium.ttf"

fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family']        = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False


# 데이터
labels = ['기본급', '상여금', '수당']
values = [220, 110, 110]

# 그라데이션 컬러맵 생성 (#53b332 → 라이트 그린)
base_rgb = np.array(to_rgb('#53b332'))
light_rgb = base_rgb + (1 - base_rgb) * 0.5
cmap = LinearSegmentedColormap.from_list('grad', [base_rgb, light_rgb])

# 각 구성 요소 색상
colors = [cmap(0.0), cmap(0.5), cmap(1.0)]

# 스택형 막대그래프 그리기
fig, ax = plt.subplots()
bottom = 0
x = 0  # 단일 카테고리이므로 x 좌표를 0으로 고정
bar_width = 0.6  # 막대 너비 (0~1)

for val, color, label in zip(values, colors, labels):
    ax.bar(
        x,
        val,
        bottom=bottom,
        color=color,
        width=bar_width,  # 너비 조절
        edgecolor='none'
    )

    # 막대 중앙에 텍스트 추가
    ax.text(
        x,                   # x 위치 (카테고리 위치)
        bottom + val / 2,    # y 위치 (아래부터 반 높이 지점)
        f'{label}({val}만 원)',  # 표시할 텍스트
        ha='center',         # 가로 정렬: 중앙
        va='center',         # 세로 정렬: 중앙
        fontproperties=font_prop,
        fontsize=12,
        color='black',
        weight='bold'
    )
    bottom += val
    

# 레이블 및 제목 설정
ax.set_xticks([x])
ax.set_xticklabels(['임금 구성'])
ax.set_ylabel('금액 (만원)')

plt.tight_layout()


# — 여기서부터 columns 레이아웃 적용 —
col1, col2 = st.columns(2)

with col1:
    # 왼쪽 컬럼에 차트
    st.pyplot(fig)

with col2:
    st.markdown("###### 💡위 범례(기본급/상여금/수당)을 클릭하면 해당 항목에 대한 설명이 나와요!")
    selected_item = st.radio("구성 요소",
                            ("기본급", "상여금", "수당"),
                            label_visibility="collapsed")	
    if selected_item == "기본급":
        st.write("👉 **월 정기 지급되는 고정급여입니다.**")
    elif selected_item == "상여금":
        st.write("👉 **기본급 외에 추가적으로 지급되는 급여입니다. 근로 계약 혹은 회사 규정에 따라 일정한 시기에 모든 근로자에게 일률적으로 지급됩니다.**")
    elif selected_item == "수당":
        st.write("👉 **근무 시간, 특수 상황, 특정 업무에 따라 지급되는 추가 급여입니다. (예: 야근 수당, 휴일 수당 등)**")

# 임금 구조 그래프 끝************************************************************************************

content = """
<h5>통상임금</h5>통상임금은 <strong>근로자가 근로계약에 따라 정기적이고 일률적으로 지급받는 임금</strong>으로,
<strong>연장근로 수당, 야간근로 수당, 휴일근로 수당 계산의 기준이 되는 중요한 지표</strong>입니다. 
기본적으로 통상임금에 할증율을 적용하여 수당을 계산하는데, 예시를 들어보겠습니다.
"""

st.markdown(
    f"""
<div style="white-space: pre-wrap;line-height: 2;">
{content}
</div>
""",
    unsafe_allow_html=True
)

st.image("이미지/day_night.png", use_container_width=True)

st.markdown(
    """
<div style="
    background-color: #f0f9f0;
    border-left: 4px solid #53b332;
    padding: 12px 16px;
    border-radius: 6px;
    margin: 12px 0;
    line-height: 2;
    white-space: pre-wrap;
">오후 10시부터 오전 06시 사이, 야간 근로를 할 시에는 ‘통상 임금’의 50%를 가산하여 받게 됩니다.
이 말인 즉, 원래 일 통상 임금이 10,000원으로 책정되는 사람이 오후 10시부터 오전 1시까지 3시간을 일하게 되면, (3시간 + 3시간*0.5) * 10,000 = 총 45,000원을 받게 된다는 뜻입니다.
</div>
""",
    unsafe_allow_html=True
)

content = """
기존에는 기본급, 상여금, 수당 중 기본급만 통상 임금에 포함되어 있었습니다.
그런데 2024년 12월 19일, 이 모든 것을 뒤바꾼 사건이 발생합니다.

그것은 바로,
**통상임금에 ‘상여금’을 포함해야 한다는 대법원 판결**이 나온 것입니다.
이는 임금 구조에 큰 변화를 불러오게 되죠.
"""

st.markdown(
    f"""
<div style="white-space: pre-wrap;line-height: 2;">
{content}
</div>
""",
    unsafe_allow_html=True
)


# 변화 전 vs 변화 후 그래프 라디오 버튼 시작**************************************************
st.markdown("##### 변화 전후 통상임금 해당 범위")

# 1) 세션 스테이트로 mode 기본값 세팅
if 'mode' not in st.session_state:
    st.session_state['mode'] = '변화 전'

mode = st.session_state['mode']  # 이전에 선택된 값(또는 기본값)을 사용

# 하이라이트할 세그먼트 결정
if mode == "변화 전":
    highlight = ['기본급']
else:
    highlight = ['기본급', '상여금']


# 데이터
labels = ['기본급', '상여금', '수당']
values = [220, 110, 110]

# 그라데이션 컬러맵 생성 (#53b332 → 라이트 그린)
base_rgb = np.array(to_rgb('#53b332'))
light_rgb = base_rgb + (1 - base_rgb) * 0.5
cmap = LinearSegmentedColormap.from_list('grad', [base_rgb, light_rgb])

# 각 구성 요소 색상
orig_colors = [cmap(0.0), cmap(0.5), cmap(1.0)]
highlight_color = '#FC5230'

# 스택형 막대 그리기
fig, ax = plt.subplots()
bottom = 0
x = 0
bar_width = 0.6
for val, orig_col, label in zip(values, orig_colors, labels):
    col = highlight_color if label in highlight else orig_col
    txt_color = 'white' if label in highlight else 'black'
    ax.bar(x, val, bottom=bottom, color=col, width=bar_width, edgecolor='none')
    ax.text(
        x, bottom + val/2,
        f'{label}({val}만 원)',
        ha='center', va='center',
        fontproperties=font_prop, fontsize=12,
        color=txt_color, weight='bold'
    )
    bottom += val
    

# 레이블 및 제목 설정
ax.set_xticks([x])
ax.set_xticklabels(['임금 구성'])
ax.set_ylabel('금액 (만원)')

plt.tight_layout()

# — 여기서부터 columns 레이아웃 적용 —
col1, col2 = st.columns(2)

with col1:
    # 왼쪽 컬럼에 차트
    st.pyplot(fig)

with col2:
    st.markdown("###### 💡변화 전/변화 후를 클릭하여 통상임금 범위를 비교해보세요!")

    # 2) 라디오 버튼을 차트 아래에 렌더링 (key='mode'로 세션 값에 연결)
    st.radio(
        "변화 전후 선택",
        ("변화 전", "변화 후"),
        key='mode',
        label_visibility="collapsed"
    )

# 변화 전 vs 변화 후 그래프 라디오 버튼 끝**************************************************


content = """
통상임금에 따라 수당이 책정되는 만큼,
연장근로나 야간근로가 많아 수당의 비율이 높은 직무의 경우에는
상여금이 포함됨에 따라 자연스레 수당이 높게 책정됩니다.
"""

st.markdown(
    f"""
<div style="white-space: pre-wrap;line-height: 2.5;">
{content}
</div>
""",
    unsafe_allow_html=True
)


# ================================================================================
# =========================대법원은 왜 통상 임금의 범위를 확대했을까요?=================================
# ============================================================================================

content = """
<h4>대법원은 왜 통상 임금의 범위를 확대했을까요?</h4>결론부터 말하자면, 초과근로수당을 아끼려는 기업의 꼼수를 막기 위해서 입니다.
이전까지 노동현장에는 임금 항목 대부분에 조건을 붙여 통상 임금을 낮추는 수법이 존재했습니다.
다시 말해, 통상임금을 기준으로 수당이 정해지니까,
통상 임금을 낮춤으로써 수당 또한 낮추고자 했던 것이죠.
이 때문에 월 통상임금이 지나치게 낮은 기형적 임금체계가 존재했습니다.

버스 기사의 임금 구조를 살펴봅시다.
"""

st.markdown(
    f"""
<div style="white-space: pre-wrap;line-height: 2;">
{content}
</div>
""",
    unsafe_allow_html=True
)

# st.image("wage.png", use_container_width=True)

st.markdown(
    """<div style="
    background-color: #f0f9f0;
    border-left: 4px solid #53b332;
    padding: 12px 16px;
    border-radius: 6px;
    margin: 12px 0;
    line-height: 2;
    white-space: pre-wrap;
">2024년 버스 기사 4호봉 기준으로 기본급은 약 223만 원, 주휴·연장·야간 수당과 상여금 수당 등을 포함한 월 실수령 총액은 약 470만 원입니다.
</div>""",
    unsafe_allow_html=True
)

content = """
연장근로, 야간근로가 많다보니 기본급 기반의 고정임금은 적고, 수당과 상여금에 의존하는 구조이죠.
상여금을 기본급으로 포함시키면 수당이 높아지니,
받아야 할 돈을 다 상여금으로 빼놓고 수당은 기본급을 기준으로 주고 있었던 것입니다.
"""

st.markdown(
    f"""
<div style="white-space: pre-wrap;line-height: 2.5;">
{content}
</div>
""",
    unsafe_allow_html=True
)

# st.image("article.png", use_container_width=True)


# ================================================================================
# =========================이제 대법원 판결이 적용되면?=================================
# ============================================================================================

st.markdown("#### 이제 대법원 판결이 적용되면?")


# 변화 전후 그래프 슬라이더 시작************************************************************************************

st.markdown("##### 변화 전후 수당 비교")

# 1) session_state 초기화
if 'overtime' not in st.session_state:
    st.session_state.overtime = 75

# 2) 내부 로직에 사용할 값 가져오기
slider_val = st.session_state.overtime

# 카테고리 & 데이터
labels      = ['기본급', '상여금', '수당']
base_pre  = [220, 110, 0]  # 변화 전
base_post = [220, 110, 0]  # 변화 후

# 변화 전 수당: slider 비율로 0~110
pre_allowance  = int(slider_val * 110 / 130)
# 변화 후 수당: slider 비율로 0~200
post_allowance = int(slider_val * 200 / 130)

# 스택형 값 리스트 완성
values_pre  = [base_pre[0],  base_pre[1],  pre_allowance]
values_post = [base_post[0], base_post[1], post_allowance]

# 그라데이션 컬러맵 (#53b332 → 라이트 그린)
base_rgb  = np.array(to_rgb('#53b332'))
light_rgb = base_rgb + (1 - base_rgb) * 0.5
cmap      = LinearSegmentedColormap.from_list('grad', [base_rgb, light_rgb])
colors    = [cmap(0.0), cmap(0.5), cmap(1.0)]

# 막대 위치 & 너비
positions = [0, 1]
bar_width = 0.6

# Figure 생성
fig, ax = plt.subplots()

# 두 개 스택형 막대 그리기
for pos, vals in zip(positions, [values_pre, values_post]):
    bottom = 0
    for val, color, label in zip(vals, colors, labels):
        ax.bar(
            pos,
            val,
            bottom=bottom,
            width=bar_width,
            color=color,
            edgecolor='none'
        )
        ax.text(
            pos,
            bottom + val / 2,
            f'{label}({val}만 원)',
            ha='center',
            va='center',
            fontproperties=font_prop,
            fontsize=11,
            color='black',
            weight='bold'
        )
        bottom += val

# 축 레이블 설정
ax.set_xticks(positions)
ax.set_xticklabels(['변화 전', '변화 후'])
ax.set_ylabel('금액 (만원)')

# y축을 0~600으로 고정하고, 100단위로 눈금 표시
ax.set_ylim(0, 600)
ax.set_yticks(range(0, 601, 100))

plt.tight_layout()
st.pyplot(fig)

# 4) 그 아래에 슬라이더
st.slider(
    "추가 근무 시간",
    min_value=75,
    max_value=130,
    value=75,
    key='overtime'
)

slider_val = st.session_state["overtime"]
st.write(f"###### ⏰월 추가 근무 시간이 {slider_val}시간일 때")


# 변화 전후 그래프 슬라이더 끝************************************************************************************


content = """상여금이 모두 통상임금에 포함되면서 수당 역시 상여금 포함분을 기준으로 책정됩니다.
이제 버스 기사들의 수당 증가분이 가파르게 오르게 되겠죠.
"""

st.markdown(
    f"""
<div style="white-space: pre-wrap;line-height: 2;">
{content}
</div>
""",
    unsafe_allow_html=True
)


# ================================================================================
# =========================서울시와 사측은?=================================
# ============================================================================================

content = """
<h4>서울시와 사측은?</h4>판결에 따른 급격한 임금 총액 상승은 감당하기 어렵다는 입장입니다.
현실적인 재정 여력과 운영 효율성도 감안해야 한다는 것이죠.
이들은 노조 요구안을 수용하면 다음과 같은 결과가 나올 것이라 보았습니다.
"""


# 노조 요구안 수용 시 주요 지표 변화 시작************************************************************************************

import streamlit as st

st.markdown(
    f"""
<div style="white-space: pre-wrap;line-height: 2;">
<h5>노조 요구안 수용 시 주요 지표 변화</h5><strong>👉 보고 싶은 지표를 선택하세요</strong>
""",
    unsafe_allow_html=True
)

# 1) 선택박스 항목
options = [
    "운수종사자 평균 연봉",
    "서울시 연간 재정지원금",
    "운전직 인건비 총액",
    "서울시민 1인당 세금부담액",
    "버스요금"
]

# 2) 각 항목별 변화 매핑
changes = {
    "운수종사자 평균 연봉" : "6,273만 → 7,872만",
    "서울시 연간 재정지원금" : "5,459억 → 8,259억",
    "운전직 인건비 총액" : "9,500억 → 1조 6,180억",
    "서울시민 1인당 세금부담액": "55,000원 → 85,000원",
    "버스요금": "1,500원 → 1,800원"
}

# 3) 선택박스 렌더링
selection = st.selectbox("",options)

# 4) 결과 표시
st.markdown(f"**{selection}**: {changes[selection]}")

# 노조 요구안 수용 시 주요 지표 변화 끝************************************************************************************

content = """
특히 서울시는 준공영제 아래 매년 수천억 원의 재정 지원을 하고 있기 때문에 통상임금 확대가 수당을 밀어올리는 구조가 될 경우, 재정 부담이 급격히 늘어난다고 주장합니다.
이에 따라 <strong>'임금 총액을 유지하고 추후 기본급 인상을 논의하자'</strong>고 말합니다.
즉, 상여금이 통상임금에 포함되더라도, 기본급을 줄이거나 수당 구조를 조정해 전체 총액은 동일하게 맞추겠다는 방향입니다.

노조는 이 방식을 <strong>‘사실상 판결 무력화’</strong>라고 보고 있습니다.
상여금을 넣어주면서 기본급이나 근무시간을 줄이는 건,
결국 받을 돈은 그대로 두고, 구조만 바꿔본 셈이라는 것이죠.

결국, 법의 취지를 따를 것인가,
재정 현실에 맞게 조율할 것인가
서울시와 사측, 그리고 노조는 여전히 해결되지 않은 평행선 위에 서 있습니다.
"""

st.markdown(
    f"""
<div style="white-space: pre-wrap;line-height: 2.5;">
{content}
</div>
""",
    unsafe_allow_html=True
)


st.image("이미지/conflict.png", use_container_width=True)


content = """
우리는 궁금해졌습니다.
언론은 이 사태를 어떻게 설명하고 있을까요?
"""

st.markdown(
    f"""
<div style="white-space: pre-wrap;line-height: 2.5;">
{content}
</div>
""",
    unsafe_allow_html=True
)

#####################명원 파트###################
import streamlit as st

# 제목
st.markdown("""
    <h2 style='margin-bottom: 0px;'>언론은 버스 파업 사태를 어떻게 조명했나?</h2>
    <hr style='margin-top: 0px;'>
""", unsafe_allow_html=True)



st.markdown("""
서울 시내버스 노조와 지자체 간에 통상 임금을 둘러싼 의견 차이가 결국 파업이라는 극단적 상황으로 치달았다. 그렇다면 이 같은 상황을 기성 언론사들은 어떤 시선으로 보도했을까. 본지는 **2024년 12월 4일부터 2025년 6월 3일까지 총 6개월간** 일간지 및 주요 경제지 **29곳의 기사 525건**을 수집해 데이터 분석을 진행해 그 차이를 조명해보기로 했다.
""")

st.markdown("""<br>""", unsafe_allow_html=True)


# 1. 헤드라인 워드클라우드
st.markdown("### 1. 헤드라인 워드클라우드: 진보는 구조, 보수는 피해에 초점")
st.image("이미지/wordcloud.png", caption="5대 언론사 헤드라인(기사 제목) 기준 워드클라우드")

st.markdown("""
우선 각 언론사가 '버스 파업'을 주제로 발행한 기사들의 제목을 형태소 단위로 빈번하게 등장한 키워드를 확인했다.  
""")

# 진보_왼쪽: 인용문, 오른쪽: 설명
col1, col2 = st.columns([1.3, 1]) 

with col1:
    st.markdown("""
    <blockquote style='padding: 15px 20px; background-color: #f9f9f9; border-left: 5px solid #999; margin: 0;'>
        <p style='margin-bottom: 10px;'>
            <span style='font-weight: bold;'>‘통상임금’ 접점 못 찾는 서울버스 전면파업 가능성</span><br>
            <span style='font-size: 0.9em; color: #666;'>(경향신문, 2025.05.07)</span>
        </p>
        <p style='margin-bottom: 0px;'>
            <span style='font-weight: bold;'>버스파업 통상임금 줄다리기 노조 “지자체가 해결해야”</span><br>
            <span style='font-size: 0.9em; color: #666;'>(한겨레, 2025.05.20)</span>
        </p>
    </blockquote>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    진보 성향 언론으로 알려진 **경향신문**과 **한겨레**는 헤드라인에서 ‘통상임금’, ‘교섭’, ‘준법투쟁’ 등 파업의 쟁점과 구조적 배경에 주목했다. 노조와 지자체 간 협상이 통상임금 문제에서 접점을 찾지 못한 채 파업으로 이어졌다는 접근이다. 
    """)

st.markdown("<br>", unsafe_allow_html=True)  # 1줄 공백

# 보수_왼쪽: 인용문, 오른쪽: 설명
col3, col4 = st.columns([1.3, 1])  # 왼쪽: 인용문, 오른쪽: 설명

with col3:
    st.markdown("""
    <blockquote style='padding: 15px 20px; background-color: #f9f9f9; border-left: 5px solid #999; margin: 0;'>
        <p style='margin-bottom: 10px;'>
            <span style='font-weight: bold;'>‘서울 버스 임단협 막판까지 진통 출근길 대란 우려</span><br>
            <span style='font-size: 0.9em; color: #666;'>(동아일보, 2025.05.28)</span>
        </p>
        <p style='margin-bottom: 10px;'>
            <span style='font-weight: bold;'>“버스 준법투쟁? <br>몰랐어요 큰일이네” 서울 출근길 시민들 당황업”</span><br>
            <span style='font-size: 0.9em; color: #666;'>(중앙일보, 2025.04.30)</span>
        </p>
        <p style='margin-bottom: 0px;'>
            <span style='font-weight: bold;'>초봉 5400만원 버스기사들의 파업”</span><br>
            <span style='font-size: 0.9em; color: #666;'>(조선일보, 2025.05.13)</span>
        </p>
    </blockquote>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    반면 보수 성향 언론인 **조선일보, 동아일보, 중앙일보**는 ‘출근길’, ‘시민 불편’, ‘대란’과 같은 키워드를 중심으로 파업이 초해할 피해를 강조했다. 특히 조선일보는 버스 기사 연봉 수준을 부각하며 이번 파업이 노조 측의 과도한 임금 인상 요구에서 기인했다는 인상을 심어주는 등, 노조의 정당성보다는 파업의 파급력과 부담에 초점을 맞추는 경향을 보였다.
    """)

st.markdown("""<br>이처럼 워드클라우드 분석 결과는 정치적 성향에 따라 언론이 어떤 문제를 ‘첫 문장’에서 가장 먼저 보여주는지를 명확히 드러낸다. 진보는 원인과 구조, 보수는 결과와 혼란을 이야기했다.""", unsafe_allow_html=True)
st.markdown("""<br>""", unsafe_allow_html=True)


# 2. 본문 첫 문단 TF-IDF
st.markdown("### 2. 본문 첫 문단 TF-IDF: 정치 성향에 따른 입장 차이는 여전")
st.image("이미지/tf_idf.png", caption="5대 언론사의 본문 첫 5문장 기준 주요 키워드의 TF-IDF 비교")

st.markdown("""
다음으로 기사 본문의 초반부 역시 독자에게 언론이 전달하고자 하는 핵심 메시지가 집약되는 지점이다. 이에 따라 본문 첫 5문장을 중심으로 **TF-IDF 분석**을 진행했다.TF-IDF는 특정 문서에서 자주 등장하지만 다른 문서에서는 상대적으로 덜 등장하는 단어를 높은 점수로 평가해, 각 문서에서 중요한 단어를 찾아주는 방법이다. 여기서는 **각 언론사마다 중점적으로 다루는 키워드가 무엇인지**를 의미한다.

**진보 성향 언론**(경향신문·한겨레)은 **‘준법투쟁’** 과 **‘노사’** 라는 키워드에서 높은 점수를 기록해 헤드라인과 마찬가지로 **파업의 구조적 문제와 노조 측 입장**을 강조했다. 특히 준법투쟁은 법적 테두리 안에서 노동권을 행사하는 전략을 의미한다. 곧 노조의 집단 행동이 단순히 운행을 지연시키는 데 있지 않고, **정당한 투쟁 방식**으로 본인들의 권리를 찾으려는 태도로 해석할 수 있다. 또한 노사라는 키워드는 사태를 **개인이나 조직의 일탈이 아니라** 양측이 정당한 위치에서 협상하는 문제라는 점을 드러려는 의도로 보인다.

반면 **보수 언론**(조선·동아·중앙일보)은 **‘인상’**, **‘지연’**, **‘출근길’** 등의 키워드를 통해 시민 피해와 **파업으로 인한 사회적 혼란을 마찬가지로 부각했다. 무엇보다 앞서 제목에서도 파업을 향해 가장 공격적인 논조를 취했던 **조선일보는**, 노사 협상의 핵심 쟁점인 ‘통상임금’을 거의 언급하지 않은 점이 눈에 띈다.

이처럼 제목과 본문 초반부에서 언론사마다 버스 파업을 두고 바라보는 시선 차이가 명확하게 구분되었다.
""", unsafe_allow_html=True)
st.markdown("""<br>""", unsafe_allow_html=True)

# 3. 본문 전체 토픽 모델링
### 토픽 개수 설정
st.markdown("### 3. 본문 전체 토픽 모델링: 그러나 기사의 주제는 대동소이하다")
col5, col6 = st.columns([1.5, 2])  

with col5:
    st.image("이미지/topic_num.png", caption="Coherence Score와 Perplexity로 정한 토픽 개수")

with col6:
    st.markdown("""
    그렇다면 헤드라인과 기사 초반부에서 뚜렷하게 드러난 **정치적 성향에 따른 논조 차이**는 **기사 전체에서도 동일하게 유지될까?**  

    최근 언론 보도의 **편향성과 양극화**는 사회적 우려를 불러일으킨다. 진보든 보수든, 지지 성향에 부합하는 뉴스라면 그 진위 여부에 관계없이 수용하는 경향이 강해지고 있기 때문이다. 특히 노동 문제는 한국 사회에서 정파 갈등의 대표적 이슈로 다뤄져 왔다는 점에서, 이번 **서울 시내버스 파업 보도 역시 유사한 경향을 보였는지** 살펴보고자 했다.
    """, unsafe_allow_html=True)


st.markdown("""
이에 본지는 기사 전체를 대상으로 **LDA(Latent Dirichlet Allocation) 기반 토픽 모델링**을 실시했다. LDA는 문서 집합에서 단어 분포를 바탕으로 **주제를 자동 분류**하는 분석 기법으로, 각 기사가 다루는 내용을 설정된 개수의 주제로 유형화할 수 있다. 그렇게 525건의 기사를 분류한 결과, 다음 네 가지 주제가 도출되었다.

""", unsafe_allow_html=True)

st.markdown("""
<p style='font-size: 0.85em; color: #666; margin: 5px;'>
※ 기사 본문은 Komoran 형태소 분석기를 활용해 <strong>형태소 단위로 토큰화</strong>한 뒤, 
명사, 형용사, 동사, 수사, 부사, 관형어를 중심으로 주요 어휘를 선별했다.  
서울, 버스 등 보편적 배경 정보와 언론사·기자 관련 고유 표현은 <strong>불용어로 사전 제거</strong>해 주제 분류의 정확도를 높였다.
</p>

<p style='font-size: 0.85em; color: #666; margin: 5px;'>
※ 토픽 개수는 <strong>Coherence Score와 Perplexity</strong> 지표를 비교 분석한 결과, 
<strong>의미론적 일관성과 분류 성능이 균형을 이루는 지점</strong>인 <strong>4개로 설정</strong>했다.  
</p>
""", unsafe_allow_html=True)

st.markdown("""<br>""", unsafe_allow_html=True)

### 토픽 도출 결과
st.markdown("#### 토픽 분석 결과")
col7, col8 = st.columns([1.7, 2.9])
with col7:
  st.image("이미지/topic_result.png", caption="(좌) 토픽별 상위(주요) 키워드")

with col8:
  st.image("이미지/topic_text.png", caption="(우) 각 토픽 설명 및 의미 부여")

st.markdown("""
<div style='margin-bottom: 8px;'><u><strong>첫 번째 토픽. 전국 동시 파업 및 연대 움직임</strong></u></div>

<p style='margin: 0 0 8px 0;'>
전국, 조정, 공동, 동시, 지역 교섭 등의 키워드가 중심을 이뤘다. 시내 버스 파업이 서울시만의 일이 아니라 <strong>전국 주요 도시 노조 간의 조직적 연대 속에서</strong> 진행되고 있음을 보여준다. 실제로 서울 외에도 부산, 창원, 광주, 울산 등 광역시를 중심으로 서울과 비슷한 노사 갈등이 벌어졌고, 알부는 실제 파업으로까지 이어졌었다. 해당 토픽도 지자체 간 공동 대응 회의나 운수노조 연맹 차원의 파업 일정 조율 등이 비중 있게 다뤄졌다. 
</p>

<div style='padding: 12px 15px; background-color: #f9f9f9; border-left: 5px solid #999; margin: 10px 0;'>

  <div style='margin-bottom: 10px;'>
    <p style='margin: 0;'><strong>“서울 등 22곳 버스노조 "교섭결렬시 28일 총파업" 전국확산 우려”</strong></p>
    <small style='color: #666;'>(<a href='http://www.segye.com/content/html/2025/05/08/20250508510451.html'>서울경제</a>, 2025.05.08)</small><br>
    <span style='font-size: 0.85em; color: #666;'>한국노총 자동차노련은 8일 오전 서울 양재동 회의실에서 전국 대표자회의를 열고 이 같은 회의 결과를 발표했다. (...) 서울, 충북, 울산, 경남 등 전국 버스노조 위원장과 실무자 등 20여명이 참석한 가운데 약 1시간 동안 진행됐다.</span>
  </div>

  <div>
    <p style='margin: 0;'><strong>“28일 전국버스 '총파업' 예고...22개 지역 노조 동시조정”</strong></p>
    <small style='color: #666;'>(<a href='http://www.fnnews.com/news/202505081337451055' target='_blank'>파이낸셜 뉴스</a>, 2025.05.08)</small><br>
    <span style='font-size: 0.85em; color: #666;'>서울, 부산, 인천, 경기 등 22개 지역 전국자동차노동조합연맹 산하 시내버스 노조가 노사교섭 결렬 시 오는 28일 동시 총파업을 예고했다.</span>
  </div>

</div>
""", unsafe_allow_html=True)
st.markdown("""<br>""", unsafe_allow_html=True)

st.markdown("""
<div style='margin-bottom: 8px;'><u><strong>두 번째 토픽. 통상임금 및 임금 체계 개선 논의</strong></u></div>

<p style='margin: 0 0 8px 0;'>
임금, 통상, 상여금, 협상, 인상, 체계, 대법원, 개편, 판결, 기본급, 인건비 등의 키워드가 중심을 이뤘다.  
이번 파업의 핵심 쟁점은 단순한 임금 인상 요구가 아니라 기본급 외에 <strong>상여금을 통상임금에 얼마나 산정할지에 있다는 점</strong>을 다룬다. 또한, 최근 상여금을 통상임금으로 볼 수 있다는 대법원 판결이 이번 협상에 노조 측에 힘을 실어준다는 시각도 제시한다.
그렇기에 다수의 기사에서 대법원 판결 이후 상여금 포함 여부를 두고 벌어진 노사 간 입장 차이를 밝히고, 양측 입장이 평행선을 이루는 대치 상황을 부각했다.
</p>

<div style='padding: 12px 15px; background-color: #f9f9f9; border-left: 5px solid #999; margin: 10px 0;'>

  <div style='margin-bottom: 10px;'>
    <p style='margin: 0;'><strong>“월급쟁이가 서울 버스기사 통상임금 분쟁을 주목해야할 이유”</strong></p>
    <small style='color: #666;'>(<a href='http://www.hani.co.kr/arti/society/labor/1200393.html' target='_blank'>한겨레</a>, 2025.05.31)</small><br>
    <span style='font-size: 0.85em; color: #666;'>결국 임금체계를 개편하더라도 어떻게 개편할지가 중요한 셈입니다. 서울시와 사업조합이 상여금 600% 가운데 얼마를 통상임금로 반영할지 ‘안’을 제시하고 이를 바탕으로 논의하는 게 순리일 것입니다.</span>
  </div>

  <div>
    <p style='margin: 0;'><strong>“서울시내버스노조 ‘통상임금 재산정 포기 요구 절대 수용 불가’”</strong></p>
    <small style='color: #666;'>(<a href='http://www.mk.co.kr/article/11316885' target='_blank'>매일경제</a>, 2025.05.14)</small><br>
    <span style='font-size: 0.85em; color: #666;'>시는 통상임금 증액에 따른 충격 완화를 위해 임단협에서 성과연봉제 전환 등 임금 체계 개편이 필요하다는 입장이다. 반면, 노조는 작년 대법원 판례에 따라 (...) 임금삭감을 목적으로 하는 임금체계 개편은 받아들일 수 없다고 맞서 노사 협상에 난항이 예상된다.</span>
  </div>

</div>
""", unsafe_allow_html=True)
st.markdown("""<br>""", unsafe_allow_html=True)

st.markdown("""
<div style='margin-bottom: 8px;'><u><strong>세 번째 토픽. 시민 불편 및 수송 대책 보도</strong></u></div>

<p style='margin: 0 0 8px 0;'>
운행, 준법투쟁, 시간, 불편, 지하철, 시민, 교통, 출근, 지연 등의 키워드로 이루어진 주제이다.  
<strong>시민 일상과 직결된 출근길 혼잡</strong>에 주목하해 준법투쟁 방식의 파업이 어떤 영향을 주었는지를 다룬 기사들이 주를 이뤘다.  
특히 지하철로 몰린 시민들의 반응을 비롯해 서울시의 셔틀버스 운행 등 자자체에서 마련한 대체 교통 수단이 주요하게 보도되었다.
</p>

<div style='padding: 12px 15px; background-color: #f9f9f9; border-left: 5px solid #999; margin: 10px 0;'>

  <div style='margin-bottom: 10px;'>
    <p style='margin: 0;'><strong>“'20분 일찍 나와' 출근길 지옥철서 '낑낑' 서울버스 준법투쟁 재개”</strong></p>
    <small style='color: #666;'>(<a href='http://news.moneytoday.co.kr/view/mtview.php?no=2025050709323667688&type=2' target='_blank'>머니투데이</a>, 2025.05.07)</small><br>
    <span style='font-size: 0.85em; color: #666;'>서울시는 준법투쟁 재개에 따라 시민 불편 최소화를 위한 특별 교통 대책을 시행했다. 지난달 30일과 마찬가지로 지하철 출근 주요 혼잡 시간을 오전 7~10시로 1시간 확대 운영하고, 1~8호선과 우이신설선 열차 투입 횟수를 47회 늘렸다.</span>
  </div>

  <div>
    <p style='margin: 0;'><strong>“준법투쟁 시작한 서울 시내버스... '걱정한 만큼 불편은 없어'”</strong></p>
    <small style='color: #666;'>(<a href='https://www.hankookilbo.com/News/Read/A2025043010090004682' target='_blank'>한국일보</a>, 2025.04.30)</small><br>
    <span style='font-size: 0.85em; color: #666;'>성동구에서 여의도로 출근하는 직장인 권모(40)씨도 "혹시 출근에 지장이 생길까 봐 평소보다 일찍 나와 지도 앱으로 배차 간격을 계속 확인하고 있다"고 했다. (...) 직장인 김해성(29)씨는 "당장 어제보다 지하철에 사람이 20~30% 많아 이동하는 내내 힘들었다"며 "평소 우르르 내리는 환승역에서도 승객이 빠지지 않아 열차 안이 매우 혼잡했다"고 전했다.</span>
  </div>

</div>
""", unsafe_allow_html=True)
st.markdown("""<br>""", unsafe_allow_html=True)

st.markdown("""
<div style='margin-bottom: 8px;'><u><strong>네 번째 토픽. 협상 진행 및 파업 일정 중심 설명</strong></u></div>

<p style='margin: 0 0 8px 0;'>
파업, 협상, 결렬, 유보, 예고, 돌입, 총파업, 교섭 등의 키워드가 중심을 이뤘다.  
<strong>파업 전후의 협상 흐름</strong> 즉 돌입 예고, 유보, 재개, 결렬 등 시계열적인 갈등 전개를 집중 보도한 기사들이 포함되었다.  
특히 파업 직전까지 이어진 막판 교섭과 그에 따른 유보·타결 여부 보도에서 이 주제가 두드러졌다. '28일'이 파업 예고일 이있던 만큼 해당 키워드가 눈에 띈다.
</p>

<div style='padding: 12px 15px; background-color: #f9f9f9; border-left: 5px solid #999; margin: 10px 0;'>

  <div style='margin-bottom: 10px;'>
    <p style='margin: 0;'><strong>“서울시 ‘시내버스 파업, 3일 이상 될 수도… 총력 대응’”</strong></p>
    <small style='color: #666;'>(<a href='http://www.edaily.co.kr/news/newspath.asp?newsid=02830646642172856' target='_blank'>이데일리</a>, 2025.05.26)</small><br>
    <span style='font-size: 0.85em; color: #666;'>서울시버스노동조합이 속한 한국노총 전국자동차노조연맹은 오는 27일까지 임금·단체협약(임단협) 협상 합의안이 도출되지 않으면 28일 첫차부터 전국 동시 파업에 돌입하겠다고 예고한 바 있다. 또 서울 시내버스 노사는 (...) 이견이 커 본교섭을 재개하지 못한 상황이다.

노조는 오는 27일 오후 1시에 교섭을 재개하자고 이날 오전 사측(서울시버스운송사업조합)에 공문을 보냈지만 아직 일정이 잡히지는 않았다.</span>
  </div>

  <div>
    <p style='margin: 0;'><strong>“서울시 버스노조는 파업 유보… ‘통상임금 문제, 법원 판단 기다릴 것’”</strong></p>
    <small style='color: #666;'>(<a href='http://www.segye.com/content/html/2025/05/28/20250528513218.html' target='_blank'>세계일보</a>, 2025.05.28)</small><br>
    <span style='font-size: 0.85em; color: #666;'>서울시버스노동조합은 28일 오전 2시 지부장 총회 투표 결과 재적인원 63명 중 49명이 파업 유보에 투표했다고 밝혔다. 앞서 시내버스 노사는 전날 오후 3시부터 9시간동안 ‘마라톤 협상’을 진행했으나 이견이 좁혀지지 않아 결국 협상이 결렬됐다. (...)</span>
  </div>

</div>
""", unsafe_allow_html=True)
st.markdown("""<br>""", unsafe_allow_html=True)

### 언론사별 토픽 분포
st.markdown("")
st.markdown("#### 언론사별 토픽 분포")
st.image("이미지/topic_rate.png", caption="5대 언론사의 총 발행 기사 대비 각 토픽의 비중")

st.markdown("""
각 언론사별로 발행한 기사 중 각 토픽이 차지하는 비중을 살펴봤을 때 흥미로운 지점이 발견되었다. 특히 헤드라인과 기사 초반부에서 선명하게 드러났던 정치적 성향에 따른 논조 차이와 달리, 전체 내용에서는 **언론사의 성향과 무관하게 네 가지 주제를 비교적 균형 있게 다루는 경향**이 나타났다. 

**조선일보**는 앞서 서두에서 노조의 요구를 자극적으로 비판했지만, **본문 중반 이후에는 통상임금 쟁점을 상대적으로 중립적으로 설명하기도** 했다. 실제로 조선일보는 전체 기사 중 33.3%에서 통상임금을, 26.7%에서 시민 불편을 다루며 주요 쟁점들을 병렬적으로 배치하는 보도 경향을 보였다. 예컨대 해당 <a href='https://www.chosun.com/national/national_general/2025/05/07/O45SVT5COBHSPC3S2DC7NXH7OU/?utm_source=bigkinds&utm_medium=original&utm_campaign=news' target='_blank'>기사 후반부</a>에서 “서울 시내버스 노사는 정기 상여금을 통상임금에 넣는 문제를 두고 입장 차를 좁히지 못하고 있다”며, 노조는 대법원 판결을 근거로 포함을 주장하고, 사측은 임금체계 개편을 전제로 협상을 요구하는 상황을 전했다.

반대의 입장을 취한다고 여겨진 언론사간에 유사한 비중을 보이는 경우도 있다. 가령 **경향신문(34.6%)과 동아일보(43.8%)는 파업 전후의 협상 경과에 가장 많은 비중**을 두고, 전개 과정에 대한 시간순 서술을 통해 독자의 이해를 도왔다.

**중앙일보(57.1%)와 한겨레(61.1%)는 통상임금 및 임금체계 개편 논의를 가장 집중적으로** 다뤘으며, 모두 임금 체계의 구조적 쟁점을 주요 의제로 부각했다는 점에서 유사한 보도 태도를 보였다. 물론 주제가 비슷하더라도 서술 방식에서 관점은 엇갈리긴 했다. 중앙일보는 사측 입장을 중심으로 ‘통상임금 포함 시 20% 이상 임금 상승’과 서울시의 재정 부담을 강조한 반면, 한겨레는 상여금 포함 여부를 둘러싼 대법원 판례의 해석 문제를 협상의 핵심으로 제시했다.

그럼에도 보이는 바와 같은 토픽 비중이 우리에게 던지는 메시지는 분명하다. 각 언론이 도입부에서 독자의 주의를 끌기 위해 정치적 프레임을 강화하면서도, 본문에서는 보도의 균형성과 설명력을 확보하려는 태도를 보였다는 점이다. 헤드라인은 정파적 색채가 강할지 몰라도 전체 기사 구성은 파업의 주요 화두가 무엇인지, 노사가 합의에 이르지 못해 초래되는 여파 및 지자체의 행정 대응과 같은 정보를 제공하는 데 집중했음을 드러낸다. 그렇기에 **정치적 성향이 뉴스 소비 방식에 큰 영향을 미치는 시대일수록, 자극적인 표현이나 초반 인상에만 의존하기보다는 전체 맥락과 주제 분포를 함께 살펴보려는 노력이 더욱 중요해졌다고 할 수 있다.**
""", unsafe_allow_html=True)

# 나가며
st.markdown("""<br>""", unsafe_allow_html=True)
st.markdown("### 그러나 표면적인 보도에 가려진 본질적인 질문들")

st.markdown("""
    <p style='margin-bottom: 15px;'>주제 분포를 따라가며 각 언론의 보도 경향을 분석했지만, 여전히 독자 입장에서 해결되지 않는 의문들이 남아 있다.</p>
""", unsafe_allow_html=True)

st.markdown("""
<p style='margin-bottom: 7px; font-weight: bold;'>► 왜 통상임금이 이토록 첨예한 갈등의 중심에 있는가?</p>
<p style='margin-bottom: 7px; font-weight: bold;'>► 노조는 실제로 어떤 근거로 임금 인상을 주장하며, 사측이 주장하는 '고연봉'은 진짜일까?</p>
<p style='margin-bottom: 7px; font-weight: bold;'>► 전국 연대를 결성할 만큼 버스 노동자들은 무엇에 위협을 느끼고 있는가?</p>
<p style='margin-bottom: 15px; font-weight: bold;'>► 시민 불편만을 이야기하기 전에, 그 불편이 반복되는 이유는 무엇인가?</p>
""", unsafe_allow_html=True)

st.markdown("""
이러한 질문들에 대해 어떤 언론도 충분한 맥락이나 통찰을 제공하지 못했다는 사실이, 이번 분석에서 가장 명확하게 드러난 지점이었다. 본지는 이러한 한계를 보완하기 위해, **노조의 성명문과 서울시의 공식 자료를 비교 분석하고, 관련 전문가 인터뷰를 통해 파업 사태에 감춰진 구조적 문제들을** 보다 깊이 있게 조명해보고자 한다.
""", unsafe_allow_html=True)


#############################경민###########################################


#############################경민###########################################





#############################기본급 + 수당 변화 그래프 (노조)###################################

from PIL import Image
import streamlit as st
import plotly.graph_objects as go

# 제목
st.markdown("""
    <h2 style='margin-bottom: 0px;'>서울시의 주장, 무엇이 문제일까?</h2>
    <hr style='margin-top: 0px;'>
""", unsafe_allow_html=True)

st.markdown("""


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


image = Image.open("이미지/서울시임금체계개편안.webp")
st.image(image, caption="서울시 임금체계 개편안", use_container_width=True)


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

import streamlit as st

std_hours = 176

st.markdown(
    "기본급이 **220만원 → 330만원**으로 바뀌고, "
    "소정근로시간이 176시간으로 고정일 때의 변화를 봅시다,<br>"
    "**시급과 수당이 어떻게 달라질까요?**",
    unsafe_allow_html=True
)

# 컬럼 배치 (간격 더 넉넉하게)
col1, spacer, col2 = st.columns([2.2, 0.5, 1.3])

with col1:
    selected_salary = st.radio("기본급을 선택하세요", [220, 330], horizontal=True, key="salary_case1")
    extra_hours = st.slider("추가근로시간 (시간)", min_value=0, max_value=100, step=5, value=20, key="hours_case1")

# 계산
hourly = (selected_salary * 10000) / std_hours
extra_pay = hourly * extra_hours

with col2:
    st.markdown("#### 💡 계산 결과")
    st.markdown(f"""
    <div style='border:1px solid #ccc; border-radius:10px; padding:10px 15px; background-color:#f9f9f9; font-size:15px; line-height:1.5'>
        <b>기본급:</b> {selected_salary}만원<br>
        <b>시급:</b> {int(hourly):,} 원<br>
        <b>수당:</b> <span style='color:green; font-size:18px'><b>{int(extra_pay):,} 원</b></span>
    </div>
    """, unsafe_allow_html=True)



#image = Image.open("이미지/기본급_시급.png")  # 예: "img/salary_chart.png"
#st.image(image, caption="4시간 근로 시 기본급 총액이 인상되면, 하나의 덩어리인 시급도 함께 오릅니다", use_container_width=True)

st.write("")
st.write("")
st.markdown("""
            그러나 기본급이 올랐는데 임금 총액을 유지한다는 말은 곧
            **추가 근로로 인정되는 시간**을 줄이겠다는 말과 같습니다.  
            """)
st.write("")
st.write("")
#############################이미지#################################

 # 두 번째 계산: 같은 추가근로시간으로 수당이 얼마나 달라질까?
import streamlit as st

std_hours = 176
target_allowance = st.slider("받고자 하는 수당 (원)", min_value=100_000, max_value=600_000, step=10_000, value=300_000)

st.markdown(f"""
#### 💬 기본급이 올라가도 임금 총액을 유지하겠다는 말은 무슨 뜻일까?
- 소정근로시간: **{std_hours}시간**
- 수당: **{target_allowance:,}원** (고정)
""")

col1, col2 = st.columns(2)

for col, salary in zip([col1, col2], [220, 330]):
    with col:
        hourly = (salary * 10000) / std_hours
        required_hours = target_allowance / hourly

        st.markdown(f"""
        <div style='border:1px solid #ccc; border-radius:10px; padding:15px; background-color:#f9f9f9'>
        <h5 style='text-align:center'>기본급: {salary}만원</h5>
        <p style='font-size:16px; line-height:1.6'>
        💸 <b>시급:</b> {int(hourly):,} 원<br>
        ⏱ <b>필요 추가근로시간:</b> <span style='color:crimson; font-size:20px'><b>{required_hours:.1f}시간</b></span>
        </p>
        </div>
        """, unsafe_allow_html=True)



##image = Image.open("이미지/수당_근로시간.png")  # 예: "img/salary_chart.png"
#st.image(image, caption="수당의 변화분이 없게 된다면, 이는 곧 인정되는 시간을 줄이겠다는 뜻입니다", use_container_width=True)

st.write("")
st.write("")

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
