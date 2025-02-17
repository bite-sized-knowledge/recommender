import streamlit as st
import pandas as pd
import sys
import webbrowser
sys.path.append("../src")

user_list = ["대현", "정제", "유선", "다은", "인성"]

topics = [
    "web", "mobile(android, ios) engineering", "hardware & iot", "ai & ml & data",
    "security & network", "db", "devops & infra", "game", "product manager",
    "design", "etc", "n/a"
]

@st.cache_resource
def get_articles(user: str) -> pd.DataFrame:
    articles = pd.read_csv(
        f"./dataset_sample/classification_inf_user{user_list.index(user) + 1}.tsv",
        sep="\t"
    )
    return articles

def save_response(user, url, title, predicted_topic, rating):
    df = pd.DataFrame({
        "url": [url],
        "title": [title],
        "topic": [topics.index(predicted_topic)],
        "rating": [rating]
    })
    file_path = f"./dataset_sample/response_user{user}.tsv"
    try:
        existing = pd.read_csv(file_path, sep="\t")
        existing = existing[existing["url"] != url]  # 기존 응답 제거
        new_df = pd.concat([existing, df], ignore_index=True)
    except FileNotFoundError:
        new_df = df 
    new_df.to_csv(file_path, sep="\t", index=False)

# ----------------- streamlit UI 구성 -----------------
st.title("Text Classification 예측 및 응답 저장")x

# 사용자 선택 (라디오 버튼)
selected_user = st.sidebar.radio("사용자 선택", user_list)
st.subheader(f"선택된 사용자 : {selected_user}")
st.markdown("---")

# 사용자별 기사 불러오기
articles_df = get_articles(selected_user)
user_idx = user_list.index(selected_user) + 1

for idx, row in articles_df.iterrows():
    st.subheader(f"🖥️ 컨텐츠 제목: {row['title']}")
    keywords = row['keywords'].split("#")
    st.markdown("""
    <style>
    .inline-keyword {
        display: inline-block;
        background-color: #ffcc80; /* Light orange background */
        color: #000; /* Black text */
        padding: 5px 10px;
        margin-right: 5px; /* Space between hashtags */
        border-radius: 15px;
        font-size: 14px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div>", unsafe_allow_html=True)
    st.markdown(
        " ".join([f"<span class='inline-keyword'>#{keyword}</span>" for keyword in keywords]),
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
   # URL 클릭하여 새 창에서 열기
    if st.button(f"📖 컨텐츠 읽기", key=f"read_{idx}"):
        webbrowser.open_new_tab(row['url'])
    
    # 예상 주제 선택
    predicted_topic = st.selectbox(
        f"예상 주제 선택 (컨텐츠 {row['title']})",
        topics
    )

    st.markdown(f"<p style='font-size:20px; font-weight:bold;'>⭐️ 키워드 분류 성능 평가</p>", 
                unsafe_allow_html=True)
    rating = st.slider(
        "1점부터 5점까지 골라주세요", min_value=1, max_value=5, key=f"rating_{idx}"
    )
    
    # 응답 저장 버튼
    if st.button(f"✅ 응답 저장", key=f"save_{idx}"):
        save_response(user_idx, row['url'], row['title'],  predicted_topic, rating)
        st.success("응답이 저장되었습니다.")
    
    st.markdown("---")