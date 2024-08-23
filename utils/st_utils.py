import streamlit as st
from utils.rag_utils import *
import time

def print_msg(prompt, retriever, rag_prompt_custom, llm):
    print(st.session_state.chat_history)
# 사용자 메시지 추가
    for chat in st.session_state.chat_history:
        with st.chat_message("user" if chat["role"] == "user" else "ai"):
            st.markdown(chat["content"])

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    res = rag_chain(prompt, retriever, rag_prompt_custom, llm)
    st.session_state.chat_history.append({"role": "ai", "content": res})

    # AI 응답을 스트리밍 방식으로 표시
    # 최종 AI 응답 표시
    with st.chat_message("ai"):
        stream_text(res)

# AI 응답을 스트리밍 방식으로 표시
def streaming_md():
    c = 0
    for chat in st.session_state.chat_history:
        if c<=len(st.session_state.chat_history)-2 :
            with st.chat_message("user" if chat["role"] == "user" else "ai"):
                st.markdown(chat["content"])
        else  :
            with st.chat_message("ai"):
                stream_text(chat["content"])
        c += 1

def stream_text(text):    
    response_placeholder = st.empty()
    response_text = ""
    for chunk in text:
        response_text += chunk
        response_placeholder.markdown(response_text)
        time.sleep(0.01)


