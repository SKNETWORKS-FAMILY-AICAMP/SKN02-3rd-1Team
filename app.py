import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
from utils.st_utils import *
from utils.rag_utils import *
from utils.stt import *


# API 키 정보 로드
load_dotenv()
DB_PATH = os.getenv("DB_PATH")

# 디스크에서 문서를 로드합니다.
vectorstore = Chroma(
    persist_directory= DB_PATH,
    embedding_function=OpenAIEmbeddings(),
    collection_name="my_db",
)

#retriever 선언
retriever = vectorstore.as_retriever()

# 템플릿 설정
template = """
현재 상황에 대해 상대방이 진정할 수 있도록하는 문장으로 답변을 시작하십시오.

Use the following context to answer the last question.

If the question is ambiguous to answer, request additional information.

If you are responding to a question related to first aid, please provide instructions in a numbered sequence to make it easier to follow.

When analyzing the question, consider who, when, what, how, and why.

If the question refers to content not available in the document, state that you do not know the information.

Always respond in Korean.
절대 한국말로 답변해
{context}
질문: {question} 어떡해
도움이 되는 답변:"""

rag_prompt_custom = PromptTemplate.from_template(template)

# gpt-4o-mini를 이용해서 LLM 설정
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# 채팅 기록 초기화 (텍스트, 오디오 나눔)
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "audio_history" not in st.session_state:
    st.session_state["audio_history"] = []

# Streamlit 애플리케이션 설정
st.image("./img/emergency.jpg",width=300)
st.title("응급처치 가이드 챗봇")

with st.sidebar:
    st.header('음성 도움 챗봇')
    st.markdown('''
    :red[채팅] 이 어려우시다면 아래 :red-background[REC]
    버튼을 눌러 말씀해주세요.''')
    btn = audio_btn(retriever, rag_prompt_custom, llm)
    
# Text 입력 발생시 로그 출력
prompt = st.chat_input("증상에 대해 알려주세요..")
if btn:
    streaming_md()
elif prompt:
    print_msg(prompt, retriever, rag_prompt_custom, llm)
