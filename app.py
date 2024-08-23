import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
import time
import pyaudio
import wave
from openai import OpenAI
from dotenv import load_dotenv
import os

# API 키 정보 로드
load_dotenv()
DB_PATH = os.getenv("DB_PATH")
# 디스크에서 문서를 로드합니다.
vectorstore = Chroma(
    persist_directory= DB_PATH,
    embedding_function=OpenAIEmbeddings(),
    collection_name="my_db",
)

retriever = vectorstore.as_retriever()


def is_relevant(query):
    keywords = [
        '응급처치', 'CPR', '구급', '치료', '응급', '개미', '익사',
        '골절', '부상', '출혈', '열사병', '고열', '뱀', '의식불명',
        '중독', '화상', '심장마비', '호흡곤란', '부러', '꺾', '숨', '의식', '체온',
        '탈진', '일사병', '열', '벌', '해파리', '호흡', '심장', '심폐', '식중독', '팔', '다리', '발', '목'
    ]
    return any(keyword in query.lower() for keyword in keywords)

# 질문에 따라 적절한 응답을 반환하는 함수


def respond_to_query(query):
    if is_relevant(query):
        return None  # 응급처치 관련 질문이라면, 기존 RAG chain이 처리하도록 None 반환
    else:
        return "저는 응급처치 매뉴얼에 대해 설명하는 AI입니다. 그러므로 해당 질문에는 답변할 수 없습니다."


def voice_to_text():
    client = OpenAI()

    audio_file = open("output.wav", "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    return transcription.text


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

# GPT-3.5 turbo를 이용해서 LLM 설정
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# RAG chain 설정


def rag_chain(query):
    # 질문이 응급처치와 관련 없는 경우, 여기서 바로 응답
    custom_response = respond_to_query(query)
    if custom_response:
        return custom_response

    # 응급처치 관련 질문이라면 기존 RAG chain을 통해 응답
    relevant_docs = retriever.get_relevant_documents(
        query, k=3)  # retriever에서 관련 문서들을 가져옴
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = rag_prompt_custom.format(context=context, question=query)

    # llm.generate() 대신 llm()을 사용하여 프롬프트를 전달

    rag_chain = {"context": retriever,
                 "question": RunnablePassthrough()} | rag_prompt_custom | llm
    response = rag_chain.invoke(prompt).content
    return response


def record_audio(output_filename, record_seconds=5, sample_rate=44100, chunk_size=1024, channels=2):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    st.write("목소리를 듣고 있어요.")

    frames = []
    for _ in range(0, int(sample_rate / chunk_size * record_seconds)):
        data = stream.read(chunk_size)
        frames.append(data)

    st.write("다 들었어요. 잠시만 기다려주세요.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    client = OpenAI()

    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    audio_file = open("output.wav", "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

    prompt = transcription.text

    return prompt

def print_msg(prompt):
# 사용자 메시지 추가
    for chat in st.session_state.chat_history:
        with st.chat_message("user" if chat["role"] == "user" else "ai"):
            st.markdown(chat["content"])

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    res = rag_chain(prompt)
    st.session_state.chat_history.append({"role": "ai", "content": res})

    # AI 응답을 스트리밍 방식으로 표시
    # 최종 AI 응답 표시
    with st.chat_message("ai"):
        response_placeholder = st.empty()
        response_text = ""
        for chunk in res:
            response_text += chunk
            response_placeholder.markdown(response_text)
            time.sleep(0.01)  # 스트리밍 효과를 위해 잠시 대기


def streaming_md():
    for chat in st.session_state.chat_history:
        with st.chat_message("user" if chat["role"] == "user" else "ai"):
            response_placeholder = st.empty()
            response_text = ""
            for chunk in chat["content"]:
                response_text += chunk
                response_placeholder.markdown(response_text)
                time.sleep(0.01)


# 채팅 기록 초기화 (텍스트, 오디오 나눔)
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "audio_history" not in st.session_state:
    st.session_state["audio_history"] = []


# Streamlit 애플리케이션 설정
st.image("./img/emergency.jpg",width=300)
st.title("응급처치 가이드 챗봇")
# icon,title = st.columns([1,6])
# with icon:
#     st.image("./img/red_corss.png")
# with title:
#     st.title("응급처치 가이드 챗봇")

def audio_btn():
    if st.button("REC", type="primary"):
        prompt = record_audio("output.wav", record_seconds=5)
        input_audio = True
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        res = rag_chain(prompt)
        st.session_state.chat_history.append({"role": "ai", "content": res})
    else:
        input_audio = False

    return input_audio

global btn

with st.sidebar:
    st.header('음성 도움 챗봇')
    st.markdown('''
    :red[채팅] 이 어려우시다면 아래 :red-background[REC]
    버튼을 눌러 말씀해주세요.''')
    btn = audio_btn()


# Text 입력 발생시 로그 출력
prompt = st.chat_input("증상에 대해 알려주세요..")
if btn:
    streaming_md()
elif prompt:
    print_msg(prompt)
