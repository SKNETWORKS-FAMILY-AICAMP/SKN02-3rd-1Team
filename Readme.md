# GPT 활용 응급처치 챗봇

<a href="https://fragrant-paprika-e91.notion.site/gpt-8aa9494a44724d40bded8185869bc5ff">노션페이지</a>

## 팀원 소개
|강민호|김서연|김진유|박경희|
|---|---|---|---|
|**팀장**</br>데이터 전처리</br>랭체인</br>웹 구현|데이터 수집</br>데이터 전처리</br>테스트|데이터 전처리</br>랭체인</br>웹 구현|웹 디자인</br>문서화</br>웹구현|

</br></br>


## 기술스택
|<img width='50' src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-3rd-1Team/blob/main/img/chat_gpt.png"> </br> GPT|<img width='120' src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-3rd-1Team/blob/main/img/lang_chain.png"> </br> |<img width='50' src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-3rd-1Team/blob/main/img/python.png"> </br> Python | <img width='50' src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-3rd-1Team/blob/main/img/streamlit_hero.png"> </br> Streamlit|
|-|-|-|-|
|AI 챗봇 제작을 위하여</br>OpanAI GPT API 사용|LLM을 활용한 어플리케이션</br>개발을 단순화할 수 있도록 하는</br>프레임워크|데이터 전처리 및 LLM,</br> 스트림릿 등 AI 챗봇 제작을 </br> 위한 사용 언어|웹 구현을 위해 사용|

</br></br>

## 1. 선정 배경
일상 속 응급처치가 필요한 상황이지만 방법을 몰라 당황한 적이 한 번 쯤 있을 것이다. 혹은 안다고 하더라도 이게 정말 올바른 방법인지 확신을 갖는 사람은 많지 않다.</br>
최근 전공의 파업으로 인하여 병원 인력이 부족한 실정이다. 응급실에 방문하는 환자의 44% 가량이 경증환자로 반드시 응급실에 방문할 필요가 없다. </br>응급실에 방문하기 전 개인이 잘못된 방법으로 응급처치를 하다 오히려 상황을 더 악화시키는 경우도 있다. </br>
누구나 응급처치에 대한 정보를 쉽게 알 수 있도록 gpt 모델에 119응급처치 방법을 Lang-Chain과 연동한 응급처치 가이드 AI 챗봇을 만들어 각 상황에 대한 올바른 응급처치 방법, 응급실 방문 필수여부 등을 가려내고자 한다.
 
</br></br>

## 2. 프로젝트 개요
### 1) 데이터 수집 및 전처리
#### 데이터 수집 및 전처리
[1] MSD 메뉴얼 홈페이지에서 각 응급상황별 조치에 대한 옵션을 선택한 뒤 텍스트를 크롤링하여 PDF로 저장한다. 해당 PDF는 제목, 챕터, 본문으로 구성된 클래스를 생성하여 제작한다. 만약 크롤링이 불가능한 정보가 있다면 url으로부터 PDF파일을 받아와 데이터에 추가한다.

[2] 응급처치 AI 챗봇을 제작하기 위해 저장된 응급처치 PDF 데이터를 불러온 뒤, RecursiveCharacterTextSplitter를 사용하여 텍스트를 청크단위로 분할한다. 이때 chunk_size=500, chunk_overlap=50으로 지정하여 청크의 최대 길이 및 텍스트 중복 옵션을 설정한다.

[3] 분할된 텍스트 청크를 Chroma.from_documents()으로 벡터 데이터베이스에 저장한다. 이 벡터 데이터베이스에서 검색 기능을 제공하는 retriever 객체를 생성한다. 이를 통해 사용자가 입력한 자연어로 된 질문에 대해 관련 텍스트를 찾아 빠르게 답변을 받을 수 있도록 구성하였다.

[4] (음성 데이터 및 이미지 데이터 처리)

[5] 수집된 데이터 구조
## 내용 추가 하기

  
### 2) 모델
#### LLM
**GPT-4o mini**</br>
이 모델은 대화형 인터페이스에 최적화된 언어 모델이다. 챗봇이라는 특성상 사용자와 AI가 대화를 통해 정보를 제공해야 하므로 해당 모델을 채택하였다.
12만8000개의 토큰으로 구성된 컨텍스트 창을 가지고 있으며 GPT-4o와 공유하는 개선된 토크나이저가 있어 영어가 아닌 텍스트 처리에 효율적이다.
평균 출력 속도는 초당 202토큰으로, 'GPT-4o' 및 GPT-3.5 터보보다 ​​2배 이상 빠르기 때문에  LLM을 사용하는 AI 개발에 적합한 모델이다.</br>

**Temperature**</br>
2에 가까워질수록 창의적인 답변을 제공한다. 정확하고 객관적인 답변을 해야하므로  0으로 지정하였다.</br>
  
### 3) 구현 방법</br>
Lang Chain 및 Streamlit을 사용하여 제작
### 1) 사용 함수 분석
#### 전처리 부분
**is_relevant(query)**</br>
주어진 쿼리(문자열)가 특정 키워드 목록 중 하나라도 포함하고 있는지를 확인하여 해당 쿼리가 관련성이 있는지를 판단한다.</br></br>

**respond_to_query(query)**</br>
사용자의 질문(query)가 응급 상황과 관련있는지 판단하여 관련있다면 기존 RAG chain이 처리할 수 있도록 None을 반환, 그렇지 않다면 응답하지 않도록 한다.</br></br>

**voice_to_text()**</br>
녹음된 사용자 음성파일을 열어 음성인식 모델(whisper)를 사용하여 텍스트로 전환한 뒤 반환한다.</br></br>

**rag_chain(query)**</br>
질문이 응급처치와 관련이 없는 경우, 미리 지정해둔 응급 상황과 관련된 질문을 해달라는 답변을 반환한다.</br>
질문이 응급처치와 관련 있는 경우, 관련 문서를 가져와서 Rag-Chain을 통해 응답한다.</br></br>

**record_audio()**</br>
이 함수는 함수는 사용자의 목소리를 녹음한 후, 해당 음성을 텍스트로 바꾸어 반환한다. 이 함수는 음성 녹음과 오디오 파일 생성, 그리고 녹음된 음성을 텍스트로 변환하는 작업을 수행한다.</br></br>

**print_msg(prompt)**</br>
사용자 메시지와 그 메시지에 해당하는 AI의 응답을 출력하는 함수이다. 챗봇 특성에 걸맞게 Stream 형식으로 출력이 되도록 하였다.</br></br>

**streaming_md()**</br>
웹의 main에서 텍스트 답변을 위한 AI 응답 출력 함수이다.</br></br>

#### Streamlit 구현부
**main**</br>
streamlit.chat_input() 함수를 사용하여 챗봇 환경을 구축하였다. 사용자가 입력한 내용이 rag_chain을 거쳐 적합한 응답을 출력한다. 사용자와 AI 아이콘에 구분을 두어 가시성을 높였다.</br></br>

**side bar**</br>
웹 좌측에 음성 챗봇을 구현하기 위해 녹음 버튼과 안내 문장을 출력하였다. REC 버튼을 누르면 5초간 녹음이 시작되며 만들어진 녹음 파일은 텍스트로 변환되어 rag_chain을 통해 main 챗봇 부로 응답을 출력한다.</br></br> 


## 3. 프로젝트 결과 </br>
<img width='500' src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-3rd-1Team/blob/main/img/run.png">
