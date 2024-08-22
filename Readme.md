# GPT 활용 응급처치 챗봇

## 팀원 소개
|강민호|김서연|김진유|박경희|
|---|---|---|---|
|**팀장**</br>데이터 전처리</br>랭체인</br>웹 구현|데이터 수집|데이터 전처리</br>랭체인</br>웹 구현|문서화|

</br></br>


## 기술스택
|<img width='50' src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-3rd-1Team/blob/main/img/chat_gpt.png"> </br> Chat GPT|<img width='120' src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-3rd-1Team/blob/main/img/lang_chain.png"> </br> |<img width='50' src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-3rd-1Team/blob/main/img/python.png"> </br> Python | <img width='50' src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-3rd-1Team/blob/main/img/streamlit_hero.png"> </br> Streamlit|
|-|-|-|-|
|AI 챗봇 제작을 위하여</br>OpanAI GPT API 사용|LLM을 활용한 어플리케이션</br>개발을 단순화할 수 있도록 하는</br>프레임워크|데이터 전처리 및 LLM,</br> 스트림릿 등 AI 챗봇 제작을 </br> 위한 사용 언어|웹 구현을 위해 사용|

</br></br>

## 1. 선정 배경
일상 속 응급처치가 필요한 상황이지만 방법을 몰라 당황한 적이 한 번 쯤 있을 것이다. 혹은 안다고 하더라도 이게 정말 올바른 방법인지 확신을 갖는 사람은 많지 않다.</br>
전공의 파업으로 인하여 병원 인력이 부족한 실정이다. 응급실에 방문하는 환자의 44% 가량이 경증환자로 반드시 응급실에 방문할 필요가 없다. </br>응급실에 방문하기 전 개인이 응급처치를 하다 잘못된 처치로 오히려 상황을 더 악화시키는 경우도 있다. </br>
응급처치에 대한 정보를 쉽게 알 수 있도록 gpt 모델에 119응급처치 방법을 Lang-Chain과 연동한 응급 AI 챗봇을 만들어 각 상황에 대한 올바른 응급처치 방법, 응급실 방문 필수여부 등을 가려내고자 한다.
 
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

</br></br> 
## 3. 프로젝트 결과</br>
<img>
