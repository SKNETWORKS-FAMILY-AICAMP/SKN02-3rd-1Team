from langchain.schema.runnable import RunnablePassthrough

def is_relevant(query):
    keywords = [
        '숨막힘', '심정지', '심장 회생', '질식', '심폐소생술', '수영', '쇼크', 
        '저혈압', '안구 화상', '눈', '표면 화상', '2도 화상', '3도 화상', '화상', 
        '열사병', '고온', '현기증', '어지러움', '피로', '두통', '시야 흐림', 
        '근육통', '오심', '구토', '저체온', '저체온증', '동상', '추위', '감전 손상', '감전', 
        '전기', '전류', '낙뢰', '낙뢰 손상', '뇌우', '고산병', '높은 고도', '산소 부족', '부종', 
        '물림', '상처', '감염', '꿀벌', '말벌', '호박벌', '쏘임', '벌레 물림', '연조직 손상', '혹', 
        '멍', '열상', '인대', '염좌', '긴장', '타박상', '뼈', '고지질혈증', '낙상', '교통 사고', '깁스', 
        '팔꿈치 골절', '다리 골절', '대퇴골', '경골', '비골' ,'응급처치', 'CPR', '구급', '치료', '응급', 
        '개미', '익사', '골절', '부상', '출혈', '열사병', '고열', '뱀', '의식불명', '중독', '화상', '심장마비', 
        '호흡곤란', '부러', '꺾', '숨', '의식', '체온', '탈진', '일사병', '열', '벌', '해파리', '호흡', '심장', 
        '심폐', '식중독', '팔', '다리', '발', '목', '손가락', '머리', '엉덩이', '허리', '발가락', '눈', '약'
    ]
    return any(keyword in query.lower() for keyword in keywords)

# 질문에 따라 적절한 응답을 반환하는 함수
def respond_to_query(query):
    if is_relevant(query):
        return None  # 응급처치 관련 질문이라면, 기존 RAG chain이 처리하도록 None 반환
    else:
        return "저는 응급처치 매뉴얼에 대해 설명하는 AI입니다. 그러므로 해당 질문에는 답변할 수 없습니다."

def rag_chain(query, retriever, rag_prompt_custom,llm):
    # 질문이 응급처치와 관련 없는 경우, 여기서 바로 응답
    custom_response = respond_to_query(query)
    if custom_response:
        return custom_response

    # 응급처치 관련 질문이라면 기존 RAG chain을 통해 응답
    relevant_docs = retriever.get_relevant_documents(
        query, k=3)  # retriever에서 관련 문서를 3개 가져옴
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = rag_prompt_custom.format(context=context, question=query)

    # llm.generate() 대신 llm()을 사용하여 프롬프트를 전달

    rag_chain = {"context": retriever,
                 "question": RunnablePassthrough()} | rag_prompt_custom | llm
    response = rag_chain.invoke(prompt).content
    return response