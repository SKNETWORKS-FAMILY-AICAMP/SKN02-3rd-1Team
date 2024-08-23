from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

loader = PyPDFLoader("data/safekorea_crawling_result.pdf")
pages = loader.load_and_split()
loader = PyPDFLoader("data/응급상황및손상.pdf")
pages2 = loader.load_and_split()
loader = PyPDFLoader("data/20230721 계곡·해수욕장 등에서의 안전사고 예방·대처요령_교육책자(최종).pdf")
pages3 = loader.load_and_split()
pages = pages + pages2 + pages3
pages = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(pages)
# 저장할 경로 지정
load_dotenv()
DB_PATH = os.getenv("DB_PATH")

# 문서를 디스크에 저장합니다. 저장시 persist_directory에 저장할 경로를 지정합니다.
vectorstore = Chroma.from_documents(
    splits, OpenAIEmbeddings(), persist_directory=DB_PATH, collection_name="my_db"
)