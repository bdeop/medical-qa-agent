import pickle
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

with open("agent/patient_records.pkl", "rb") as f:
    records = pickle.load(f)

texts = [record[1] for record in records]
metadatas = [{'patient_id': record[0]} for record in records]
documents = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadatas)]

embedding = OpenAIEmbeddings()
db = Chroma.from_documents(documents, embedding=embedding, persist_directory="chroma_store")
db.persist()
