import pickle
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load combined patient records
with open("agent/patient_records.pkl", "rb") as f:
    records = pickle.load(f)  # list of (patient_id, text)

# Setup the text splitter to chunk long documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # max tokens per chunk
    chunk_overlap=100,     # overlap to preserve context
)

# Chunk each document and collect into a list
chunked_documents = []
for patient_id, full_text in records:
    chunks = splitter.split_text(full_text)
    for i, chunk in enumerate(chunks):
        chunked_documents.append(Document(
            page_content=chunk,
            metadata={"patient_id": patient_id, "chunk_index": i}
        ))

# Initialize embedding model
embedding = OpenAIEmbeddings()

# Create vector store from chunked documents
db = Chroma.from_documents(chunked_documents, embedding=embedding, persist_directory="chroma_store")
db.persist()
