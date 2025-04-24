# import streamlit as st
# from langchain_openai import ChatOpenAI
# from langchain.chains import RetrievalQA
# from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
#
# st.set_page_config(page_title="🩺 Medical QA Agent", layout="centered")
# st.title("🩺 Medical Question Answering Agent (Synthea)")
#
# embedding = OpenAIEmbeddings()
# vectordb = Chroma(persist_directory="chroma_store", embedding_function=embedding)
# retriever = vectordb.as_retriever(search_kwargs={"k": 3})
# llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
# qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
#
# question = st.text_input("Ask a question about a synthetic patient")
#
# if question:
#     with st.spinner("Thinking..."):
#         result = qa_chain(question)
#         st.subheader("💡 Answer")
#         st.write(result['result'])
#
#         st.subheader("📄 Sources")
#         for doc in result['source_documents']:
#             st.write(doc.metadata)

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma

# Page Config
st.set_page_config(page_title="🩺 Medical QA Agent", page_icon="💬", layout="centered")

# Title & Description
st.markdown("<h1 style='text-align: center;'>🩺 Medical QA Agent (Synthea)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask natural language questions about synthetic patient records generated using Synthea. Powered by GPT-4o + ChromaDB + LangChain.</p>", unsafe_allow_html=True)

# Initialize components
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory="chroma_store", embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Text Area for user question
st.markdown("### 🧾 Enter Your Question")
question = st.text_area("Example: What conditions does patient 1234abcd have?", height=100)

# Show answer
if question.strip():
    with st.spinner("🤖 Thinking..."):
        result = qa_chain(question)

        st.markdown("### 💡 Answer")
        st.success(result['result'])

        st.markdown("### 📄 Source Documents")
        for i, doc in enumerate(result['source_documents']):
            with st.expander(f"Source {i+1} — Patient ID: {doc.metadata.get('patient_id', 'N/A')}"):
                st.text(doc.page_content)