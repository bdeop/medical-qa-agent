import streamlit as st
import pandas as pd
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from clarifier_chain import get_clarification_chain
from visualization import plot_medical_timeline, format_labs_vitals

st.set_page_config(page_title="Medical QA Agent", layout="wide")
st.title("🩺 Medical Question Answering Agent")

embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory="chroma_store", embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

clarifier = get_clarification_chain()

question = st.text_area("Ask a medical question about a synthetic patient:", height=100)

if question.strip():
    # clarification = clarifier.run(question)
    # st.markdown("**🤔 Clarification / Context:**")
    # st.info(clarification)

    with st.spinner("Generating answer..."):
        result = qa_chain(question)
        st.subheader("💡 Answer")
        st.success(result["result"])


st.markdown("---")
st.markdown("### 🔍 Explore Patient Labs and Timeline")
pid = st.text_input("Enter Patient ID to view labs and history timeline")
if pid:
    try:
        obs = pd.read_csv("data/observations.csv")
        cond = pd.read_csv("data/conditions.csv")

        patient_obs = obs[obs["PATIENT"] == pid]
        patient_cond = cond[cond["PATIENT"] == pid]

        st.markdown("#### 🧪 Lab & Vitals Summary")
        df = format_labs_vitals(patient_obs)
        st.dataframe(df, use_container_width=True)

        st.markdown("#### 🩺 Condition Timeline")
        fig = plot_medical_timeline(patient_cond)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading patient data: {e}")
