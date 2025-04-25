import streamlit as st
import pandas as pd
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from clarifier_chain import get_clarification_chain
from visualization import plot_medical_timeline, format_labs_vitals

st.set_page_config(page_title="Medical QA Agent", layout="wide")
st.title("🩺 Medical Question Answering Agent")

# Setup LLM and retriever
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory="chroma_store", embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Clarifier (optional enhancement for future use)
#clarifier = get_clarification_chain()

# UI Tabs
tab1, tab2 = st.tabs(["💬 Ask Questions", "📊 Patient Labs & Timeline"])

# --- Tab 1: QA Interface ---
with tab1:
    st.subheader("💬 Ask a medical question about a patient")
    with st.form("qa_form"):
        question = st.text_area("Enter your question here:", height=100)
        submit = st.form_submit_button("Submit")

    if submit and question.strip():
        with st.spinner("Generating answer..."):
            result = qa_chain(question)
            st.subheader("💡 Answer")
            st.success(result["result"])

            # st.subheader("📄 Source Documents")
            # for doc in result["source_documents"]:
            #     st.markdown(f"- `Patient ID:` `{doc.metadata.get('patient_id', 'N/A')}`")
            #     st.code(doc.page_content[:500] + "...")

# --- Tab 2: Labs & Timeline Viewer ---
with tab2:
    st.subheader("📊 Explore Patient Labs and Medical History Timeline")
    pid = st.text_input("Enter Patient ID to view labs and history")
    if pid:
        try:
            obs = pd.read_csv("data/observations.csv")
            cond = pd.read_csv("data/conditions.csv")

            patient_obs = obs[obs["PATIENT"] == pid]
            patient_cond = cond[cond["PATIENT"] == pid]

            st.markdown("#### 🧪 Lab & Vitals Summary")
            df = format_labs_vitals(patient_obs)
            st.dataframe(df, use_container_width=True)

            st.markdown("#### 📅 Condition Timeline")
            fig = plot_medical_timeline(patient_cond)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading patient data: {e}")
