import streamlit as st
import google.generativeai as genai
import os
import fitz
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from gemini_llm import GeminiLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

#st.set_page_config(layout="")
st.title(" PDF QnA ")
file_input = st.file_uploader("Upload a PDF file", type="pdf")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

genai.configure(api_key="AIzaSyBUaquTv5t4GWxHNtrqTTbADX8HoINBD1s")

st.markdown("Chat History")
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-end; margin: 10px 0;'>
                <div style='max-width: 70%; background-color: #2f2f2f; color: white; padding: 10px; border-radius: 10px; text-align: right;'>
                    <div style='display: flex; align-items: center; justify-content: flex-end;'>
                        <span style='margin-left: 8px; font-weight: bold;'>You</span>
                        <img src='https://img.icons8.com/ios-filled/30/808080/user.png' style='margin-left: 5px;'/>
                    </div>
                    <div style='margin-top: 5px;'>{msg}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-start; margin: 10px 0;'>
                <div style='max-width: 70%; background-color: #d3d3d3; color: black; padding: 10px; border-radius: 10px;'>
                    <div style='display: flex; align-items: center;'>
                        <img src='https://img.icons8.com/ios-filled/30/808080/robot-2.png' style='margin-right: 5px;'/>
                        <span style='margin-right: 8px; font-weight: bold;'>Gemini</span>
                    </div>
                    <div style='margin-top: 5px;'>{msg}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        

##########

if file_input and not st.session_state.pdf_uploaded:
    pdf_bytes = file_input.read()
    text = ''
    docs = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            page_text = page.get_text()
            text += page_text
            docs.append(Document(page_content=page_text))

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding_model)
    st.session_state.vectorstore = vectorstore

    llm = GeminiLLM(api_key="AIzaSyBUaquTv5t4GWxHNtrqTTbADX8HoINBD1s")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    st.session_state.qa_chain = qa_chain
    st.session_state.pdf_uploaded = True

user_input = st.text_input("Ask a question from the PDF")

if st.button("Generate Response") and user_input and st.session_state.qa_chain:
    response = st.session_state.qa_chain.run({"question": user_input})
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Gemini", response))
    st.rerun()



#streamlit run /Users/leeza/Desktop/Digisprint/Streamlit/streamlit_langchain.py