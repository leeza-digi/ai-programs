import streamlit as st
import google.generativeai as genai
import os
import io
import fitz
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from fpdf import FPDF

st.title("Document Editor (Structured)")
file_input = st.file_uploader("Upload a PDF file", type="pdf")

genai.configure(api_key="AIzaSyBUaquTv5t4GWxHNtrqTTbADX8HoINBD1s")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
model = genai.GenerativeModel('gemini-1.5-flash-latest')

def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf_output = io.BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    pdf_output.write(pdf_bytes)
    pdf_output.seek(0)
    return pdf_output

def clean_text_for_pdf(text):
    return text.encode('latin-1', 'replace').decode('latin-1')

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

if file_input and not st.session_state.pdf_uploaded:
    pdf_bytes = file_input.read()
    text = ''
    docs = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            page_text = page.get_text()
            text += page_text
            docs.append(Document(page_content=page_text))

    vectorstore = FAISS.from_documents(docs, embedding_model)
    st.session_state.vectorstore = vectorstore
    st.session_state.text = text
    st.session_state.pdf_uploaded = True

if "text" in st.session_state:
    st.subheader("Editing Instructions")
    edit_prompt = st.text_area("Describe how you want to change the document")

    if st.button("Generate New Document"):
        full_instruction = f"""
You are a helpful assistant. Modify the following document based on the instructions below.

Document:
{st.session_state.text}

Instructions:
{edit_prompt}

Return ONLY the revised document text, no explanations.
"""
        response = model.generate_content(full_instruction)
        st.session_state.modified_text = response.text.strip()

if "modified_text" in st.session_state:
    st.subheader("Modified Document Preview")
    st.text_area("Modified content:", value=st.session_state.modified_text, height=400)
    modified_pdf = create_pdf(clean_text_for_pdf(st.session_state.modified_text))
    st.download_button(
        label="Download Modified Document as PDF",
        data=modified_pdf,
        file_name="modified_document.pdf",
        mime="application/pdf"
    )

if "text" in st.session_state:
    original_pdf = create_pdf(clean_text_for_pdf(st.session_state.text))
    st.download_button(
        label="Download Original Extracted Text as PDF",
        data=original_pdf,
        file_name="extracted_text.pdf",
        mime="application/pdf"
    )








#streamlit run "/Users/leeza/Desktop/Digisprint/Streamlit Document/document_reader.py"

                                                                                                                                                                                        