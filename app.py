import os
import re
import shutil
import tempfile
import uuid
from io import BytesIO
from pathlib import Path
from subprocess import PIPE, run

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def convert_doc_to_pdf_native(doc_file: Path, output_dir: Path = Path("."), timeout: int = 60):
    """Convert a .doc or .docx file to PDF using LibreOffice."""
    try:
        process = run(
            ['soffice', '--headless', '--convert-to', 'pdf:writer_pdf_Export', '--outdir', output_dir.resolve(), doc_file.resolve()],
            stdout=PIPE, stderr=PIPE,
            timeout=timeout, check=True
        )
        stdout = process.stdout.decode("utf-8")
        re_filename = re.search(r'-> (.*?) using filter', stdout)
        return Path(re_filename[1]).resolve() if re_filename else None
    except Exception as e:
        st.error(f"Error converting {doc_file.name} to PDF: {e}")
        return None


def get_pdf_text(pdf_docs):
    """Extract text from PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


def get_text_chunks(text):
    """Split text into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)


def get_vector_store(chunks):
    """Generate embeddings and store in FAISS."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    """Load Gemini model for QA processing."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not available, respond with "answer is not available in the context".
    If a user enters RFP documnent heading and asks you for content in the heading then give the user content according to the heading after analyzing the rfp
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)


def clear_chat_history():
    """Clear chat history."""
    st.session_state.messages = [{"role": "assistant", "content": "Upload PDFs or Word files and ask me a question"}]


def user_input(user_question):
    """Retrieve relevant document sections and generate an answer."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response


def main():
    st.set_page_config(page_title="Gemini PDF & Word Chatbot", page_icon="🤖")

    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader("Upload PDF or Word Files", type=["pdf", "doc", "docx"], accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                temp_dir = Path(tempfile.mkdtemp())

                pdf_files = []
                for file in uploaded_files:
                    file_path = temp_dir / file.name
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())

                    if file.name.endswith((".doc", ".docx")):
                        pdf_file = convert_doc_to_pdf_native(file_path, output_dir=temp_dir)
                        if pdf_file:
                            pdf_files.append(pdf_file)
                    else:
                        pdf_files.append(file_path)

                raw_text = get_pdf_text(pdf_files)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing Complete")

    st.title("RFP Analyzer")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Upload PDFs or Word files and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                full_response = "".join(response['output_text'])
                st.write(full_response)
        if response:
            st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
