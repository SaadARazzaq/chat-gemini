import os
import io
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_file_text(uploaded_files):
    """Extract text from PDF and Word documents"""
    text = ""
    for file in uploaded_files:
        # Process PDF files
        if file.name.endswith('.pdf'):
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        # Process Word documents
        elif file.name.endswith(('.doc', '.docx')):
            doc = Document(io.BytesIO(file.read()))
            for para in doc.paragraphs:
                text += para.text + "\n"
            text += "\n"  # Add space between documents
    
    return text

def get_text_chunks(text):
    """Split text into manageable chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=1000
    )
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    """Create and save vector embeddings"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Create the QA chain with prompt template"""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, 
    make sure to provide all the details. If the answer isn't in the context,
    say "answer is not available in the context". Don't invent answers.
    If asked about RFP headings, analyze and provide appropriate content.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def clear_chat_history():
    """Reset chat history"""
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload documents and ask questions!"}
    ]

def user_input(user_question):
    """Handle user queries and generate responses"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load vector store with safety override for serialization
    vector_store = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    docs = vector_store.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    return response

def main():
    """Main application layout and logic"""
    st.set_page_config(
        page_title="Smart RFP Analyzer",
        page_icon="📄",
        layout="centered"
    )

    # Sidebar configuration
    with st.sidebar:
        st.title("Document Upload")
        st.subheader("Upload your RFP documents")
        uploaded_files = st.file_uploader(
            "Select PDF or Word files",
            type=['pdf', 'doc', 'docx'],
            accept_multiple_files=True
        )
        
        if st.button("Process Documents"):
            with st.spinner("Analyzing documents..."):
                if not uploaded_files:
                    st.warning("Please upload documents first!")
                    return
                
                raw_text = get_file_text(uploaded_files)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Documents processed successfully!")

    # Main chat interface
    st.title("📄 Smart RFP Analyzer")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Initialize chat history
    if "messages" not in st.session_state:
        clear_chat_history()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask about your RFP documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                response = user_input(prompt)
                answer = response.get('output_text', 'Unable to generate response')
                
                # Stream the response
                placeholder = st.empty()
                full_response = ''
                for chunk in answer.split():
                    full_response += chunk + " "
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

if __name__ == "__main__":
    main()
