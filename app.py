import os
from PyPDF2 import PdfReader
import docx
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

def get_document_text(uploaded_files):
    """Extract text from PDF and Word documents"""
    text = ""
    for file in uploaded_files:
        if file.name.endswith('.pdf'):
            # Process PDF files
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
                
        elif file.name.endswith('.docx'):
            # Process Word documents
            doc = docx.Document(file)
            # Extract paragraphs
            for para in doc.paragraphs:
                text += para.text + "\n"
            # Extract tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + "\n"
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
    """Create vector embeddings store"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Create the conversation chain with prompt template"""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    Follow these rules:
    1. Provide all relevant details if available in the context
    2. If answer isn't in context, say "answer is not available in the context"
    3. Never provide false information
    4. For RFP heading requests, analyze thoroughly and provide appropriate content
    
    Context:\n{context}\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        client=genai,
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
        {"role": "assistant", "content": "Upload documents to begin"}
    ]

def user_input(user_question):
    """Process user question and generate response"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    
    # Load vector store with safety override
    vector_store = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # Find relevant document sections
    docs = vector_store.similarity_search(user_question)
    
    # Generate response
    chain = get_conversational_chain()
    return chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

def main():
    """Main application interface"""
    st.set_page_config(
        page_title="Smart RFP Analyser",
        page_icon="📄",
        layout="centered"
    )

    # Sidebar Configuration
    with st.sidebar:
        st.title("Control Panel")
        st.subheader("Upload Documents")
        
        # File uploader with type restrictions
        uploaded_files = st.file_uploader(
            "Choose PDF or Word files",
            type=["pdf", "docx"],
            accept_multiple_files=True
        )
        
        # Processing button
        if st.button("Process Documents"):
            if not uploaded_files:
                st.warning("Please upload files first")
                return
                
            with st.spinner("Analyzing documents..."):
                raw_text = get_document_text(uploaded_files)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Documents processed successfully!")

        st.button("Clear Chat History", on_click=clear_chat_history)

    # Main Chat Interface
    st.title("📄 Smart RFP Analyser")
    st.caption("Upload PDF/Word documents and ask questions about their content")

    # Initialize chat history
    if "messages" not in st.session_state:
        clear_chat_history()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask about your documents..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
            
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                response = user_input(prompt)
                full_response = response.get("output_text", "Could not generate response")
                
                # Display response
                st.write(full_response)
                
        # Add assistant response to history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

if __name__ == "__main__":
    main()
