import streamlit as st
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    TextLoader,
    PythonLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.retrieval_qa.base import RetrievalQA
import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
import pandas as pd
from io import StringIO

# ========================
#      UI Configuration
# ========================
primary_color = "#2E86C1"
secondary_color = "#28B463"
background_color = "#F8F9F9"
text_color = "#2C3E50"
font_family = "Helvetica Neue"

st.set_page_config(
    page_title="Project Senior",
    page_icon="📚",
    layout="centered"
)

custom_css = f"""
    <style>
    .stApp {{
        background-color: {background_color};
        color: {text_color};
        font-family: {font_family};
    }}
    .stButton>button {{
        background-color: {primary_color};
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: {secondary_color};
        transform: scale(1.05);
    }}
    .stFileUploader>div>div>div>button {{
        background-color: {primary_color};
        color: Black !important;
    }}
    .uploaded-file {{
        padding: 15px;
        background: #EBF5FB;
        border-radius: 10px;
        margin: 10px 0;
    }}
    .response-box {{
        padding: 20px;
        background: #EAEDED;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 4px solid {primary_color};
    }}
    .sidebar .sidebar-content {{
        background: {background_color};
    }}
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ========================
#      Sidebar Settings
# ========================
with st.sidebar:
    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk Size", 512, 2048, 1024)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7)
    top_p = st.slider("Top-P", 0.1, 1.0, 0.9)

# ========================
#    Main Application
# ========================
st.title("📚 Project Senior")
st.markdown("### AI-powered Document & Code Analysis")

# Supported file types
SUPPORTED_FILES = {
    "pdf": PDFPlumberLoader,
    "txt": TextLoader,
    "py": PythonLoader,
    "html": UnstructuredHTMLLoader,
    "md": UnstructuredMarkdownLoader,
    "js": TextLoader,
    "c": TextLoader,
    "cpp": TextLoader,
    "java": TextLoader,
    "h": TextLoader,
    "hpp": TextLoader,
    "cs": TextLoader,
    "php": TextLoader,
    "rb": TextLoader,
    "go": TextLoader,
    "csv": CSVLoader,
    "xls": UnstructuredExcelLoader,
    "xlsx": UnstructuredExcelLoader,
    "pptx": UnstructuredPowerPointLoader
}


# File upload section
uploaded_file = st.file_uploader(
    "Upload your document or code file",
    type=list(SUPPORTED_FILES.keys()),
    help="Supported formats: PDF, TXT, Python, JavaScript, C/C++, Java, HTML, Markdown,CSV, xls, xlsx, pptx and more"
)

def process_file(uploaded_file):
    """Handle different file types and load documents"""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    temp_path = f"temp.{file_extension}"
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if file_extension not in SUPPORTED_FILES:
        os.remove(temp_path)
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    loader_class = SUPPORTED_FILES[file_extension]
    loader = loader_class(temp_path)
    docs = loader.load()
    os.remove(temp_path)
    return docs

def create_text_splitter(file_extension):
    """Create appropriate text splitter based on file type"""
    code_extensions = ['py', 'html', 'md', 'js', 'c', 'cpp', 'java', 
                      'h', 'hpp', 'cs', 'php', 'rb', 'go']
    if file_extension in code_extensions:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
    return SemanticChunker(HuggingFaceEmbeddings())

if uploaded_file is not None:
    # Display file info
    with st.expander("📁 Uploaded File Details", expanded=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{len(uploaded_file.getvalue()) / 1024:.2f} KB")

    # Process documents
    try:
        with st.spinner("🔍 Analyzing content..."):
            docs = process_file(uploaded_file)
            file_extension = uploaded_file.name.split('.')[-1].lower()
            text_splitter = create_text_splitter(file_extension)
            documents = text_splitter.split_documents(docs)

        # Embedding and vector store
        with st.spinner("🧠 Building knowledge base..."):
            embedder = HuggingFaceEmbeddings()
            vector = FAISS.from_documents(documents, embedder)
            retriever = vector.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 10}
            )

        # Initialize LLM with sidebar settings
        llm = Ollama(
            model="deepseek-r1:1.5b",
            temperature=temperature,
            top_p=top_p
        )

        # Enhanced prompt template
        prompt_template = """
        You are an expert analyst. Follow these steps:
        1. Carefully analyze the context from the document
        2. Identify key concepts and relationships
        3. For code files, also analyze structure and functionality
        4. Provide a clear, structured response
        5. Include technical details when appropriate
        
        Context: {context}
        Question: {question}
        
        Provide your analysis:"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)
        
        # Setup processing chain
        llm_chain = LLMChain(
            llm=llm,
            prompt=QA_CHAIN_PROMPT,
            verbose=False
        )

        document_prompt = PromptTemplate(
            input_variables=["page_content", "source"],
            template="Source: {source}\nContent: {page_content}",
        )

        combine_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context",
            document_prompt=document_prompt
        )

        qa = RetrievalQA(
            combine_documents_chain=combine_documents_chain,
            retriever=retriever,
            return_source_documents=True
        )

        # Question input
        user_input = st.text_input("💡 Ask about the document:", placeholder="Type your question here...")

        # Handle question
        if user_input:
            with st.spinner("🧠 Processing your question..."):
                try:
                    response = qa(user_input)["result"]
                    sources = list(set(
                        doc.metadata['source'] 
                        for doc in qa(user_input)["source_documents"]
                    ))
                    
                    st.markdown("### 📝 Response")
                    st.markdown(f"<div class='response-box'>{response}</div>", unsafe_allow_html=True)
                    
                    with st.expander("🔍 View Sources"):
                        for source in sources:
                            st.write(f"- {source}")
                            
                except Exception as e:
                    st.error(f"Error processing request: {str(e)}")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("👋 Welcome! Please upload a document or code file to begin analysis.")
    st.markdown("### Supported Formats:")
    cols = st.columns(3)
    file_types = {
    "📄 Documents": ["PDF", "TXT"],
    "💻 Code Files": ["Python", "JavaScript", "C/C++", "Java", "C#", "PHP", "Ruby", "Go"],
    "📈 Data Formats": ["CSV", "XLS", "XLSX"],
    "📊 Presentations": ["PPTX"]

}
    
    for col, (category, formats) in zip(cols, file_types.items()):
        with col:
            st.markdown(f"**{category}**")
            for fmt in formats:
                st.markdown(f"- {fmt}")