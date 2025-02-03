import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.retrieval_qa.base import RetrievalQA

# Color palette
primary_color = "#1E90FF"
secondary_color = "#FF6347"
background_color = "#F5F5F5"
text_color = "#2C3E50"

# Custom CSS
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}
    .stButton>button {{
        background-color: {primary_color};
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .stTextInput>div>div>input {{
        border: 2px solid {primary_color};
        border-radius: 10px;
        padding: 12px;
        font-size: 16px;
    }}
    .stFileUploader>div>div>div>button {{
        background-color: {secondary_color};
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }}
    </style>
""", unsafe_allow_html=True)

# Streamlit app title
st.title("RAG System Builder with PDF and Code Support")

# File uploader for PDF and code files
uploaded_file = st.file_uploader(
    "Upload a file (PDF or code files: .java, .py, .c, .cpp)", 
    type=["pdf", "java", "py", "c", "cpp"]
)

if uploaded_file is not None:
    file_type = uploaded_file.type
    file_name = uploaded_file.name

    # Save uploaded file temporarily
    temp_file_path = f"temp_{file_name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    if file_type == "application/pdf":
        # Load PDF
        loader = PDFPlumberLoader(temp_file_path)
        docs = loader.load()
    else:
        # Load code files as plain text
        with open(temp_file_path, "r", encoding="utf-8") as f:
            code_content = f.read()
        docs = [{"page_content": code_content, "metadata": {"source": file_name}}]

    # Split into chunks
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = []
    for doc in docs:
        if isinstance(doc, dict):
            documents.append(doc["page_content"])
        else:
            documents.append(doc.page_content)
    documents = text_splitter.split_documents(documents)

    # Instantiate the embedding model
    embedder = HuggingFaceEmbeddings()

    # Create the vector store and fill it with embeddings
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Define the LLM
    llm = Ollama(model="deepseek-r1:1.5b")

    # Define the prompt
    prompt = """
    1. Use the following pieces of context to answer the question at the end.
    2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
    3. Keep the answer crisp and limited to 3-4 sentences.
    Context: {context}
    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    llm_chain = LLMChain(
        llm=llm,
        prompt=QA_CHAIN_PROMPT,
        callbacks=None,
        verbose=True)

    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\ncontent:{page_content}\nsource:{source}",
    )

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
        callbacks=None)

    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        verbose=True,
        retriever=retriever,
        return_source_documents=True)

    # User input
    user_input = st.text_input("Ask a question related to the file:")

    # Process user input
    if user_input:
        with st.spinner("Processing..."):
            response = qa(user_input)["result"]
            st.subheader("Response:")
            st.write(response)
else:
    st.info("Please upload a PDF or code file to proceed.")
