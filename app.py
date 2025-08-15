import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from transformers import pipeline

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# Streamlit UI
st.set_page_config(page_title="📚 Q&A Chatbot", layout="centered")
st.title("📚 DocuMind AI chatbot")
st.write("Upload your PDF and ask questions. Your document, your answers.")

# PDF upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)

    if not text.strip():
        st.error("No readable text found in the PDF.")
    else:
        # Split text into chunks
        splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}  # change to "cuda" for GPU
        )

        # Create vector store
        vector_store = Chroma.from_texts(chunks, embeddings)

        # Load FLAN-T5 locally
        flan_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=256,
            temperature=0.1
        )

        llm = HuggingFacePipeline(pipeline=flan_pipeline)

        # Prompt for QA
        prompt = ChatPromptTemplate.from_template("""
        Use the provided context to answer the question.
        If you don’t know, say "I don't know".

        Context:
        {context}

        Question:
        {input}
        """)

        # Document chain
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Retrieval chain
        retriever = vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Ask question
        question = st.text_input("Ask a question about the PDF:")
        if question:
            with st.spinner("Thinking..."):
                result = retrieval_chain.invoke({"input": question})

            # Handle different output formats
            answer = result.get("answer") or result.get("result") or "No answer found."
            st.success(answer)
