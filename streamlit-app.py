import streamlit as st
import fitz  # PyMuPDF for PDF extraction
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = st.secrets["GOOGLE_API_KEY"]

if not API_KEY:
    st.error("❌ API Key not found! Please check your .env file or Streamlit secrets.")
else:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

model = genai.GenerativeModel("gemini-1.5-pro")

st.title("📄 AI Document Query Assistant")

# Step 1: Upload PDF
uploaded_file = st.file_uploader("📂 Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("🔄 Processing your document..."):
        # Read PDF and extract text
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(text)

        # Convert text chunks into vector embeddings
        embedding_model = HuggingFaceEmbeddings()
        vector_store = FAISS.from_texts(chunks, embedding_model)

        st.success("✅ File processed successfully! You can now ask questions.")

        # Step 2: Query Input + "Get Answer" Button
        query = st.text_input("🔍 Enter your query:")

        if st.button("Get Answer"):
            if query.strip():
                with st.spinner("🔎 Searching for an answer..."):
                    # Perform similarity search
                    results = vector_store.similarity_search(query, k=3)
                    context = "\n".join([doc.page_content for doc in results])

                    # Generate answer using Gemini AI
                    response = model.generate_content(f"Context:\n{context}\n\nUser query: {query}")
                    answer = response.text

                    # Display answer
                    st.subheader("📢 Answer:")
                    st.write(answer)
            else:
                st.warning("⚠️ Please enter a query before clicking 'Get Answer'.")
