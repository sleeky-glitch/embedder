import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader  
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec  
from langchain.text_splitter import CharacterTextSplitter
from PIL import Image
import openai
import tempfile
import os
import easyocr  # Import EasyOCR

class PDFEmbedder:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.index_name = "gujpaperembeddings"

        # Initialize Pinecone client using st.secrets
        self.pc = PineconeClient(api_key=st.secrets["PINECONE_API_KEY"]) 
        # Create Pinecone index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # OpenAI embeddings typically have a dimension of 1536
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws', 
                    region='us-east-1'  
                )
            )

    def embed_pdf(self, pdf_file):
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.getbuffer())  # Write the uploaded file to the temp file
            temp_file_path = temp_file.name  # Get the path of the temp file

        # Load PDF data
        loader = PyMuPDFLoader(temp_file_path) 
        documents = loader.load()

        # If no documents are loaded, try OCR with EasyOCR
        if not documents:
            st.warning(f"No extractable text found in '{pdf_file.name}'. Attempting OCR with EasyOCR...")
            documents = self.perform_ocr_with_easyocr(temp_file_path)

        # Debugging: Check the number of documents loaded
        st.write(f"Loaded {len(documents)} documents from '{pdf_file.name}'.")

        # Split documents into smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=4)
        docs = text_splitter.split_documents(documents)

        # Debugging: Check the number of chunks created
        st.write(f"Created {len(docs)} document chunks from '{pdf_file.name}'.")

        # Initialize Pinecone index with documents
        self.docsearch = Pinecone.from_documents(docs, self.embeddings, index_name=self.index_name)
        return len(docs)

    def perform_ocr_with_easyocr(self, pdf_file_path):
        # Convert PDF to images and perform OCR using EasyOCR
        reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader for English
        images = []
        extracted_texts = []
        with tempfile.TemporaryDirectory() as temp_dir:
            # Load PDF and convert each page to an image
            loader = PyMuPDFLoader(pdf_file_path)
            pdf_document = loader.load()
            for i, page in enumerate(pdf_document):
                image_path = os.path.join(temp_dir, f"page_{i}.png")
                page.save(image_path)  # Save page as image
                images.append(image_path)

            # Perform OCR on each image using EasyOCR
            for image_path in images:
                result = reader.readtext(image_path)
                text = " ".join([res[1] for res in result])  # Extract text from the result
                if text.strip():  # Only add non-empty texts
                    extracted_texts.append(text)

        return extracted_texts

# Streamlit application
def main():
    st.title("PDF Vectorizer with EasyOCR")
    st.write("Upload your PDF files to vectorize them using OpenAI embeddings and store them in Pinecone.")

    pdf_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if st.button("Upload and Vectorize"):
        if pdf_files:
            embedder = PDFEmbedder()
            total_chunks = 0
            for pdf_file in pdf_files:
                # Embed the PDF directly from the UploadedFile object
                chunks = embedder.embed_pdf(pdf_file)
                total_chunks += chunks

            st.success(f"Successfully embedded {total_chunks} document chunks into Pinecone index '{embedder.index_name}'.")
        else:
            st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
