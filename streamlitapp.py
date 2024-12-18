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
        # Load PDF data
        loader = PyMuPDFLoader(pdf_file) 
        documents = loader.load()

        # If no documents are loaded, try OCR with OpenAI
        if not documents:
            st.warning(f"No extractable text found in '{pdf_file.name}'. Attempting OCR with OpenAI...")
            documents = self.perform_ocr_with_openai(pdf_file)

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

    def perform_ocr_with_openai(self, pdf_file):
        # Convert PDF to images and perform OCR using OpenAI
        images = []
        extracted_texts = []
        with tempfile.TemporaryDirectory() as temp_dir:
            # Load PDF and convert each page to an image
            loader = PyMuPDFLoader(pdf_file)
            pdf_document = loader.load()
            for i, page in enumerate(pdf_document):
                image_path = os.path.join(temp_dir, f"page_{i}.png")
                page.save(image_path)  # Save page as image
                images.append(image_path)

            # Perform OCR on each image using OpenAI
            for image_path in images:
                with open(image_path, "rb") as image_file:
                    response = openai.Image.create(
                        file=image_file,
                        model="text-davinci-003",  # Use the appropriate model
                        prompt="Extract text from this image.",
                        n=1,
                        size="1024x1024"
                    )
                    text = response['data'][0]['text']
                    if text.strip():  # Only add non-empty texts
                        extracted_texts.append(text)

        return extracted_texts

# Streamlit application
def main():
    st.title("PDF Vectorizer with OpenAI OCR")
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
