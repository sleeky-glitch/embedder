import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader  
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec  
from langchain.text_splitter import CharacterTextSplitter

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

        # Split documents into smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=4)
        docs = text_splitter.split_documents(documents)

        # Initialize Pinecone index with documents
        self.docsearch = Pinecone.from_documents(docs, self.embeddings, index_name=self.index_name)
        return len(docs)

# Streamlit application
def main():
    st.title("PDF Vectorizer")
    st.write("Upload your PDF files to vectorize them using OpenAI embeddings and store them in Pinecone.")

    pdf_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if st.button("Upload and Vectorize"):
        if pdf_files:
            embedder = PDFEmbedder()
            total_chunks = 0
            for pdf_file in pdf_files:
                # Save the uploaded file temporarily
                with open(pdf_file.name, "wb") as f:
                    f.write(pdf_file.getbuffer())

                # Embed the PDF
                chunks = embedder.embed_pdf(pdf_file.name)
                total_chunks += chunks

                # Optionally, remove the temporary file
                os.remove(pdf_file.name)

            st.success(f"Successfully embedded {total_chunks} document chunks into Pinecone index '{embedder.index_name}'.")
        else:
            st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
