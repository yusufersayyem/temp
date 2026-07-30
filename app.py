import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# تحميل متغيرات البيئة
load_dotenv()

DATA_PATH = "data/"
COLLECTION_NAME = "my_pdf_documents"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Step 1: Load raw PDF(s)
def load_pdf_files(data_path):
    print("⏳ جاري تحميل ملفات الـ PDF...")
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    print(f"✅ تم تحميل {len(documents)} صفحة.")
    return documents

# Step 2: Create Chunks
def create_chunks(extracted_data):
    print("⏳ جاري تقطيع النصوص (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    print(f"✅ تم إنشاء {len(text_chunks)} قطعة نصية.")
    return text_chunks

# Step 3: Initialize Embedding Model
def get_embedding_model():
    return MistralAIEmbeddings(model="mistral-embed")

if __name__ == "__main__":
    # Implementation of stages
    documents = load_pdf_files(DATA_PATH)
    text_chunks = create_chunks(documents)
    embedding_model = get_embedding_model()

    # Step 4: Upload & Store embeddings in Qdrant Cloud
    print("Texts are being converted to vectors and uploaded to Qdrant Cloud...")
    
    db = QdrantVectorStore.from_documents(
        documents=text_chunks,
        embedding=embedding_model,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME
    )

    print("The data has been successfully uploaded and stored in Qdrant Cloud!")
