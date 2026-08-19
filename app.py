import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Özel Yükleyiciler ve OCR Araçları
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from pdf2image import convert_from_path
import pytesseract

st.set_page_config(page_title="Denizcilik SMS Asistanı", page_icon="🚢")
st.title("🚢 Denizcilik SMS Asistanı")

DOCS_DIR = "docs"
INDEX_DIR = "faiss_index"

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

# Klasördeki Tüm PDF ve DOCX Dosyalarını İşleyen Fonksiyon
def load_all_documents(folder_path):
    all_docs = []
    if not os.path.exists(folder_path):
        return all_docs
        
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        ext = os.path.splitext(file)[1].lower()

        try:
            # 1. PDF İşleme (Normal veya Taranmış/OCR)
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                # Eğer okunan metin çok kısa ise veya boşsa OCR çalıştır
                if not docs or sum(len(d.page_content.strip()) for d in docs) < 50:
                    st.info(f"🔍 Taranmış PDF algılandı, OCR yapılıyor: {file}")
                    images = convert_from_path(file_path)
                    ocr_text = ""
                    for i, image in enumerate(images):
                        text = pytesseract.image_to_string(image, lang="tur+eng")
                        ocr_text += f"\n--- Sayfa {i+1} ---\n" + text
                    
                    if ocr_text.strip():
                        docs = [Document(page_content=ocr_text, metadata={"source": file})]
                
                all_docs.extend(docs)

            # 2. Word (.docx) İşleme
            elif ext == ".docx":
                st.info(f"📄 Word belgesi işleniyor: {file}")
                loader = Docx2txtLoader(file_path)
                all_docs.extend(loader.load())

        except Exception as e:
            st.error(f"❌ {file} işlenirken hata oluştu: {str(e)}")

    return all_docs

@st.cache_resource
def load_or_create_vectorstore():
    if os.path.exists(INDEX_DIR):
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    
    documents = load_all_documents(DOCS_DIR)
    if not documents:
        st.warning("Doküman bulunamadı veya okunamadı.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(INDEX_DIR)
    st.success("Vektör veritabanı taranmış PDF ve Word desteğiyle oluşturuldu!")
    return vectorstore
