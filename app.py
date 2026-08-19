import os
import shutil
import subprocess
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Belge Yükleyiciler ve OCR Araçları
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
from pdf2image import convert_from_path
import pytesseract
import docx

# ---------------------------------------------------------
# 1. SAYFA VE UYGULAMA YAPILANDIRMASI
# ---------------------------------------------------------
st.set_page_config(page_title="Denizcilik SMS Asistanı", page_icon="🚢", layout="wide")
st.title("🚢 Denizcilik SMS & Prosedür Asistanı")

DOCS_DIR = "docs"
INDEX_DIR = "faiss_index"

# ---------------------------------------------------------
# 2. SİDEBAR VE API KEY KONTROLÜ
# ---------------------------------------------------------
st.sidebar.header("⚙️ Ayarlar")

openrouter_api_key = st.secrets.get("OPENROUTER_API_KEY") if "OPENROUTER_API_KEY" in st.secrets else None

if not openrouter_api_key:
    openrouter_api_key = st.sidebar.text_input("OpenRouter API Key", type="password", help="API anahtarınızı buraya girin.")

# ---------------------------------------------------------
# 3. EMBEDDINGS VE KAYNAK METADATA'SINI KORUYAN YÜKLEYİCİ
# ---------------------------------------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

def load_all_documents(folder_path):
    all_documents = []
    
    if not os.path.exists(folder_path):
        return all_documents, f"'{folder_path}' klasörü bulunamadı."

    files = os.listdir(folder_path)
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        ext = os.path.splitext(file)[1].lower()

        try:
            # A. PDF İşleme (Normal Metin veya Scan/OCR)
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                # Metadata temizleme/düzenleme (sadece dosya adını alalım)
                for d in docs:
                    d.metadata["source_file"] = file
                    if "page" in d.metadata:
                        d.metadata["page_label"] = f"Sayfa {d.metadata['page'] + 1}"
                    else:
                        d.metadata["page_label"] = "Belirtilmedi"

                # Okunan metin yetersizse OCR (Tesseract) çalıştır
                if not docs or sum(len(d.page_content.strip()) for d in docs) < 50:
                    st.info(f"🔍 Taranmış PDF/OCR İşleniyor: {file}")
                    images = convert_from_path(file_path)
                    ocr_docs = []
                    for i, image in enumerate(images):
                        text = pytesseract.image_to_string(image, lang="tur+eng")
                        if text.strip():
                            ocr_docs.append(Document(
                                page_content=text, 
                                metadata={"source_file": file, "page_label": f"Sayfa {i+1} (OCR)"}
                            ))
                    docs = ocr_docs
                
                all_documents.extend(docs)

            # B. Modern Word (.docx) İşleme
            elif ext == ".docx":
                st.info(f"📄 Word Belgesi (.docx) İşleniyor: {file}")
                try:
                    loader = Docx2txtLoader(file_path)
                    docx_docs = loader.load()
                    for d in docx_docs:
                        d.metadata["source_file"] = file
                        d.metadata["page_label"] = "Word Dokümanı"
                    all_documents.extend(docx_docs)
                except Exception:
                    # Yedek okuma yöntemi (python-docx)
                    doc = docx.Document(file_path)
                    full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                    if full_text.strip():
                        all_documents.append(Document(
                            page_content=full_text, 
                            metadata={"source_file": file, "page_label": "Word Dokümanı"}
                        ))

            # C. Eski Word (.doc) İşleme (Antiword)
            elif ext == ".doc":
                st.info(f"📄 Eski Format Word Belgesi (.doc) İşleniyor: {file}")
                try:
                    result = subprocess.run(["antiword", file_path], capture_output=True, text=True, check=True)
                    doc_text = result.stdout
                    if doc_text.strip():
                        all_documents.append(Document(
                            page_content=doc_text, 
                            metadata={"source_file": file, "page_label": "Eski Word (.doc)"}
                        ))
                except Exception as doc_err:
                    st.error(f"❌ Eski '.doc' dosyası okunamadı ({file}): {str(doc_err)}")

            # D. Excel (.xlsx) İşleme
            elif ext == ".xlsx":
                st.info(f"📊 Excel Belgesi İşleniyor: {file}")
                loader = UnstructuredExcelLoader(file_path, mode="single")
                excel_docs = loader.load()
                for d in excel_docs:
                    d.metadata["source_file"] = file
                    d.metadata["page_label"] = "Excel Sayfası"
                all_documents.extend(excel_docs)

        except Exception as e:
            st.error(f"❌ '{file}' işlenirken hata oluştu: {str(e)}")

    return all_documents, None

# ---------------------------------------------------------
# 4. FAISS İNDEKS YÖNETİMİ
# ---------------------------------------------------------
@st.cache_resource
def load_or_create_vectorstore():
    if os.path.exists(INDEX_DIR):
        return FAISS.load_local(
            INDEX_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    
    documents, error = load_all_documents(DOCS_DIR)

    if error or not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(INDEX_DIR)
    return vectorstore

vectorstore = load_or_create_vectorstore()

if st.sidebar.button("🔄 Vektör İndeksini Yeniden Oluştur"):
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    st.cache_resource.clear()
    st.rerun()

# ---------------------------------------------------------
# 5. RAG VE KAYNAK GÖSTERME YAPISI
# ---------------------------------------------------------
retriever = None
llm = None

if openrouter_api_key and vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2,
        default_headers={
            "HTTP-Referer": "https://streamlit.io",
            "X-Title": "Maritime Agent RAG"
        }
    )

    system_prompt = (
        "Sen uzman bir denizcilik ve operasyon asistansın.\n"
        "ÖNCELİKLİ GÖREVİN: Aşağıda verilen şirket dokümanı bağlamını (context) kullanarak yanıt vermek.\n"
        "EĞER sorunun cevabı şirket dokümanlarında TAM OLARAK yoksa veya eksikse:\n"
        "1. Önce şirket dokümanlarında geçen ilgili kısımları aktar.\n"
        "2. Ardından genel denizcilik mevzuatı (IMO, SOLAS, MARPOL, STCW vb.) ve genel bilgi birikimini kullanarak eksik kısımları tamamla.\n"
        "3. Yanıtında hangi bilgilerin şirket dokümanından, hangi bilgilerin genel denizcilik bilgisinden geldiğini açıkça belirt.\n\n"
        "Bağlam:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

# ---------------------------------------------------------
# 6. SOHBET GEÇMİŞİ VE ARAYÜZ
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("SMS prosedürleri, Word (.doc/.docx) veya Excel belgeleri hakkında soru sorun..."):
    if not openrouter_api_key:
        st.error("⚠️ Lütfen sol menüden OpenRouter API anahtarınızı girin.")
    elif not vectorstore:
        st.error("⚠️ 'docs' klasöründe okunabilir doküman bulunamadı. Lütfen dosyaları 'docs' klasörüne yükleyip indeksi yenileyin.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Dokümanlar taranıyor ve referanslar hazırlanıyor..."):
                try:
                    # 1. Alakalı Dokümanları Çek
                    relevant_docs = retriever.invoke(user_input)
                    
                    # 2. Bağlam Metnini Oluştur
                    context_text = "\n\n".join([d.page_content for d in relevant_docs])
                    
                    # 3. LLM Yanıtını Üret
                    formatted_prompt = prompt.format(context=context_text, question=user_input)
                    response_text = llm.invoke(formatted_prompt).content
                    
                    # 4. Kaynak Bilgilerini (Citation) Derle
                    sources = []
                    seen = set()
                    for doc in relevant_docs:
                        file_name = doc.metadata.get("source_file", doc.metadata.get("source", "Bilinmeyen Dosya"))
                        page_info = doc.metadata.get("page_label", "")
                        source_str = f"📄 **{file_name}** ({page_info})"
                        
                        if source_str not in seen:
                            seen.add(source_str)
                            sources.append(source_str)

                    # 5. Yanıta Kaynakları Ekle
                    final_response = response_text
                    if sources:
                        final_response += "\n\n---\n**📚 Kullanılan Kaynaklar:**\n" + "\n".join([f"- {src}" for src in sources])

                    st.markdown(final_response)
                    st.session_state.messages.append({"role": "assistant", "content": final_response})

                except Exception as e:
                    st.error(f"Yanıt oluşturulurken bir hata oluştu: {str(e)}")
