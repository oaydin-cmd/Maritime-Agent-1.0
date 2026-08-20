import os
import shutil
import io
import datetime
import sqlite3
import urllib.request
import ssl
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

# SSL sertifika kısıtlamaları olan ağlarda bağlantı engellerini esnetir
os.environ["CURL_CA_BUNDLE"] = ""

# ---------------------------------------------------------
# 1. SAYFA YAPILANDIRMASI VE AĞ KONTROLÜ
# ---------------------------------------------------------
st.set_page_config(page_title="Denizcilik SMS & Exper Asistanı", page_icon="⚓", layout="wide")
st.title("⚓ Denizcilik Teknik & SMS Uzman Asistanı (DeepSeek)")

DOCS_DIR = "docs"
INDEX_DIR = "faiss_index"
DB_PATH = "chat_history.db"

def check_internet_connection():
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        urllib.request.urlopen("https://api.deepseek.com", timeout=5, context=ctx)
        return True, "Erişim başarılı."
    except Exception as e:
        return False, f"Ağ uyarısı: {str(e)}"

# ---------------------------------------------------------
# 2. SQLITE VERİTABANI (KALICI SOHBET HAFIZASI)
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            role TEXT,
            content TEXT,
            docx_blob BLOB,
            file_name TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_message_to_db(role, content, docx_bytes=None, file_name=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    binary_data = sqlite3.Binary(docx_bytes) if isinstance(docx_bytes, bytes) else None
    cursor.execute(
        "INSERT INTO messages (timestamp, role, content, docx_blob, file_name) VALUES (?, ?, ?, ?, ?)",
        (now, role, content, binary_data, file_name)
    )
    conn.commit()
    conn.close()

def load_messages_from_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, role, content, docx_blob, file_name FROM messages ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    
    messages = []
    for row in rows:
        msg = {
            "id": row[0],
            "timestamp": row[1],
            "role": row[2],
            "content": row[3]
        }
        if row[4] and isinstance(row[4], (bytes, bytearray)) and len(row[4]) > 0:
            msg["docx_bytes"] = bytes(row[4])
            msg["file_name"] = row[5] or "SMS_Risk_Assessment.docx"
        messages.append(msg)
    return messages

def clear_db_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages")
    conn.commit()
    conn.close()

init_db()

# ---------------------------------------------------------
# 3. YALIN SOL MENÜ VE SECRETS KONTROLÜ
# ---------------------------------------------------------
st.sidebar.header("⚙️ Sistem Durumu")

net_ok, net_msg = check_internet_connection()
if net_ok:
    st.sidebar.success("🌐 İnternet Bağlantısı Aktif")
else:
    st.sidebar.warning(f"⚠️ Bağlantı Sorunu: {net_msg}")

api_key = st.secrets.get("DEEPSEEK_API_KEY") or st.secrets.get("OPENROUTER_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENROUTER_API_KEY")

if api_key:
    st.sidebar.success("🔑 API Key Tanımlı (Secrets)")
else:
    st.sidebar.error("❌ API Key Bulunamadı! Lütfen `.streamlit/secrets.toml` dosyanızı kontrol edin.")

st.sidebar.markdown("---")
st.sidebar.header("🧠 Hafıza Yönetimi")

if st.sidebar.button("🗑️ Tüm Sohbet Hafızasını Sıfırla"):
    clear_db_history()
    st.session_state.messages = []
    st.success("Hafıza temizlendi.")
    st.rerun()

# ---------------------------------------------------------
# 4. EMBEDDINGS VE DOKÜMAN İNDEKSLEME
# ---------------------------------------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

def load_all_documents(folder_path):
    all_documents = []
    if not os.path.exists(folder_path):
        return all_documents, f"'{folder_path}' klasörü bulunamadı."

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        ext = os.path.splitext(file)[1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                for d in docs:
                    d.metadata["source_file"] = file
                    d.metadata["page_label"] = f"Sayfa {d.metadata.get('page', 0) + 1}"
                if not docs or sum(len(d.page_content.strip()) for d in docs) < 50:
                    images = convert_from_path(file_path)
                    ocr_docs = []
                    for i, image in enumerate(images):
                        text = pytesseract.image_to_string(image, lang="tur+eng")
                        if text.strip():
                            ocr_docs.append(Document(page_content=text, metadata={"source_file": file, "page_label": f"Sayfa {i+1} (OCR)"}))
                    docs = ocr_docs
                all_documents.extend(docs)

            elif ext == ".docx":
                try:
                    loader = Docx2txtLoader(file_path)
                    docx_docs = loader.load()
                    for d in docx_docs:
                        d.metadata["source_file"] = file
                        d.metadata["page_label"] = "Word Dokümanı"
                    all_documents.extend(docx_docs)
                except Exception:
                    doc = docx.Document(file_path)
                    full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                    if full_text.strip():
                        all_documents.append(Document(page_content=full_text, metadata={"source_file": file, "page_label": "Word Dokümanı"}))

            elif ext == ".xlsx":
                loader = UnstructuredExcelLoader(file_path, mode="single")
                excel_docs = loader.load()
                for d in excel_docs:
                    d.metadata["source_file"] = file
                    d.metadata["page_label"] = "Excel Sayfası"
                all_documents.extend(excel_docs)
        except Exception as e:
            st.error(f"❌ '{file}' işlenirken hata: {str(e)}")

    return all_documents, None

@st.cache_resource
def load_or_create_vectorstore():
    if os.path.exists(INDEX_DIR):
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    documents, error = load_all_documents(DOCS_DIR)
    if error or not documents:
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(INDEX_DIR)
    return vectorstore

vectorstore = load_or_create_vectorstore()

if st.sidebar.button("🔄 Doküman İndeksini Yenile"):
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    st.cache_resource.clear()
    st.rerun()

# ---------------------------------------------------------
# 5. AKILLI API & MODEL YÖNLENDİRMESİ
# ---------------------------------------------------------
retriever = None

if api_key and vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    if api_key.startswith("sk-or-"):
        api_base = "https://openrouter.ai/api/v1"
        target_models = ["deepseek/deepseek-chat", "deepseek/deepseek-r1:free", "deepseek/deepseek-chat:free"]
    else:
        api_base = "https://api.deepseek.com"
        target_models = ["deepseek-chat", "deepseek-reasoner"]

    system_prompt = (
        "Sen uzun yıllar ocean-going gemilerde Süvari/Başmühendis olarak görev yapmış, şu an ise Kıdemli DPA, "
        "SIRE 2.0 Enspektörü ve Deniz Emniyeti Uzmanısın.\n\n"
        "YAKLAŞIMIN VE MİSYONUN:\n"
        "1. BİREBİR ALINTI YAPMA: Verilen SMS ve teknik dokümanlardaki metinleri kopyala-yapıştır yapma! "
        "Bilgiyi özümse, kendi mesleki tecrübenle harmanla ve bir deniz uzmanı gibi pratik, sektörel bir dille açıkla.\n"
        "2. OPERASYONEL YORUM EKLE: Kuralın veya prosedürün sadece ne olduğunu değil; Neden önemli olduğunu, "
        "güvertede/makinede uygularken yapılan tipik hataları ve bir PSC/SIRE denetiminde enspektörün burayı nasıl sorgulayacağını belirt.\n"
        "3. PRATİK TAVSİYE VER: Kullanıcıya sadece teorik bilgi sunma, vardiya zabitinin veya çarkçının sahada uygulayabileceği somut adımlar öner.\n"
        "4. KİŞİSEL HAFIZAYI KULLAN: Kullanıcının daha önceki mesajlarda bahsettiği gemi tipi, rütbesi veya özel durumlarını yanıtlarda dikkate al.\n\n"
        "GEÇMİŞ SOHBET HAFIZASI:\n{chat_history}\n\n"
        "REFERANS SMS VE TEKNİK DOKÜMAN İÇERİĞİ:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

# ---------------------------------------------------------
# 6. SOHBET ARAYÜZÜ VE İŞLEME
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = load_messages_from_db()

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Gemi operasyonları, SMS prosedürleri veya denetimler hakkında bir soru sorun..."):
    if not api_key:
        st.error("⚠️ `.streamlit/secrets.toml` içinde geçerli bir API Key bulunamadı.")
    elif not vectorstore:
        st.error("⚠️ Doküman bulunamadı. Lütfen 'docs' klasörüne PDF ekleyin.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        save_message_to_db("user", user_input)
        
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("DeepSeek SMS dokümanlarını ve denizcilik prosedürlerini inceliyor..."):
                relevant_docs = retriever.invoke(user_input)
                context_text = "\n\n".join([d.page_content for d in relevant_docs]) if relevant_docs else "İlgili SMS dokümanı bulunamadı."
                
                recent_history = load_messages_from_db()[-10:]
                history_str = "\n".join([f"{m['timestamp']} - {m['role'].upper()}: {m['content']}" for m in recent_history])

                formatted_prompt = prompt.format(context=context_text, chat_history=history_str, question=user_input)
                
                llm_response = None
                last_error = ""

                for model_name in target_models:
                    try:
                        llm = ChatOpenAI(
                            model=model_name,
                            openai_api_key=api_key,
                            openai_api_base=api_base,
                            temperature=0.3,
                            timeout=60,
                            max_retries=1,
                            default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "Maritime Expert Assistant"}
                        )
                        llm_response = llm.invoke(formatted_prompt)
                        break
                    except Exception as err:
                        last_error = str(err)
                        continue

                if llm_response:
                    response_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

                    sources = []
                    seen = set()
                    for doc in relevant_docs:
                        file_name = doc.metadata.get("source_file", doc.metadata.get("source", "Doküman"))
                        page_info = doc.metadata.get("page_label", "")
                        source_str = f"📄 {file_name} ({page_info})"
                        if source_str not in seen:
                            seen.add(source_str)
                            sources.append(source_str)

                    final_response = response_text
                    if sources:
                        final_response += "\n\n---\n**İncelediğim Şirket Dokümanları:**\n" + "\n".join([f"- {src}" for src in sources])

                    st.markdown(final_response)
                    save_message_to_db("assistant", final_response)
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                else:
                    st.error(f"❌ API Hatası: {last_error if last_error else 'Servis yanıt vermedi.'}\n\nLütfen secrets dosyasındaki API key'inizi ve hesabınızdaki bakiye/kota durumunu kontrol edin.")
