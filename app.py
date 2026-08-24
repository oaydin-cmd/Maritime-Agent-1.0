import os
import shutil
import io
import datetime
import sqlite3
import pandas as pd
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# EnsembleRetriever ve BM25Retriever doğru paket olan langchain_community üzerinden çekiliyor
from langchain_community.retrievers import EnsembleRetriever, BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
from pdf2image import convert_from_path
import pytesseract
import docx

os.environ["CURL_CA_BUNDLE"] = ""

# ---------------------------------------------------------
# 1. SAYFA YAPILANDIRMASI
# ---------------------------------------------------------
st.set_page_config(page_title="Denizcilik SMS & Asistan", page_icon="⚓", layout="wide")
st.title("⚓ Denizcilik & Sohbet Asistanı (Hybrid RAG v2.0)")

DOCS_DIR = "docs"
INDEX_DIR = "faiss_index"
DB_PATH = "chat_history.db"

# ---------------------------------------------------------
# DOKÜMAN OLUŞTURMA FONKSİYONLARI (WORD & EXCEL)
# ---------------------------------------------------------
def create_docx_bytes(content_text, title="Denizcilik SMS Raporu"):
    doc = docx.Document()
    doc.add_heading(title, 0)
    for line in content_text.split("\n"):
        if line.startswith("# "):
            doc.add_heading(line.replace("# ", "").strip(), level=1)
        elif line.startswith("## "):
            doc.add_heading(line.replace("## ", "").strip(), level=2)
        elif line.startswith("### "):
            doc.add_heading(line.replace("### ", "").strip(), level=3)
        else:
            doc.add_paragraph(line)
            
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

def create_excel_bytes(content_text):
    lines = [line.strip() for line in content_text.split("\n") if line.strip()]
    data = []
    
    for idx, line in enumerate(lines, 1):
        clean_line = line.lstrip("*-•1234567890. ").strip()
        if clean_line and not clean_line.startswith("---") and not clean_line.startswith("İncelediğim Şirket"):
            data.append({"No": idx, "İçerik / Madde / Detay": clean_line})
        
    df = pd.DataFrame(data if data else [{"No": 1, "İçerik / Madde / Detay": content_text}])
    
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Rapor_Verisi')
    return bio.getvalue()

# ---------------------------------------------------------
# 2. OTOMATİK ONARIMLI SQLITE VERİTABANI
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
    table_exists = cursor.fetchone()
    
    if table_exists:
        cursor.execute("PRAGMA table_info(messages)")
        columns = [column[1] for column in cursor.fetchall()]
        if "excel_blob" not in columns:
            cursor.execute("DROP TABLE messages")
            conn.commit()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            role TEXT,
            content TEXT,
            docx_blob BLOB,
            docx_file_name TEXT,
            excel_blob BLOB,
            excel_file_name TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_message_to_db(role, content, docx_bytes=None, docx_file_name=None, excel_bytes=None, excel_file_name=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    docx_binary = sqlite3.Binary(docx_bytes) if isinstance(docx_bytes, bytes) else None
    excel_binary = sqlite3.Binary(excel_bytes) if isinstance(excel_bytes, bytes) else None
    
    cursor.execute(
        "INSERT INTO messages (timestamp, role, content, docx_blob, docx_file_name, excel_blob, excel_file_name) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (now, role, content, docx_binary, docx_file_name, excel_binary, excel_file_name)
    )
    conn.commit()
    conn.close()

def load_messages_from_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, role, content, docx_blob, docx_file_name, excel_blob, excel_file_name FROM messages ORDER BY id ASC")
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
            msg["docx_file_name"] = row[5] or "SMS_Raporu.docx"
        if row[6] and isinstance(row[6], (bytes, bytearray)) and len(row[6]) > 0:
            msg["excel_bytes"] = bytes(row[6])
            msg["excel_file_name"] = row[7] or "SMS_Raporu.xlsx"
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
# 3. KATEGORİ SEÇİMLİ SOL MENÜ
# ---------------------------------------------------------
st.sidebar.header("⚙️ Sistem Ayarları")

api_key = st.secrets.get("OPENROUTER_API_KEY") or st.secrets.get("DEEPSEEK_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("DEEPSEEK_API_KEY")

if api_key:
    st.sidebar.success("🔑 API Key Tanımlı")
else:
    st.sidebar.error("❌ API Key Bulunamadı!")

st.sidebar.markdown("---")
st.sidebar.header("📂 Doküman Filtresi")

categories = ["Tüm Dokümanlar"]
if os.path.exists(DOCS_DIR):
    subfolders = [f for f in os.listdir(DOCS_DIR) if os.path.isdir(os.path.join(DOCS_DIR, f))]
    categories.extend(subfolders)

selected_category = st.sidebar.selectbox("Arama Yapılacak Kategori:", categories)

st.sidebar.markdown("---")
st.sidebar.header("🧠 Hafıza Yönetimi")

if st.sidebar.button("🗑️ Tüm Sohbet Hafızasını Sıfırla"):
    clear_db_history()
    st.session_state.messages = []
    st.success("Hafıza temizlendi.")
    st.rerun()

# ---------------------------------------------------------
# 4. KLASÖR BAZLI EMBEDDINGS VE HYBRID DOKÜMAN İNDEKSLEME
# ---------------------------------------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

def load_all_documents(folder_path):
    all_documents = []
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return all_documents, None

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            rel_path = os.path.relpath(root, folder_path)
            category = rel_path if rel_path != "." else "Genel"

            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    for d in docs:
                        d.metadata["source_file"] = file
                        d.metadata["category"] = category
                        d.metadata["page_label"] = f"Sayfa {d.metadata.get('page', 0) + 1}"
                    if not docs or sum(len(d.page_content.strip()) for d in docs) < 50:
                        try:
                            images = convert_from_path(file_path)
                            ocr_docs = []
                            for i, image in enumerate(images):
                                text = pytesseract.image_to_string(image, lang="tur+eng")
                                if text.strip():
                                    ocr_docs.append(Document(page_content=text, metadata={"source_file": file, "category": category, "page_label": f"Sayfa {i+1} (OCR)"}))
                            docs = ocr_docs
                        except Exception:
                            pass
                    all_documents.extend(docs)

                elif ext == ".docx":
                    try:
                        loader = Docx2txtLoader(file_path)
                        docx_docs = loader.load()
                        for d in docx_docs:
                            d.metadata["source_file"] = file
                            d.metadata["category"] = category
                            d.metadata["page_label"] = "Word Dokümanı"
                        all_documents.extend(docx_docs)
                    except Exception:
                        doc = docx.Document(file_path)
                        full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                        if full_text.strip():
                            all_documents.append(Document(page_content=full_text, metadata={"source_file": file, "category": category, "page_label": "Word Dokümanı"}))

                elif ext == ".xlsx":
                    loader = UnstructuredExcelLoader(file_path, mode="single")
                    excel_docs = loader.load()
                    for d in excel_docs:
                        d.metadata["source_file"] = file
                        d.metadata["category"] = category
                        d.metadata["page_label"] = "Excel Sayfası"
                    all_documents.extend(excel_docs)
            except Exception as e:
                st.error(f"❌ '{file}' işlenirken hata: {str(e)}")

    return all_documents, None

@st.cache_resource
def load_or_create_retrievers():
    documents, error = load_all_documents(DOCS_DIR)
    if error or not documents:
        return None, None, []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    splits = text_splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 5
    
    return vectorstore, bm25_retriever, splits

vectorstore, bm25_retriever, all_splits = load_or_create_retrievers()

if st.sidebar.button("🔄 Doküman İndeksini Yenile"):
    st.cache_resource.clear()
    st.rerun()

# ---------------------------------------------------------
# 5. AKILLI HYBRID RETRIEVER VE PROMPT
# ---------------------------------------------------------
if api_key and vectorstore:
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )

    if api_key.startswith("sk-or-"):
        api_base = "https://openrouter.ai/api/v1"
        target_models = [
            "google/gemini-2.0-flash-001",
            "meta-llama/llama-3.3-70b-instruct",
            "deepseek/deepseek-chat"
        ]
    else:
        api_base = "https://api.deepseek.com"
        target_models = ["deepseek-chat", "deepseek-reasoner"]

    current_time_str = datetime.datetime.now().strftime("%d.%m.%Y, %A")

    system_prompt = (
        f"Bugünün tarihi ve günü: {current_time_str}.\n"
        "Sen akıllı, zeki ve uzman bir denizcilik/SMS asistanısın.\n\n"
        "KURALLAR:\n"
        "1. 'REFERANS DOKÜMAN' alanında sunulan tüm bilgileri dikkatlice tara.\n"
        "2. Kullanıcının sorduğu bulgu, madde numarası (örn: 4.13), kural veya doküman detayını eksiksiz yanıtla.\n"
        "3. Dokümanlarda bilgi varsa asla 'dokümanı göremiyorum' deme; doğrudan maddeleri çıkar ve analiz et.\n"
        "4. Yanıtı düzenli, başlıklandırılmış ve okunabilir formatta ver.\n\n"
        "REFERANS DOKÜMAN:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

# ---------------------------------------------------------
# 6. SOHBET ARAYÜZÜ
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = load_messages_from_db()

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        col1, col2 = st.columns(2)
        if message.get("docx_bytes"):
            with col1:
                st.download_button(
                    label="📥 Word Formatında İndir (.docx)",
                    data=message["docx_bytes"],
                    file_name=message.get("docx_file_name", "SMS_Raporu.docx"),
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key=f"dl_doc_{idx}"
                )
        if message.get("excel_bytes"):
            with col2:
                st.download_button(
                    label="📊 Excel Formatında İndir (.xlsx)",
                    data=message["excel_bytes"],
                    file_name=message.get("excel_file_name", "SMS_Raporu.xlsx"),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_xls_{idx}"
                )

if user_input := st.chat_input("Mesajınızı yazın..."):
    if not api_key:
        st.error("⚠️ API Key tanımlı değil.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        save_message_to_db("user", user_input)
        
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Dokümanlar taranıyor ve yanıt hazırlanıyor..."):
                
                relevant_docs = []
                if vectorstore:
                    fetched_docs = ensemble_retriever.invoke(user_input)
                    
                    if selected_category != "Tüm Dokümanlar":
                        relevant_docs = [d for d in fetched_docs if d.metadata.get("category") == selected_category]
                    else:
                        relevant_docs = fetched_docs
                
                context_text = "\n\n".join([f"[{d.metadata.get('source_file')} - {d.metadata.get('page_label')}]:\n" + d.page_content for d in relevant_docs]) if relevant_docs else "Yok"

                history_objs = []
                raw_history = load_messages_from_db()[:-1]
                for m in raw_history[-8:]:
                    if m["role"] == "user":
                        history_objs.append(HumanMessage(content=m["content"]))
                    else:
                        history_objs.append(AIMessage(content=m["content"]))

                formatted_prompt = prompt.format_messages(
                    context=context_text,
                    chat_history=history_objs,
                    question=user_input
                )
                
                llm_response = None
                last_error = ""

                for model_name in target_models:
                    try:
                        llm = ChatOpenAI(
                            model=model_name,
                            openai_api_key=api_key,
                            openai_api_base=api_base,
                            temperature=0.3,
                            timeout=20,
                            max_retries=0,
                            default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "Expert Assistant"}
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
                        file_name = doc.metadata.get("source_file", "Doküman")
                        page_info = doc.metadata.get("page_label", "")
                        cat_info = doc.metadata.get("category", "Genel")
                        source_str = f"📄 [{cat_info}] {file_name} ({page_info})"
                        if source_str not in seen:
                            seen.add(source_str)
                            sources.append(source_str)

                    final_response = response_text
                    if sources:
                        final_response += "\n\n---\n**İncelediğim Şirket Dokümanları:**\n" + "\n".join([f"- {src}" for src in sources])

                    st.markdown(final_response)

                    word_keywords = ["form", "rapor", "hazırla", "oluştur", "docx", "word", "maddeler", "checklist", "incele"]
                    excel_keywords = ["excel", "tablo", "xlsx", "liste", "listele", "maddeler", "rapor"]

                    docx_bytes, docx_file_name = None, None
                    excel_bytes, excel_file_name = None, None
                    time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')

                    if any(kw in user_input.lower() for kw in word_keywords):
                        docx_file_name = f"SMS_Rapor_{time_stamp}.docx"
                        docx_bytes = create_docx_bytes(final_response, title="SMS & Denizcilik Asistan Raporu")

                    if any(kw in user_input.lower() for kw in excel_keywords):
                        excel_file_name = f"SMS_Rapor_{time_stamp}.xlsx"
                        excel_bytes = create_excel_bytes(final_response)

                    col1, col2 = st.columns(2)
                    if docx_bytes:
                        with col1:
                            st.download_button(
                                label="📥 Word Formatında İndir (.docx)",
                                data=docx_bytes,
                                file_name=docx_file_name,
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                key=f"dl_new_doc_{len(st.session_state.messages)}"
                            )
                    if excel_bytes:
                        with col2:
                            st.download_button(
                                label="📊 Excel Formatında İndir (.xlsx)",
                                data=excel_bytes,
                                file_name=excel_file_name,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"dl_new_xls_{len(st.session_state.messages)}"
                            )

                    save_message_to_db(
                        role="assistant", 
                        content=final_response, 
                        docx_bytes=docx_bytes, 
                        docx_file_name=docx_file_name,
                        excel_bytes=excel_bytes,
                        excel_file_name=excel_file_name
                    )
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": final_response,
                        "docx_bytes": docx_bytes,
                        "docx_file_name": docx_file_name,
                        "excel_bytes": excel_bytes,
                        "excel_file_name": excel_file_name
                    })
                else:
                    st.error(f"❌ API Hatası: {last_error if last_error else 'Servis yanıt vermedi.'}")
