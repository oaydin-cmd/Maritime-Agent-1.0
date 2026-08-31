import os
import sqlite3
import datetime
import io
import docx
import requests
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# OpenRouter için ChatOpenAI
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------
# 1. STREAMLIT SAYFA YAPILANDIRMASI
# ---------------------------------------------------------
st.set_page_config(
    page_title="Maritime & SMS RAG Assistant",
    page_icon="⚓",
    layout="wide"
)

st.title("⚓ Denizcilik & SMS RAG Asistanı")

# ---------------------------------------------------------
# 2. SQLITE VERİTABANI VE MIGRATION
# ---------------------------------------------------------
DB_FILE = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            docx_bytes BLOB,
            file_name TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("PRAGMA table_info(messages)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if "docx_bytes" not in columns:
        cursor.execute("ALTER TABLE messages ADD COLUMN docx_bytes BLOB")
    if "file_name" not in columns:
        cursor.execute("ALTER TABLE messages ADD COLUMN file_name TEXT")
        
    conn.commit()
    conn.close()

def save_message_to_db(role, content, docx_bytes=None, file_name=None):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO messages (role, content, docx_bytes, file_name)
        VALUES (?, ?, ?, ?)
    """, (role, content, docx_bytes, file_name))
    conn.commit()
    conn.close()

def load_messages_from_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content, docx_bytes, file_name FROM messages ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    
    messages = []
    for r in rows:
        messages.append({
            "role": r[0],
            "content": r[1],
            "docx_bytes": r[2],
            "file_name": r[3]
        })
    return messages

def clear_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages")
    conn.commit()
    conn.close()

init_db()

# ---------------------------------------------------------
# 3. WORD (.DOCX) DOKÜMANI OLUŞTURUCU
# ---------------------------------------------------------
def create_docx_bytes(content_text, title="SMS & Denizcilik Asistan Raporu"):
    doc = docx.Document()
    doc.add_heading(title, level=0)
    doc.add_paragraph(f"Tarih: {datetime.datetime.now().strftime('%d.%m.%Y %H:%M')}")
    doc.add_paragraph("--------------------------------------------------")
    
    lines = content_text.split('\n')
    for line in lines:
        if line.startswith('### '):
            doc.add_heading(line.replace('### ', ''), level=3)
        elif line.startswith('## '):
            doc.add_heading(line.replace('## ', ''), level=2)
        elif line.startswith('# '):
            doc.add_heading(line.replace('# ', ''), level=1)
        elif line.startswith('- ') or line.startswith('* '):
            doc.add_paragraph(line[2:], style='List Bullet')
        else:
            doc.add_paragraph(line)
            
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

# ---------------------------------------------------------
# 4. DIRECT GEMINI API ÇAĞRISI (TOKEN / KEY HANDLER)
# ---------------------------------------------------------
HARDCODED_GEMINI_KEY = "AQ.Ab8RN6KiqPcMC_qS3DXOYuRe8EaNMIGoxrm-6f2hf3s2IWZGaQ"
gemini_api_key = st.secrets.get("GEMINI_API_KEY", HARDCODED_GEMINI_KEY)
openrouter_api_key = st.secrets.get("OPENROUTER_API_KEY", None)

def call_direct_gemini(prompt_text, api_key):
    """
    Hem standart API key hem de OAuth Bearer Token formatlarını destekleyen direkt REST çağrısı.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    # Eğer OAuth token ise Authorization header'ı ekle
    if api_key.startswith("AQ.") or api_key.startswith("ya29."):
        headers["Authorization"] = f"Bearer {api_key}"
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt_text}]
        }]
    }
    
    res = requests.post(url, headers=headers, json=payload)
    if res.status_code == 200:
        data = res.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    else:
        raise Exception(f"HTTP {res.status_code}: {res.text}")

@st.cache_resource
def load_vectorstore():
    index_path = "faiss_index"
    if os.path.exists(index_path):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.sidebar.error(f"Vektör Veritabanı Hatası: {e}")
            return None
    return None

vectorstore = load_vectorstore()

# ---------------------------------------------------------
# 5. YAN MENÜ (SIDEBAR) & MODEL SEÇİMİ
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Kontrol Paneli")
    
    st.subheader("📚 Doküman Veritabanı")
    if vectorstore:
        st.success("🟢 FAISS Veritabanı Aktif")
    else:
        st.warning("🟡 FAISS Veritabanı Bulunamadı")
    
    st.divider()
    
    st.subheader("🤖 AI Model Sağlayıcı")
    
    available_providers = []
    if gemini_api_key:
        available_providers.append("Google Gemini (Direct API)")
    if openrouter_api_key:
        available_providers.append("OpenRouter (Gemini 2.0 Flash Lite)")
        
    if available_providers:
        selected_provider = st.selectbox("Sağlayıcı Seçin:", available_providers)
    else:
        st.error("🔑 API Key bulunamadı.")
        selected_provider = None

    st.divider()

    st.subheader("🔍 Arama Hassasiyeti")
    k_docs = st.slider("Getirilecek Doküman Sayısı (k):", min_value=1, max_value=10, value=5)
    
    st.divider()

    if st.button("🗑️ Sohbet Geçmişini Temizle", use_container_width=True):
        clear_db()
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# 6. PROMPT YAPISI
# ---------------------------------------------------------
retriever = None
if vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_docs})

current_time_str = datetime.datetime.now().strftime("%d.%m.%Y, %A")

system_prompt = (
    f"Bugünün tarihi ve günü: {current_time_str}.\n"
    "Sen akıllı, zeki ve doğal yanıt veren bir asistansın. Şirket dokümanları, formlar, raporlar ve denizcilik/SMS konularında uzmansın.\n\n"
    "KURALLAR:\n"
    "1. Kullanıcı ne derse ona doğrudan, mantıklı ve insan gibi yanıt ver.\n"
    "2. Tarih, gün veya saat sorulduğunda sana verilen güncel tarih bilgisini kullan.\n"
    "3. Form, liste veya rapor istendiğinde düzenli, başlıklandırılmış ve net bir format sun.\n"
    "4. 'REFERANS DOKÜMAN' alanında bilgi varsa gelen metindeki verileri analiz edip detaylıca yanıtla.\n\n"
    "REFERANS DOKÜMAN:\n{context}"
)

# ---------------------------------------------------------
# 7. SOHBET ARAYÜZÜ VE İŞLEME
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = load_messages_from_db()

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message.get("docx_bytes"):
            st.download_button(
                label="📥 Word Formatında İndir (.docx)",
                data=message["docx_bytes"],
                file_name=message.get("file_name", "SMS_Raporu.docx"),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=f"dl_{idx}"
            )

if user_input := st.chat_input("Mesajınızı yazın..."):
    if not selected_provider:
        st.error("⚠️ Lütfen geçerli bir API Key tanımlayın.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        save_message_to_db("user", user_input)
        
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Yazıyor..."):
                keywords = [
                    "sms", "prosedür", "denetim", "sire", "solas", "marpol", "kural", "form", 
                    "checklist", "tanker", "gemi", "güverte", "makine", "isps", "ism", "rapor", 
                    "manta", "pdf", "docx", "doküman", "belge", "dosya", "incele", "maddeler",
                    "hazırla", "oluştur", "özet", "analiz", "listele", "raporla"
                ]
                needs_rag = any(kw in user_input.lower() for kw in keywords)
                
                relevant_docs = []
                if retriever and needs_rag:
                    relevant_docs = retriever.invoke(user_input)
                
                context_text = "\n\n".join([d.page_content for d in relevant_docs]) if relevant_docs else "Yok"

                # Sohbet geçmişini metin formatında birleştir
                raw_history = load_messages_from_db()[:-1]
                history_text = ""
                for m in raw_history[-6:]:
                    role_str = "Kullanıcı" if m["role"] == "user" else "Asistan"
                    history_text += f"{role_str}: {m['content']}\n"

                full_prompt = (
                    f"{system_prompt.format(context=context_text)}\n\n"
                    f"ÖNCEKİ KONUŞMALAR:\n{history_text}\n"
                    f"Kullanıcı: {user_input}\n"
                    f"Asistan:"
                )

                response_text = None
                last_error = ""

                try:
                    if "Google Gemini (Direct API)" in selected_provider:
                        response_text = call_direct_gemini(full_prompt, gemini_api_key)
                    else:
                        llm = ChatOpenAI(
                            model="google/gemini-2.0-flash-lite-001",
                            openai_api_key=openrouter_api_key,
                            openai_api_base="https://openrouter.ai/api/v1",
                            temperature=0.4
                        )
                        res = llm.invoke(full_prompt)
                        response_text = res.content

                except Exception as err:
                    last_error = str(err)

                if response_text:
                    sources = []
                    seen = set()
                    for doc in relevant_docs:
                        file_name = doc.metadata.get("source_file", doc.metadata.get("source", "Doküman"))
                        page_info = doc.metadata.get("page_label", "")
                        source_str = f"📄 {file_name} ({page_info})" if page_info else f"📄 {file_name}"
                        if source_str not in seen:
                            seen.add(source_str)
                            sources.append(source_str)

                    final_response = response_text
                    if sources:
                        final_response += "\n\n---\n**İncelediğim Şirket Dokümanları:**\n" + "\n".join([f"- {src}" for src in sources])

                    st.markdown(final_response)

                    form_keywords = ["form", "rapor", "hazırla", "oluştur", "docx", "word", "maddeler", "checklist", "incele"]
                    docx_bytes = None
                    file_name = None

                    if any(kw in user_input.lower() for kw in form_keywords):
                        file_name = f"SMS_Rapor_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.docx"
                        docx_bytes = create_docx_bytes(final_response, title="SMS & Denizcilik Asistan Raporu")
                        
                        st.download_button(
                            label="📥 Word Formatında İndir (.docx)",
                            data=docx_bytes,
                            file_name=file_name,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key=f"dl_new_{len(st.session_state.messages)}"
                        )

                    save_message_to_db("assistant", final_response, docx_bytes=docx_bytes, file_name=file_name)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": final_response,
                        "docx_bytes": docx_bytes,
                        "file_name": file_name
                    })
                else:
                    st.error(f"❌ API Hatası: {last_error if last_error else 'Servis yanıt vermedi.'}")
