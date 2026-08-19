import os
import shutil
import subprocess
import io
import datetime
import sqlite3
import urllib.request
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
from docx import Document as DocxDocument
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

# ---------------------------------------------------------
# 1. SAYFA VE UYGULAMA YAPILANDIRMASI
# ---------------------------------------------------------
st.set_page_config(page_title="Denizcilik SMS & Hafızalı Asistan", page_icon="🚢", layout="wide")
st.title("🚢 Denizcilik SMS Asistanı (Kalıcı Hafıza Sistemli)")

DOCS_DIR = "docs"
INDEX_DIR = "faiss_index"
DB_PATH = "chat_history.db"

# ---------------------------------------------------------
# 2. İNTERNET ERİŞİM TESTİ
# ---------------------------------------------------------
def check_internet_connection():
    """OpenRouter ve genel internet erişimini test eder."""
    try:
        urllib.request.urlopen("https://openrouter.ai", timeout=5)
        return True, "Erişim Başarılı"
    except Exception as e:
        return False, str(e)

# ---------------------------------------------------------
# 3. SÜREKLİ VERİTABANI (SQLITE KALICI HAFIZA)
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
# 4. SİDEBAR, AYARLAR VE DURUM KONTROLÜ
# ---------------------------------------------------------
st.sidebar.header("⚙️ Ayarlar & Sistem Durumu")

# Ağ Bağlantısı Kontrolü
net_ok, net_msg = check_internet_connection()
if net_ok:
    st.sidebar.success("🌐 İnternet & OpenRouter Erişimi Aktif")
else:
    st.sidebar.error(f"⚠️ İnternet / Bağlantı Sorunu: {net_msg}")

openrouter_api_key = st.secrets.get("OPENROUTER_API_KEY") if "OPENROUTER_API_KEY" in st.secrets else None

if not openrouter_api_key:
    openrouter_api_key = st.sidebar.text_input("OpenRouter API Key", type="password")

if st.sidebar.button("🗑️ Tüm Sohbet Hafızasını Sıfırla"):
    clear_db_history()
    st.session_state.messages = []
    st.success("Tüm sohbet veritabanı temizlendi.")
    st.rerun()

# ---------------------------------------------------------
# 5. EMBEDDINGS VE DOKÜMAN YÜKLEME
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

            elif ext == ".doc":
                try:
                    result = subprocess.run(["antiword", file_path], capture_output=True, text=True, check=True)
                    if result.stdout.strip():
                        all_documents.append(Document(page_content=result.stdout, metadata={"source_file": file, "page_label": "Eski Word (.doc)"}))
                except Exception as doc_err:
                    st.error(f"❌ '.doc' hatası ({file}): {str(doc_err)}")

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(INDEX_DIR)
    return vectorstore

vectorstore = load_or_create_vectorstore()

if st.sidebar.button("🔄 SMS Doküman İndeksini Yenile"):
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    st.cache_resource.clear()
    st.rerun()

# ---------------------------------------------------------
# 6. WORD RAPOR / FORM ÜRETİCİ
# ---------------------------------------------------------
def _build_docx_table(doc, table_data):
    if not table_data:
        return
    rows_cnt = len(table_data)
    cols_cnt = max(len(r) for r in table_data)
    
    t = doc.add_table(rows=rows_cnt, cols=cols_cnt)
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER

    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            if c_idx < cols_cnt:
                cell = t.cell(r_idx, c_idx)
                cell.paragraphs[0].text = str(val)
                p = cell.paragraphs[0]
                p.style.font.size = Pt(8.5)
                if r_idx == 0:
                    if p.runs:
                        p.runs[0].font.bold = True
                    shd = OxmlElement('w:shd')
                    shd.set(qn('w:val'), 'clear')
                    shd.set(qn('w:color'), 'auto')
                    shd.set(qn('w:fill'), 'D9D9D9')
                    cell._tc.get_or_add_tcPr().append(shd)
    doc.add_paragraph("\n")

def create_advanced_sms_docx(work_title, content_text):
    doc = DocxDocument()
    for section in doc.sections:
        section.top_margin = Inches(0.6)
        section.bottom_margin = Inches(0.6)
        section.left_margin = Inches(0.6)
        section.right_margin = Inches(0.6)

    title_p = doc.add_paragraph()
    title_run = title_p.add_run("SAFETY MANAGEMENT SYSTEM (SMS)\nRISK ASSESSMENT & PERMIT TO WORK")
    title_run.bold = True
    title_run.font.size = Pt(16)
    title_run.font.color.rgb = RGBColor(0, 51, 102)
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    header_table = doc.add_table(rows=2, cols=4)
    header_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    header_table.style = 'Table Grid'
    
    fields = [
        ("Vessel Name:", "M/T "),
        ("IMO No:", ""), 
        ("Date & Time:", datetime.datetime.now().strftime("%d/%m/%Y %H:%M")),
        ("Location / Tank:", ""),
        ("Work Description:", str(work_title)[:40]),
        ("Permit No:", "RA-2026-"),
        ("Risk Level:", "HIGH / MEDIUM / LOW"),
        ("Status:", "APPROVED")
    ]
    
    for i, (label, val) in enumerate(fields):
        row_idx = i // 4
        col_idx = (i % 4)
        if row_idx < 2:
            cell = header_table.cell(row_idx, col_idx)
            cell.paragraphs[0].text = f"{label} {val}"
            cell.paragraphs[0].runs[0].font.bold = True
            cell.paragraphs[0].runs[0].font.size = Pt(9)

    doc.add_paragraph("\n")

    lines = content_text.split('\n')
    table_data = []

    for line in lines:
        l = line.strip()
        if not l:
            continue

        if l.startswith('# '):
            if table_data:
                _build_docx_table(doc, table_data)
                table_data = []
            doc.add_heading(l.replace('# ', ''), level=1)
        elif l.startswith('## '):
            if table_data:
                _build_docx_table(doc, table_data)
                table_data = []
            doc.add_heading(l.replace('## ', ''), level=2)
        elif l.startswith('### '):
            if table_data:
                _build_docx_table(doc, table_data)
                table_data = []
            doc.add_heading(l.replace('### ', ''), level=3)
        elif l.startswith('|'):
            cells = [c.strip() for c in l.split('|')[1:-1]]
            if cells and not all(set(c).issubset({'-', ':', ' '}) for c in cells):
                table_data.append(cells)
        else:
            if table_data:
                _build_docx_table(doc, table_data)
                table_data = []
            
            if l.startswith('* ') or l.startswith('- '):
                doc.add_paragraph(l[2:], style='List Bullet')
            else:
                doc.add_paragraph(l)

    if table_data:
        _build_docx_table(doc, table_data)

    doc.add_heading("Safety Checklist & Controls", level=2)
    p_check = doc.add_paragraph()
    p_check.add_run("[  ] Risk Assessment briefed to all team members (Toolbox Talk Completed)\n").font.size = Pt(9.5)
    p_check.add_run("[  ] Required PPE available and inspected\n").font.size = Pt(9.5)
    p_check.add_run("[  ] Isolation / LOTO applied (If applicable)\n").font.size = Pt(9.5)
    p_check.add_run("[  ] Gas test performed and readings logged (If applicable)\n").font.size = Pt(9.5)
    p_check.add_run("[  ] Communication established with Duty Officer / Bridge").font.size = Pt(9.5)

    doc.add_heading("Authorisation & Signatures", level=2)
    sig_table = doc.add_table(rows=2, cols=3)
    sig_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    sig_table.style = 'Table Grid'
    
    headers = ["Prepared By (Person in Charge)", "Checked By (Safety Officer)", "Approved By (Master / Ch.Eng)"]
    for idx, text in enumerate(headers):
        cell = sig_table.cell(0, idx)
        cell.paragraphs[0].text = text
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(9)
        
    for idx in range(3):
        cell = sig_table.cell(1, idx)
        cell.paragraphs[0].text = "\nName:\nRank:\nSignature:\nDate:"
        cell.paragraphs[0].runs[0].font.size = Pt(8.5)

    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

# ---------------------------------------------------------
# 7. RAG PROMPT VE LLM AYARI
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
        request_timeout=45,  # Ağ gecikmeleri için zaman aşımı süresi uzatıldı
        default_headers={"HTTP-Referer": "https://streamlit.io", "X-Title": "Maritime Agent RAG"}
    )

    system_prompt = (
        "Sen Kıdemli bir DPA, Enspektör ve Deniz Emniyeti Uzmanısın.\n"
        "GÖREVİN: Kullanıcının taleplerini yanıtlarken hem verilen SMS Dokümanlarını hem de "
        "GEÇMİŞ SOHBET HAFIZASINI dikkate alarak tam uyumlu, teknik yanıtlar veya Risk Değerlendirme formları üretmektir.\n\n"
        "ÖNEMLİ: Geçmiş sohbetlerde kullanıcının belirttiği kişisel bilgiler (adı, rütbesi, gemisi, önceki talepleri veya özel tercihleri) "
        "varsa bunları aklında tut ve yanıtlarında/formlarında kullan.\n\n"
        "GÜNCEL GEÇMİŞ SOHBET HAFIZASI:\n{chat_history}\n\n"
        "SMS BAĞLAM DOKÜMANLARI:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

# ---------------------------------------------------------
# 8. SOHBET GEÇMİŞİ VE ARAYÜZ
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = load_messages_from_db()

# Geçmiş mesajları render et
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        docx_data = message.get("docx_bytes")
        if docx_data and isinstance(docx_data, bytes) and len(docx_data) > 0:
            st.download_button(
                label="📥 Denetim Uyumlu Formu Word (.docx) Olarak İndir",
                data=docx_data,
                file_name=message.get("file_name", "SMS_Risk_Assessment.docx"),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=f"dl_btn_db_{idx}"
            )

# Yeni Mesaj Girdisi
if user_input := st.chat_input("Mesajınız... (Örn: 'Sintine tankı temizliği için risk değerlendirmesi yap')"):
    if not openrouter_api_key:
        st.error("⚠️ Lütfen sol menüden OpenRouter API anahtarınızı girin.")
    elif not net_ok:
        st.error("❌ Sunucuda internet/ağ bağlantısı yok. Lütfen ağ bağlantınızı veya güvenlik duvarı ayarlarınızı kontrol edin.")
    elif not vectorstore:
        st.error("⚠️ Doküman bulunamadı. Lütfen 'docs' klasörünü kontrol edin.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        save_message_to_db("user", user_input)
        
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Sohbet hafızası ve SMS prosedürleri inceleniyor..."):
                try:
                    relevant_docs = retriever.invoke(user_input)
                    context_text = "\n\n".join([d.page_content for d in relevant_docs]) if relevant_docs else "İlgili SMS dokümanı bulunamadı."
                    
                    recent_history = load_messages_from_db()[-15:]
                    history_str = "\n".join([f"{m['timestamp']} - {m['role'].upper()}: {m['content']}" for m in recent_history])

                    formatted_prompt = prompt.format(context=context_text, chat_history=history_str, question=user_input)
                    llm_response = llm.invoke(formatted_prompt)
                    response_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

                    if not response_text.strip():
                        response_text = "⚠️ Yapay zeka boş yanıt döndürdü. Lütfen sorgunuzu tekrar iletin."

                    sources = []
                    seen = set()
                    for doc in relevant_docs:
                        file_name = doc.metadata.get("source_file", doc.metadata.get("source", "Bilinmeyen Dosya"))
                        page_info = doc.metadata.get("page_label", "")
                        source_str = f"📄 **{file_name}** ({page_info})"
                        if source_str not in seen:
                            seen.add(source_str)
                            sources.append(source_str)

                    final_response = response_text
                    if sources:
                        final_response += "\n\n---\n**📚 Dokümandan Faydalanılan Referanslar:**\n" + "\n".join([f"- {src}" for src in sources])

                    st.markdown(final_response)

                    docx_bytes = None
                    file_name = None
                    keywords = ["form", "risk", "değerlendirme", "permit", "izin", "assessment", "çalışma", "temizlik"]
                    if any(kw in user_input.lower() for kw in keywords):
                        try:
                            docx_file = create_advanced_sms_docx(user_input, response_text)
                            docx_bytes = docx_file.getvalue()
                            file_name = f"SMS_RA_{user_input[:15].replace(' ', '_')}.docx"
                            
                            if docx_bytes and isinstance(docx_bytes, bytes):
                                st.download_button(
                                    label="📥 Denetim Uyumlu Formu Word (.docx) Olarak İndir",
                                    data=docx_bytes,
                                    file_name=file_name,
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    key=f"dl_btn_current_{len(st.session_state.messages)}"
                                )
                        except Exception as docx_err:
                            st.warning(f"⚠️ Word formu oluşturulurken bir uyarı alındı: {str(docx_err)}")

                    save_message_to_db("assistant", final_response, docx_bytes, file_name)
                    
                    msg_obj = {"role": "assistant", "content": final_response}
                    if docx_bytes and isinstance(docx_bytes, bytes):
                        msg_obj["docx_bytes"] = docx_bytes
                        msg_obj["file_name"] = file_name

                    st.session_state.messages.append(msg_obj)

                except Exception as e:
                    st.error(f"❌ Ağ veya API Yanıt Hatası: {str(e)}")
