import os
import shutil
import subprocess
import io
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
st.set_page_config(page_title="Denizcilik SMS Asistanı", page_icon="🚢", layout="wide")
st.title("🚢 Denizcilik SMS & Detaylı Risk/Form Asistanı")

DOCS_DIR = "docs"
INDEX_DIR = "faiss_index"

# ---------------------------------------------------------
# 2. SİDEBAR VE API KEY KONTROLÜ
# ---------------------------------------------------------
st.sidebar.header("⚙️ Ayarlar")
openrouter_api_key = st.secrets.get("OPENROUTER_API_KEY") if "OPENROUTER_API_KEY" in st.secrets else None

if not openrouter_api_key:
    openrouter_api_key = st.sidebar.text_input("OpenRouter API Key", type="password")

# ---------------------------------------------------------
# 3. EMBEDDINGS VE DOKÜMAN YÜKLEME
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

# ---------------------------------------------------------
# 4. FAISS İNDEKS KONTROLÜ
# ---------------------------------------------------------
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

if st.sidebar.button("🔄 İndeksi Yenile"):
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    st.cache_resource.clear()
    st.rerun()

# ---------------------------------------------------------
# 5. PROFESYONEL VE TAM DETAYLI WORD FORMU GELİŞTİRİCİ
# ---------------------------------------------------------
def create_advanced_sms_docx(work_title, content_text):
    """SIRE 2.0 / SMS Standartlarında Tablolu Word Dokümanı Hazırlar."""
    doc = DocxDocument()
    
    # Sayfa Kenar Boşlukları (Daraltılmış - Daha fazla sığsın)
    for section in doc.sections:
        section.top_margin = Inches(0.6)
        section.bottom_margin = Inches(0.6)
        section.left_margin = Inches(0.6)
        section.right_margin = Inches(0.6)

    # Başlık ve Üst Bilgi
    title_p = doc.add_paragraph()
    title_run = title_p.add_run("SAFETY MANAGEMENT SYSTEM (SMS)\nRISK ASSESSMENT & PERMIT TO WORK")
    title_run.bold = True
    title_run.font.size = Pt(16)
    title_run.font.color.rgb = RGBColor(0, 51, 102) # Koyu Deniz Mavi
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Gemi / Operasyon Üst Bilgi Tablosu
    header_table = doc.add_table(rows=2, cols=4)
    header_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    header_table.style = 'Table Grid'
    
    fields = [
        ("Vessel Name:", "M/T "); ("IMO No:", ""); 
        ("Date & Time:", ""); ("Location / Tank:", "");
        ("Work Description:", work_title[:40]); ("Permit No:", "RA-2026-");
        ("Risk Level:", "HIGH / MEDIUM / LOW"); ("Status:", "APPROVED")
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

    # Yapay Zekanın Ürettiği İçeriği İşle ve İncele
    lines = content_text.split('\n')
    in_table = False
    table_data = []

    for line in lines:
        l = line.strip()
        if not l:
            continue

        # Başlık Kontrolleri
        if l.startswith('# '):
            doc.add_heading(l.replace('# ', ''), level=1)
        elif l.startswith('## '):
            doc.add_heading(l.replace('## ', ''), level=2)
        elif l.startswith('### '):
            doc.add_heading(l.replace('### ', ''), level=3)
        # Tablo Satırı tespiti (| ile başlayan satırlar)
        elif l.startswith('|'):
            cells = [c.strip() for c in l.split('|')[1:-1]]
            # Ayraç satırlarını (---|---) atla
            if cells and not all(set(c).issubset({'-', ':', ' '}) for c in cells):
                table_data.append(cells)
        else:
            # Eğer tablodan çıkıldıysa mevcut birikmiş tabloyu oluştur
            if table_data:
                _build_docx_table(doc, table_data)
                table_data = []
            
            if l.startswith('* ') or l.startswith('- '):
                doc.add_paragraph(l[2:], style='List Bullet')
            else:
                doc.add_paragraph(l)

    # Kalan tablo varsa bas
    if table_data:
        _build_docx_table(doc, table_data)

    # Onay Kutuları / Checklist Alanı
    doc.add_heading("Safety Checklist & Controls", level=2)
    p_check = doc.add_paragraph()
    p_check.add_run("[  ] Risk Assessment briefed to all team members (Toolbox Talk Completed)\n").font.size = Pt(9.5)
    p_check.add_run("[  ] Required PPE available and inspected\n").font.size = Pt(9.5)
    p_check.add_run("[  ] Isolation / LOTO applied (If applicable)\n").font.size = Pt(9.5)
    p_check.add_run("[  ] Gas test performed and readings logged (If applicable)\n").font.size = Pt(9.5)
    p_check.add_run("[  ] Communication established with Duty Officer / Bridge").font.size = Pt(9.5)

    # İmza Blokları
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

def _build_docx_table(doc, table_data):
    """Markdown tablo verisini şık bir Word Tablosuna çevirir."""
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
                cell.paragraphs[0].text = val
                p = cell.paragraphs[0]
                p.style.font.size = Pt(8.5)
                # Başlık satırı stili
                if r_idx == 0:
                    p.runs[0].font.bold = True
                    # Gri arka plan shading yapılabilir
                    shading = OxmlElement('w:shd')
                    shading.set(qn('w:val'), 'clear')
                    shading.set(qn('w:color'), 'auto')
                    shading.set(qn('w:fill'), 'D9D9D9')
                    cell._tc.get_or_add_tcPr().append(shading)
    doc.add_paragraph("\n")

# ---------------------------------------------------------
# 6. MÜKEMMELLEŞTİRİLMİŞ RAG PROMPT VE SİSTEM TALİMATI
# ---------------------------------------------------------
retriever = None
llm = None

if openrouter_api_key and vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.1, # Daha kesin, teknik ve tutarlı yanıtlar için düşürüldü
        default_headers={"HTTP-Referer": "https://streamlit.io", "X-Title": "Maritime Agent RAG"}
    )

    # YÜZEYSEL YANITLARI ENGELLEYEN KATI SMS SYSTEM PROMPT'U
    system_prompt = (
        "Sen Kıdemli bir DPA (Designated Person Ashore), Enspektör ve Deniz Temizliği/Emniyeti Uzmanısın.\n"
        "GÖREVİN: Kullanıcının talebine göre SMS (Safety Management System) standartlarına tam uyumlu, "
        "SIRE 2.0 ve PSC denetimlerinden geçecek kadar DETAYLI, UYGULANABİLİR ve TEKNİK bir Risk Değerlendirmesi / İzin Protokolü üretmek.\n\n"
        "ŞU YÜZEYSEL VEYA GENEL YANITLARI KESİNLİKLE VERME! Her adımı adeta gemide o an yapılıyormuş gibi somutlaştır.\n\n"
        "EĞER BİR RİSK DEĞERLENDİRMESİ / FORM İSTENİYORSA AŞAĞIDAKİ YAPIYI TABLO (MARKDOWN TABLE) FORMATINDA ZORUNLU OLARAK KULLAN:\n\n"
        "### 1. Operasyon Adımları ve Detaylı Risk Matrisi\n"
        "Aşağıdaki kolonları içeren bir Markdown Tablosu oluştur:\n"
        "| No | Operasyon Adımı / Tehlike (Hazard) | Olası Sonuç (Consequence) | İlk Risk (Initial Risk) | Kontrol Önlemleri & Prosedürler (Control Measures) | Nihai Risk (Residual Risk) | Sorumlu Zabit |\n"
        "|---|---|---|---|---|---|---|\n"
        "(Her operasyon için en az 4-6 farklı teknik risk adımı ekle: Gaz, Elektrik, Yüksekten Düşme, İletişim Kesintisi, PPE, PPE Yetersizliği vb.)\n\n"
        "### 2. Zorunlu Kişisel Koruyucu Donanımlar (PPE) ve Özel Ekipmanlar\n"
        "(Spesifik olarak hangi PPE ve ölçüm cihazlarının [ör. EEBD, Çoklu Gaz Dedektörü, Calibrated Oxygen Meter, Full Body Harness] gerektiğini yaz)\n\n"
        "### 3. Acil Durum & Kaçış Prosedürleri (Emergency Protocols)\n"
        "(Kaza, gaz sızıntısı veya yaralanma anında atılacak spesifik adımlar)\n\n"
        "Bağlam Dokümanları:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

# ---------------------------------------------------------
# 7. SOHBET VE EKRAN ARAYÜZÜ
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "docx_bytes" in message:
            st.download_button(
                label="📥 Denetim Uyumlu Formu Word (.docx) Olarak İndir",
                data=message["docx_bytes"],
                file_name=message.get("file_name", "SMS_Risk_Assessment.docx"),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

if user_input := st.chat_input("Örn: 'Kazan dairesinde sıcak çalışma (Hot Work) için detaylı risk değerlendirmesi yap'"):
    if not openrouter_api_key:
        st.error("⚠️ Lütfen sol menüden OpenRouter API anahtarınızı girin.")
    elif not vectorstore:
        st.error("⚠️ Doküman bulunamadı. Lütfen 'docs' klasörünü kontrol edip indeksi yenileyin.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("SMS prosedürleri taranıyor, denetim matrisi ve teknik detaylar hazırlanıyor..."):
                try:
                    relevant_docs = retriever.invoke(user_input)
                    context_text = "\n\n".join([d.page_content for d in relevant_docs])
                    
                    formatted_prompt = prompt.format(context=context_text, question=user_input)
                    response_text = llm.invoke(formatted_prompt).content
                    
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

                    msg_data = {"role": "assistant", "content": final_response}
                    
                    # Eğer kullanıcı form/risk talebinde bulunduysa
                    keywords = ["form", "risk", "değerlendirme", "permit", "izin", "assessment", "çalışma", "temizlik"]
                    if any(kw in user_input.lower() for kw in keywords):
                        docx_file = create_advanced_sms_docx(user_input, response_text)
                        docx_bytes = docx_file.getvalue()
                        file_name = f"SMS_RA_{user_input[:15].replace(' ', '_')}.docx"
                        
                        st.download_button(
                            label="📥 Denetim Uyumlu Formu Word (.docx) Olarak İndir",
                            data=docx_bytes,
                            file_name=file_name,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                        msg_data["docx_bytes"] = docx_bytes
                        msg_data["file_name"] = file_name

                    st.session_state.messages.append(msg_data)

                except Exception as e:
                    st.error(f"Yanıt oluşturulurken bir hata oluştu: {str(e)}")
