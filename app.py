import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Denizcilik SMS Asistanı", page_icon="🚢")
st.title("🚢 Denizcilik SMS & Prosedür Asistanı (Kalıcı Vektör Hafızası)")

# 1. API Anahtarını Streamlit Secrets veya Yan Menüden Alma
openrouter_api_key = st.secrets.get("OPENROUTER_API_KEY")

if not openrouter_api_key:
    openrouter_api_key = st.sidebar.text_input("OpenRouter API Key Giriniz", type="password")

# Sabit Klasör ve İndeks Yolları
DOCS_DIR = "docs"
INDEX_DIR = "faiss_index"

# 2. Embedding Modelini Başlatma (Cache Kullanarak Hızlandırma)
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

# 3. FAISS Vektör İndeksini Yükleme veya Sıfırdan Oluşturma Fonksiyonu
@st.cache_resource
def load_or_create_vectorstore():
    # Eğer daha önce işlenmiş indeks varsa doğrudan diskten yükle
    if os.path.exists(INDEX_DIR):
        st.info("Kalıcı FAISS indeksi diskten yükleniyor...")
        return FAISS.load_local(
            INDEX_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    
    # İndeks yoksa 'docs' klasöründeki PDF'leri işle
    if os.path.exists(DOCS_DIR) and os.listdir(DOCS_DIR):
        st.info("İlk kurulum: 'docs' klasöründeki PDF'ler işleniyor ve vektör veritabanı oluşturuluyor...")
        loader = PyPDFDirectoryLoader(DOCS_DIR)
        documents = loader.load()
        
        if not documents:
            st.warning("'docs' klasöründe okunabilir PDF bulunamadı.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(INDEX_DIR)
        st.success("Vektör veritabanı başarıyla oluşturuldu ve 'faiss_index' klasörüne kaydedildi!")
        return vectorstore
    else:
        st.warning(f"Lütfen projenizin kök dizinine '{DOCS_DIR}' adında bir klasör oluşturup PDF'lerinizi içine atın.")
        return None

# Session State Başlatma
if "messages" not in st.session_state:
    st.session_state.messages = []

# API Key Varsa Sistemi Kur
if openrouter_api_key:
    vectorstore = load_or_create_vectorstore()

    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # LLM Yapılandırması
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

        # Şirket Dokümanı + Genel Bilgi Hibrit Prompt'u
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

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # RAG Zinciri
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Geçmiş Mesajları Ekran Katmanında Göster
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Kullanıcı Etkileşimi
        if user_input := st.chat_input("SMS prosedürü veya genel denizcilik konuları hakkında soru sorun..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Yanıt hazırlanıyor..."):
                    response_text = rag_chain.invoke(user_input)
                    st.markdown(response_text)

            st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Yan Menüye Yeniden İndeksleme Butonu Ekleme
    if st.sidebar.button("Vektör İndeksini Yeniden Oluştur"):
        if os.path.exists(INDEX_DIR):
            import shutil
            shutil.rmtree(INDEX_DIR)
        st.cache_resource.clear()
        st.rerun()

else:
    st.info("Lütfen Streamlit Secrets alanına `OPENROUTER_API_KEY` ekleyin veya sol menüden API anahtarınızı girin.")
