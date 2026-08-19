import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Denizcilik SMS Asistanı", page_icon="🚢")
st.title("🚢 Denizcilik SMS & Prosedür Asistanı (Bulut Tabanlı)")

# API Key Kontrolü
openrouter_api_key = st.sidebar.text_input("OpenRouter API Key Giriniz", type="password")

uploaded_files = st.sidebar.file_uploader(
    "Şirket PDF Dokümanlarını Yükleyin", 
    type=["pdf"], 
    accept_multiple_files=True
)

if uploaded_files and openrouter_api_key:
    with st.spinner("Dokümanlar bulutta işleniyor..."):
        all_docs = []
        for uploaded_file in uploaded_files:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            all_docs.extend(docs)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(all_docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # OpenRouter Bağlantısı
        llm = ChatOpenAI(
            model="google/gemini-2.0-flash-exp:free",
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.2
        )

        system_prompt = (
            "Sen şirket içi belgelere dayalı yanıt veren resmi bir denizcilik ve operasyon asistansın.\n"
            "Sadece sana sunulan aşağıdaki bağlamı (context) kullanarak soruya cevap ver.\n"
            "Sorunun cevabı dokümanda yoksa kibarca 'Bu bilgi şirket dokümanlarında bulunmamaktadır.' de.\n\n"
            "Bağlam:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])

        # Dokümanları birleştirme fonksiyonu
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # LCEL RAG Zinciri Yapısı (v0.3+ Uyumlu)
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        st.success("SMS dokümanları başarıyla yüklendi!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("SMS prosedürü veya formlar hakkında soru sorun..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Yanıt hazırlanıyor..."):
                response_text = rag_chain.invoke(user_input)
                st.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})

elif not openrouter_api_key:
    st.info("Lütfen sol menüden geçerli bir OpenRouter API anahtarı girin.")
