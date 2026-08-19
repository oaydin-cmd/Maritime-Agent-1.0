import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Denizcilik SMS Asistanı", page_icon="🚢")
st.title("🚢 Denizcilik SMS & Prosedür Asistanı (Bulut Tabanlı)")

# Google API Key Kontrolü
api_key = st.sidebar.text_input("Google Gemini API Key Giriniz", type="password")

uploaded_files = st.sidebar.file_uploader(
    "Şirket PDF Dokümanlarını Yükleyin", 
    type=["pdf"], 
    accept_multiple_files=True
)

if uploaded_files and api_key:
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

        # Bulut Üzerinde Çalışan Yapay Zeka Modeli
       llm = ChatOpenAI(
    model="google/gemini-flash-1.5",
    openai_api_key=openrouter_api_key,
    openai_api_base="https://openrouter.ai/api/v1"
)

        system_prompt = (
            "Sen şirket içi belgelere dayalı yanıt veren resmi bir denizcilik ve operasyon asistansın.\n"
            "Sadece sana sunulan aşağıdaki bağlamı (context) kullanarak soruya cevap ver.\n"
            "Sorunun cevabı dokümanda yoksa kibarca 'Bu bilgi şirket dokümanlarında bulunmamaktadır.' de.\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

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
                response = rag_chain.invoke({"input": user_input})
                st.markdown(response["answer"])

        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
elif not api_key:
    st.info("Lütfen sol menüden geçerli bir Google Gemini API anahtarı girin.")
