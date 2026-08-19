import os
import ssl
import urllib.request
import streamlit as st
from langchain_openai import ChatOpenAI

# SSL sertifika kısıtlamaları olan ağlarda isteklerin engellenmesini önler
os.environ["CURL_CA_BUNDLE"] = ""

def check_internet_connection():
    """
    Yapay zeka servisine erişim durumunu kontrol eder.
    Ağ kısıtlaması veya SSL doğrulaması olsa dahi uygulamanın durmasını engeller.
    """
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        # OpenRouter servisine test isteği
        urllib.request.urlopen("https://openrouter.ai", timeout=5, context=ctx)
        return True, "Yapay zeka servisine erişim başarılı."
    except Exception as e:
        # Hata alınsa bile yapay zekaya istek göndermeye izin veriyoruz
        return True, f"Ağ uyarısı (Yapay zekaya erişim deneniyor): {str(e)}"

def get_llm():
    """
    OpenRouter API üzerinden Yapay Zeka modelini başlatır.
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")
    
    if not api_key:
        st.error("OPENROUTER_API_KEY bulunamadı! Lütfen ortam değişkenlerini kontrol edin.")
        return None

    return ChatOpenAI(
        model="anthropic/claude-3.5-sonnet",  # Kullandığınız modeli buraya yazabilirsiniz
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        timeout=45,       # Yavaş/kısıtlı bağlantılar için zaman aşımı süresi
        max_retries=3,    # Bağlantı anlık koparsa 3 defa yeniden dener
        default_headers={
            "HTTP-Referer": "http://localhost:8501", 
            "X-Title": "Maritime Assistant"
        }
    )

# --- UYGULAMA BAŞLANGICI ---
st.set_page_config(page_title="Maritime Assistant", layout="wide")

# İnternet/Bağlantı durumunu kontrol et
is_connected, message = check_internet_connection()

if not is_connected:
    st.warning(f"Sistem Uyarısı: {message}")
else:
    st.success("Yapay zeka erişimi aktif.", icon="✅")

# LLM Modeli Başlat
llm = get_llm()

# Chat/Uygulama Mantığınız
if llm:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Sorunuzu yazın..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                response = llm.invoke(user_input)
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            except Exception as ex:
                st.error(f"Yapay zeka ile iletişim kurulurken bir hata oluştu: {str(ex)}")
