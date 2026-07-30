import streamlit as st
import os
import base64
from dotenv import load_dotenv
import streamlit.components.v1 as components
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# تحميل متغيرات البيئة
load_dotenv()

# الإعدادات العامة والأيقونات
SYSTEM_AVATAR = "https://cdn-icons-png.flaticon.com/512/4712/4712035.png"
USER_AVATAR = "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&auto=format&fit=crop&q=80"
AD_AVATAR = "https://cdn-icons-png.flaticon.com/512/2997/2997311.png"

# 🗄️ إعدادات Qdrant Cloud
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "my_pdf_documents"

# 📢 قاعدة بيانات الإعلانات
ADS_DATA = [
    {"image": "/content/ads/ad1.webp", "url": "https://voyager.mynu.app/restaurant/675af6c4fc92f8671caef3cc", "title": "مطعم فاخر - عروض خاصة"},
    {"image": "/content/ads/ad2.webp", "url": "https://www.facebook.com/najmatalmosulco/", "title": "شركة نجمة الموصل"},
    {"image": "/content/ads/ad3.webp", "url": "https://baly.iq/taxi/", "title": "تطبيق بلي - توصيل سريع"},
    {"image": "/content/ads/ad4.webp", "url": "https://www.iq.zain.com/ar", "title": "زين العراق - أحدث العروض"},
    {"image": "/content/ads/ad5.webp", "url": "https://www.facebook.com/profile.php?id=100063940127604", "title": "إعلان راعي المنصة"},
    {"image": "/content/ads/ad6.webp", "url": "https://www.facebook.com/larsafoundation/", "title": "مؤسسة لارسا"},
    {"image": "/content/ads/ad7.webp", "url": "https://www.facebook.com/barqmouslba/", "title": "برق الموصل"},
    {"image": "/content/ads/ad8.webp", "url": "https://www.facebook.com/p/%D9%85%D8%AC%D9%85%D8%B9-%D8%B3%D9%8A%D8%AF-%D8%A7%D9%84%D8%A7%D8%B3%D8%B9%D8%A7%D8%B1-3-%D9%81%D8%B1%D8%B9-%D8%A7%D9%84%D9%85%D8%AC%D9%85%D9%88%D8%B9%D8%A9-100066359418433/?locale=ku_TR", "title": "مجمع سيد الاسعار"},
    {"image": "/content/ads/ad9.webp", "url": "https://www.facebook.com/anaskashmola/", "title": "خدمات إعلانية متميزة"},
    {"image": "/content/ads/ad10.webp", "url": "https://alnoor.edu.iq/ar/", "title": "جامعة النور الأهلية"}
]

@st.cache_data
def get_base64_image(image_path):
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
            ext = os.path.splitext(image_path)[1].replace(".", "").lower()
            mime_types = {"gif": "image/gif", "webp": "image/webp", "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
            mime_type = mime_types.get(ext, f"image/{ext}")
            return f"data:{mime_type};base64,{encoded_string}"
        else:
            return "https://picsum.photos/500/200"
    except Exception:
        return "https://picsum.photos/500/200"

@st.cache_resource
def get_vectorstore():
    embedding_model = MistralAIEmbeddings(model="mistral-embed")
    
    # الاتصال بـ Qdrant Cloud باستخدام URL و API Key
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    
    # استخدام QdrantVectorStore للربط مع المجموعة السحابية
    db = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model
    )
    return db

@st.cache_resource
def get_rag_chain():
    vectorstore = get_vectorstore()
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0.1, max_retries=2)
    system_prompt = (
        "أنت مساعد ذكي مخصص للإجابة عن استفسارات مديرية تربية نينوى وجامعة الموصل.\n"
        "استخدم المعلومات الواردة في السياق التالي فقط للإجابة على سؤال المستخدم.\n"
        "إذا لم تكن الإجابة موجودة في السياق، قل بوضوح أنك لا تملك المعلومة الرسمية حول ذلك.\n"
        "أجب باللغة العربية وبشكل دقيق ومختصر.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vectorstore.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)

def render_ads_carousel():
    """عرض الإعلانات في كورسويل (Carousel) متحرّك - صور كاملة بدون نصوص أو أزرار"""
    slides_html = ""
    for ad in ADS_DATA:
        img_src = get_base64_image(ad["image"])
        slides_html += f"""
        <div class="swiper-slide">
            <a href="{ad['url']}" target="_blank" class="ad-card-link">
                <div class="ad-card">
                    <img src="{img_src}" alt="{ad['title']}" />
                </div>
            </a>
        </div>
        """

    carousel_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swiper@11/swiper-bundle.min.css" />
        <style>
            body {{ margin: 0; font-family: system-ui, -apple-system, sans-serif; background: transparent; }}
            .swiper {{ width: 100%; padding: 10px 5px 30px 5px; }}
            .swiper-slide {{ width: 240px; }}
            
            .ad-card-link {{
                text-decoration: none;
                display: block;
            }}
            
            .ad-card {{
                width: 100%;
                height: 140px;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                background: #f1f5f9;
            }}
            
            .ad-card:hover {{
                transform: translateY(-4px) scale(1.02);
                box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            }}
            
            .ad-card img {{
                width: 100%;
                height: 100%;
                object-fit: cover;
                display: block;
            }}
        </style>
    </head>
    <body>
        <div class="swiper mySwiper">
            <div class="swiper-wrapper">
                {slides_html}
            </div>
            <div class="swiper-pagination"></div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/swiper@11/swiper-bundle.min.js"></script>
        <script>
            var swiper = new Swiper(".mySwiper", {{
                slidesPerView: "auto",
                spaceBetween: 15,
                grabCursor: true,
                autoplay: {{
                    delay: 2800,
                    disableOnInteraction: false,
                }},
                pagination: {{
                    el: ".swiper-pagination",
                    clickable: true,
                }},
            }});
        </script>
    </body>
    </html>
    """
    components.html(carousel_html, height=195)

def process_rag_response(user_query):
    try:
        rag_chain = get_rag_chain()
        
        with st.chat_message("assistant", avatar=SYSTEM_AVATAR):
            with st.spinner("جاري البحث ..."):
                response = rag_chain.invoke({'input': user_query})
                result = response["answer"]
                st.markdown(result)
        
        st.session_state.messages.append({'role': 'assistant', 'type': 'text', 'content': result})
        st.session_state.bot_response_count += 1

        # عرض الكاروسيل كل 3 إجابات
        if st.session_state.bot_response_count % 3 == 0:
            with st.chat_message("assistant", avatar=AD_AVATAR):
                st.write("📢 **عروض وإعلانات رعاية المنصة:**")
                render_ads_carousel()
            
            st.session_state.messages.append({'role': 'assistant', 'type': 'carousel'})

    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة الطلب: {str(e)}")

def main():
    st.set_page_config(
        page_title="المجيب الآلي - تربية نينوى وجامعة الموصل",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'bot_response_count' not in st.session_state:
        st.session_state.bot_response_count = 0

    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
            html, body, [class*="css"] {
                font-family: 'Cairo', sans-serif;
                direction: ltr !important;
                text-align: left !important;
            }
            [data-testid="stSidebar"], [data-testid="stSidebarNav"], [data-testid="collapsedControl"] { display: none !important; }
            footer {visibility: hidden;}
            header [data-testid="stAppDeployButton"] {display: none;}
        </style>
    """, unsafe_allow_html=True)

    # الهيدر العلوي بأيقونة الروبوت 🤖
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; direction: ltr; margin-bottom: 15px;">
            <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: white; width: 45px; height: 45px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 24px;">🤖</div>
            <h1 style="margin: 0; color: #3b82f6; font-weight: 700; font-size: 22px;">المجيب الآلي لتربية نينوى وجامعة الموصل</h1>
        </div>
    """, unsafe_allow_html=True)

    # عرض الكاروسيل ثابتاً في أعلى الصفحة
    st.write("🌟 **إعلانات متميزة:**")
    render_ads_carousel()

    # عرض سجل المحادثة
    for message in st.session_state.messages:
        if message.get('type') == 'carousel':
            with st.chat_message("assistant", avatar=AD_AVATAR):
                st.write("📢 **عروض وإعلانات رعاية المنصة:**")
                render_ads_carousel()
        else:
            avatar = USER_AVATAR if message['role'] == 'user' else SYSTEM_AVATAR
            st.chat_message(message['role'], avatar=avatar).markdown(message['content'])

    prompt = st.chat_input("اكتب سؤالك هنا...")
    if prompt:
        st.chat_message('user', avatar=USER_AVATAR).markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'type': 'text', 'content': prompt})
        process_rag_response(prompt)

if __name__ == "__main__":
    main()
