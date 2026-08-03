import os
import base64
from dotenv import load_dotenv
import chainlit as cl
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# تحميل متغيرات البيئة
load_dotenv()

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

def get_rag_chain():
    embedding_model = MistralAIEmbeddings(model="mistral-embed")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model
    )
    
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

def generate_ads_html():
    """إنشاء HTML الكاروسيل لعرضه داخل Chainlit"""
    slides_html = ""
    for ad in ADS_DATA:
        img_src = get_base64_image(ad["image"])
        slides_html += f"""
        <div class="swiper-slide">
            <a href="{ad['url']}" target="_blank" class="ad-card-link">
                <div class="ad-card">
                    <img src="{img_src}" class="ad-bg-blur" alt="" />
                    <div class="glass-overlay"></div>
                    <img src="{img_src}" class="ad-main-img" alt="{ad['title']}" />
                </div>
            </a>
        </div>
        """

    return f"""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swiper@11/swiper-bundle.min.css" />
    <style>
        .swiper {{ width: 100%; padding: 10px 5px 30px 5px; }}
        .swiper-slide {{ width: 240px; }}
        .ad-card-link {{ text-decoration: none; display: block; }}
        .ad-card {{
            position: relative;
            width: 100%;
            aspect-ratio: 16 / 9;
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 6px 16px rgba(0,0,0,0.12);
            transition: transform 0.25s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #0f172a;
        }}
        .ad-card:hover {{ transform: translateY(-5px) scale(1.02); }}
        .ad-bg-blur {{
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            object-fit: cover; filter: blur(14px) brightness(0.85); transform: scale(1.2); z-index: 1;
        }}
        .glass-overlay {{
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 14px; z-index: 2;
        }}
        .ad-main-img {{
            position: relative; z-index: 3; max-width: 100%; max-height: 100%;
            object-fit: contain; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.25));
        }}
    </style>
    <div class="swiper mySwiper">
        <div class="swiper-wrapper">
            {slides_html}
        </div>
        <div class="swiper-pagination"></div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/swiper@11/swiper-bundle.min.js"></script>
    <script>
        new Swiper(".mySwiper", {{
            slidesPerView: "auto",
            spaceBetween: 15,
            grabCursor: true,
            autoplay: {{ delay: 2800, disableOnInteraction: false }},
            pagination: {{ el: ".swiper-pagination", clickable: true }},
        }});
    </script>
    """

@cl.on_chat_start
async def on_chat_start():
    """تجهيز السلسلة وحفظها في الجلسة عند بداية المحادثة"""
    # تهيئة RAG Chain
    rag_chain = get_rag_chain()
    cl.user_session.set("rag_chain", rag_chain)
    cl.user_session.set("bot_response_count", 0)

    # عرض إعلان ترحيبي ثنائي الابعاد في البداية
    ads_html = generate_ads_html()
    await cl.Message(
        content=f"🤖 **مرحباً بك في المجيب الآلي لتربية نينوى وجامعة الموصل**\n\n🌟 **إعلانات متميزة:**\n{ads_html}",
        author="المساعد"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """معالجة الرسائل القادمة من المستخدم"""
    rag_chain = cl.user_session.get("rag_chain")
    bot_response_count = cl.user_session.get("bot_response_count")

    # إظهار مؤشر الانتظار أثناء جلب الإجابة
    msg = cl.Message(content="")
    await msg.send()

    try:
        # استدعاء الـ RAG Chain asynchronously
        res = await rag_chain.ainvoke({'input': message.content})
        answer = res["answer"]
        
        # تحديث الرسالة بالنتيجة
        msg.content = answer
        await msg.update()

        # تحديث العداد
        bot_response_count += 1
        cl.user_session.set("bot_response_count", bot_response_count)

        # عرض الكاروسيل كل 3 إجابات
        if bot_response_count % 3 == 0:
            ads_html = generate_ads_html()
            await cl.Message(
                content=f"📢 **عروض وإعلانات رعاية المنصة:**\n{ads_html}",
                author="الإعلانات"
            ).send()

    except Exception as e:
        msg.content = f"حدث خطأ أثناء معالجة الطلب: {str(e)}"
        await msg.update()
