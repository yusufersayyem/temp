import os
import base64
import random
from dotenv import load_dotenv
import chainlit as cl

# مكتبات LangChain و Qdrant الحديثة (Async)
from qdrant_client import AsyncQdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# تحميل متغيرات البيئة
load_dotenv()

# 🗄️ إعدادات Qdrant Cloud و Mistral
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
    """تحويل الصور إلى Base64 للعرض المباشر"""
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


def generate_ads_html_cards(ads_list):
    """
    توليد بطاقات إعلانية مصممة بنفس الطابع الزجاجي (Glassmorphic Cards)
    ومناسبة لشبكة Chainlit
    """
    cards_html = """
    <style>
        .ads-container {
            display: flex;
            gap: 12px;
            overflow-x: auto;
            padding: 10px 5px;
            direction: rtl;
        }
        .ad-card {
            min-width: 200px;
            max-width: 220px;
            height: 120px;
            border-radius: 12px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: transform 0.2s ease;
            background: #0f172a;
        }
        .ad-card:hover {
            transform: scale(1.03);
        }
        .ad-bg-blur {
            position: absolute;
            width: 100%; height: 100%;
            object-fit: cover;
            filter: blur(10px) brightness(0.7);
            transform: scale(1.2);
        }
        .ad-main-img {
            position: relative;
            z-index: 2;
            width: 100%; height: 100%;
            object-fit: contain;
        }
        .ad-title {
            position: absolute;
            bottom: 0; left: 0; right: 0;
            z-index: 3;
            background: rgba(0,0,0,0.6);
            color: white;
            font-size: 11px;
            padding: 4px 8px;
            text-align: center;
            font-family: 'Cairo', sans-serif;
        }
    </style>
    <div class="ads-container">
    """
    for ad in ads_list:
        img_src = get_base64_image(ad["image"])
        cards_html += f"""
        <a href="{ad['url']}" target="_blank" style="text-decoration: none;">
            <div class="ad-card">
                <img src="{img_src}" class="ad-bg-blur" />
                <img src="{img_src}" class="ad-main-img" alt="{ad['title']}" />
                <div class="ad-title">{ad['title']}</div>
            </div>
        </a>
        """
    cards_html += "</div>"
    return cards_html


async def build_rag_chain():
    """تجهيز الـ Async RAG Chain لسرعة الاستجابة تحت الضغط العالي"""
    embedding_model = MistralAIEmbeddings(model="mistral-embed")
    
    # استخدام AsyncQdrantClient لتفادي حظر الـ Threads مع آلاف المستخدمين
    async_client = AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    
    db = QdrantVectorStore(
        client=async_client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model
    )
    
    llm = ChatMistralAI(
        model="mistral-large-latest", 
        temperature=0.1, 
        max_retries=2,
        streaming=True
    )
    
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
    return create_retrieval_chain(db.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)


# =========================================================
# أحداث Chainlit Events
# =========================================================

@cl.on_chat_start
async def on_chat_start():
    """بداية الجلسة لكافة المستخدمين الجدد"""
    # تهيئة عداد الإجابات لكل جلسة مستخدم بحد ذاتها
    cl.user_session.set("bot_response_count", 0)
    
    # بناء الـ RAG Chain وتخزينه بالسرعة العالية بالذاكرة الخاصة بالمستخدم
    rag_chain = await build_rag_chain()
    cl.user_session.set("rag_chain", rag_chain)

    # 1. عرض الرسالة الترحيبية الهامة
    welcome_message = """
    ## 🤖 المجيب الآلي (تربية نينوى & جامعة الموصل)
    أهلاً بك! أنا المساعد الذكي المخصص للإجابة عن الاستفسارات والتعليمات الرسمية.
    
    🌟 **إعلانات ورعاة المنصة:**
    """
    
    # 2. إرسال الإعلانات الترحيبية في بداية الشات
    ads_html = generate_ads_html_cards(ADS_DATA[:5]) # عرض أول 5 إعلانات
    await cl.Message(
        content=welcome_message + ads_html,
        author="النظام"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """دالة معالجة الاستفسارات وإرجاع الإجابات بالبث اللحظي"""
    rag_chain = cl.user_session.get("rag_chain")
    bot_response_count = cl.user_session.get("bot_response_count", 0)
    
    # إنشاء رسالة فارغة لتأكيد بدء الرد (Streaming Response)
    msg = cl.Message(content="")
    await msg.send()

    try:
        # تنفيذ استرجاع البيانات والـ LLM بشكل Async
        async for chunk in rag_chain.astream({"input": message.content}):
            if "answer" in chunk:
                await msg.stream_token(chunk["answer"])

        await msg.update()

        # تحديث العداد الخاص بالإجابات
        bot_response_count += 1
        cl.user_session.set("bot_response_count", bot_response_count)

        # عرض شريط/بطاقات الإعلانات بعد كل 3 إجابات (كما في كودك السابق)
        if bot_response_count % 3 == 0:
            random_ads = random.sample(ADS_DATA, min(4, len(ADS_DATA)))
            ads_html = generate_ads_html_cards(random_ads)
            
            await cl.Message(
                content=f"📢 **عروض وإعلانات رعاية المنصة:**\n\n" + ads_html,
                author="الإعلانات"
            ).send()

    except Exception as e:
        await cl.Message(content=f"⚠️ حدث خطأ أثناء معالجة الطلب: {str(e)}").send()
