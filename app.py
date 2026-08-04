import os
import random
from dotenv import load_dotenv
import chainlit as cl

# 🟢 استيراد Qdrant و LangChain
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

# 🟢 إعدادات Qdrant Cloud
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "my_pdf_documents"  # اسم مجموعة البيانات على Qdrant Cloud

# ==================== 📢 قاعدة بيانات الإعلانات ====================
# تم فصل الإعلانات المصورة عن الإعلانات النصية الرفيعة
FEATURED_ADS = [
    # 📱 إعلان زين كاش المصور (Featured Ad)
    {
        "brand": "📲 زين كاش - Zain Cash",
        "offer": "فعّل محفظتك الآن واحصل على باقة بيانات مجانية حصرياً لطلاب نينوى!",
        "link": "https://www.zaincash.iq",
        # ضع رابط الصورة الحقيقي هنا (بانر أفقي)
        "image_url": "https://i.ibb.co/Xz95m4m/Zain-Cash-Banner.png", # رابط افتراضي كنموذج
        "badge": "الراعي الرسمي"
    }
]

SLIM_ADS_DATABASE = [
    # إعلانات نصية رفيعة للأجوبة المتكررة
    {
        "brand": "📲 زين كاش",
        "offer": "حوّل أموالك واستلمها بسرعة وسهولة عبر Zain Cash.",
        "link": "https://www.zaincash.iq",
        "badge": "الشريك الاستراتيجي"
    },
    {
        "brand": "🍕 مطعم الأصيل",
        "offer": "خصم 15% للطلاب والكوادر التعليمية",
        "link": "https://instagram.com/example_restaurant",
        "badge": "إعلان رعاية"
    },
    {
        "brand": "💻 شركة التقنية",
        "offer": "تصميم مواقع وتطبيقات للشركات والجامعات",
        "link": "https://wa.me/964000000000",
        "badge": "راعي المساعد"
    },
    {
        "brand": "☕ كافيه النخبة",
        "offer": "جلسات هادئة للدراسة بالقرب من المجمّع الثاني",
        "link": "https://maps.google.com",
        "badge": "توصية"
    }
]

# دالة تنسيق الشريط الرفيع (الافتراضي)
def get_slim_banner_markdown(ad: dict) -> str:
    return f"\n\n---\n` 📢 {ad['badge']} ` **[{ad['brand']}]({ad['link']})** — {ad['offer']} 🔗 *[تفاصيل أكثر]({ad['link']})*"

# دالة تنسيق كارت الصورة (للبداية فقط)
def get_image_ad_markdown(ad: dict) -> str:
    """
    تقوم بتنسيق الإعلان على شكل كارت صورة تحت إعلان نصي (الداكن الرفيع)
    """
    text_banner = f"` 📢 {ad['badge']} ` **[{ad['brand']}]({ad['link']})** — {ad['offer']}"
    image_display = f"\n\n[![ZainCash Ad]({ad['image_url']})]({ad['link']})" # الصورة قابلة للنقر أيضاً
    return f"\n\n---\n{text_banner}\n{image_display}\n---"


# ==================== 🧠 إعداد RAG Chain ====================

def load_rag_chain():
    """تحميل سلسلة الـ RAG مع Qdrant و Mistral"""
    embedding_model = MistralAIEmbeddings(model="mistral-embed")
    
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    
    vectorstore = QdrantVectorStore(
        client=client,
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
    return create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={'k': 3}), 
        combine_docs_chain
    )

# ==================== 💬 أحداث الشات ====================

@cl.on_chat_start
async def on_chat_start():
    # 🟢 1. عند البدء: عرض إعلان زين كاش المصور (كارت بانر) كترحيب خاص
    featured_zain_ad = FEATURED_ADS[0]
    
    # تنسيق الكارت المصور
    welcome_ad_card = get_image_ad_markdown(featured_zain_ad)
    
    welcome_message = (
        "أهلاً بك! 👋\n"
        "المساعد الآلي للمديرية العامة لتربية نينوى وجامعة الموصل.. قم بطرح أي سؤال أو استفسار لديك."
        "\n\n---"
        f"\n**برعاية:**\n{welcome_ad_card}"
    )
    
    await cl.Message(content=welcome_message).send()
    
    # 🟢 تحميل الـ RAG Chain وتخزينها في جلسة المستخدم
    rag_chain = load_rag_chain()
    cl.user_session.set("rag_chain", rag_chain)

@cl.on_message
async def on_message(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")
    
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # 🟢 بث الإجابة تدريجياً
        async for chunk in rag_chain.astream({"input": message.content}):
            if "answer" in chunk:
                await msg.stream_token(chunk["answer"])
        
        # 🟢 2. في الأجوبة العادية: اختيار إعلان نصي رفيع عشوائي (غير مصور)
        random_slim_ad = random.choice(SLIM_ADS_DATABASE)
        banner_text = get_slim_banner_markdown(random_slim_ad)
        
        await msg.stream_token(banner_text)
        await msg.update()
        
    except Exception as e:
        msg.content = f"⚠️ حدث خطأ أثناء معالجة الطلب: {str(e)}"
        await msg.update()
