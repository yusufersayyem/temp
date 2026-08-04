import os
import re
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

# ==================== 📢 نظام الإعلانات المخصصة ====================

# قاعدة بيانات صغيرة للإعلانات حسب الكلمات المفتاحية
ADS_DATABASE = {
    "مكتبة": {
        "title": "📚 مكتبة النجاح الجامعية - الموصل",
        "content": (
            "تتوفر لدينا كافة الملازم والكتب المنهجية لكليات جامعة الموصل "
            "مع خدمة التوصيل السريع داخل المحافظة.\n\n"
            "📞 [اضغط هنا للحجز والتواصل عبر واتساب](https://wa.me/9640000000000)"
        )
    },
    "معهد": {
        "title": "🌟 معهد المستقبل التعليمي",
        "content": (
            "دورات تقوية لطلاب الثانوية والجامعة في نينوى مع كادر تدريسي كفء.\n\n"
            "📌 خصم 15% عند استخدام كود: `NINEVEH2026`"
        )
    },
    "default": {
        "title": "📢 الراعي الرسمي للتطبيق",
        "content": (
            "هل تبحث عن خدمات طباعة وتنسيق البحوث والاطاريح؟\n"
            "تواصل مع **مركز الرواد للخدمات الطلابية** في الموصل قرب الجامعة."
        )
    }
}

def get_relevant_ad(text: str) -> cl.Text:
    """اختيار الإعلان المناسب بناءً على الكلمات المفتاحية في النص"""
    text_lower = text.lower()
    selected_ad = ADS_DATABASE["default"]

    if any(keyword in text_lower for keyword in ["جامعة", "كلي", "قسم", "كتاب", "ملزمة", "مكتبة"]):
        selected_ad = ADS_DATABASE["مكتبة"]
    elif any(keyword in text_lower for keyword in ["دورة", "امتحان", "معهد", "تربية", "مدرسة", "استاذ"]):
        selected_ad = ADS_DATABASE["معهد"]

    return cl.Text(
        name=selected_ad["title"],
        content=selected_ad["content"],
        display="side"  # 🟢 عرض الإعلان في الشريط الجانبي
    )

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
    # 🟢 1. إظهار إعلان افتراضي في الشريط الجانبي عند فتح الشات
    default_ad = get_relevant_ad("default")
    
    await cl.Message(
        content="تحية طيبة ... المساعد الآلي للمديرية العامة للتربية في محافظة نينوى وجامعة الموصل. تفضل بطرح أي سؤال لديك.",
        elements=[default_ad]
    ).send()
    
    # 🟢 2. تحميل الـ RAG Chain وتخزينها في جلسة المستخدم
    rag_chain = load_rag_chain()
    cl.user_session.set("rag_chain", rag_chain)

@cl.on_message
async def on_message(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")
    
    # 🟢 1. تحديث الشريط الجانبي بإعلان ذكي يناسب سؤال المستخدم الحالي
    dynamic_ad = get_relevant_ad(message.content)
    
    msg = cl.Message(content="", elements=[dynamic_ad])
    await msg.send()
    
    try:
        # 🟢 2. بث الإجابة تدريجياً (Streaming)
        async for chunk in rag_chain.astream({"input": message.content}):
            if "answer" in chunk:
                await msg.stream_token(chunk["answer"])
        
        # 🟢 3. إضافة Footer إعلاني خفيف في نهاية كل رد
        footer_ad = "\n\n---\n📢 **إعلان:** *احصل على خصومات الملازم والكتب لطلاب جامعة الموصل عبر التواصل مع مكتبة النجاح.*"
        await msg.stream_token(footer_ad)
        
        await msg.update()
        
    except Exception as e:
        msg.content = f"⚠️ حدث خطأ أثناء معالجة الطلب: {str(e)}"
        await msg.update()
