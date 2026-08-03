import os
import random
from dotenv import load_dotenv
import chainlit as cl

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# تحميل متغيرات البيئة من ملف .env
load_dotenv()

# مسار قاعدة بيانات الـ FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"

# 📢 قاعدة بيانات الإعلانات الكارتية المصممة أنيقاً
ADS_DATA = [
    {
        "title": "مطعم فاخر - عروض خاصة 🍔",
        "description": "استمتع بأشهى الوجبات مع خصومات مميزة خصيصاً لمستخدمي المنصة.",
        "url": "https://voyager.mynu.app/restaurant/675af6c4fc92f8671caef3cc",
        "image_url": "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=500&q=80",
        "btn_text": "طلب الآن 👈"
    },
    {
        "title": "شركة نجمة الموصل 🌟",
        "description": "حلول برمجية وخدمات تسويق رقمي متكاملة لكافة القطاعات والمؤسسات.",
        "url": "https://www.facebook.com/najmatalmosulco/",
        "image_url": "https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=500&q=80",
        "btn_text": "تواصل معنا 📞"
    },
    {
        "title": "جامعة النور الأهلية 🎓",
        "description": "سجل الآن في أحدث التخصصات الأكاديمية والطبية في محافظة نينوى.",
        "url": "https://alnoor.edu.iq/ar/",
        "image_url": "https://images.unsplash.com/photo-1523050854058-8df90110c9f1?w=500&q=80",
        "btn_text": "زيارة الموقع الرسمي 🌐"
    },
    {
        "title": "تطبيق بلي - توصيل سريع 🚖",
        "description": "احصل على أفضل خدمات التنقل والتوصيل السريع داخل المدينة بسهولة.",
        "url": "https://baly.iq/taxi/",
        "image_url": "https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?w=500&q=80",
        "btn_text": "حمل التطبيق الان 📲"
    }
]


def load_rag_chain():
    """تحميل نموذج الـ RAG وقاعدة بيانات FAISS"""
    embedding_model = MistralAIEmbeddings(model="mistral-embed")
    vectorstore = FAISS.load_local(
        DB_FAISS_PATH, 
        embedding_model, 
        allow_dangerous_deserialization=True
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


async def send_fancy_ad_card():
    """عرض كارت إعلاني أنيق بتنسيق HTML/CSS (الطريقة الموصى بها)"""
    ad = random.choice(ADS_DATA)
    
    card_html = f"""
    <div style="
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        overflow: hidden;
        background: #ffffff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        margin: 12px 0;
        max-width: 420px;
        direction: rtl;
        text-align: right;
        font-family: 'Cairo', system-ui, sans-serif;
    ">
        <div style="position: relative;">
            <img src="{ad['image_url']}" style="width: 100%; height: 160px; object-fit: cover; display: block;" />
            <span style="
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(15, 23, 42, 0.75);
                color: #ffffff;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: bold;
                backdrop-filter: blur(4px);
            ">📢 إعلان برعاية</span>
        </div>
        
        <div style="padding: 16px;">
            <h3 style="margin: 0 0 8px 0; color: #0f172a; font-size: 16px; font-weight: 700;">{ad['title']}</h3>
            <p style="margin: 0 0 14px 0; color: #475569; font-size: 13px; line-height: 1.5;">{ad['description']}</p>
            
            <a href="{ad['url']}" target="_blank" style="
                display: block;
                text-align: center;
                background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
                color: #ffffff !important;
                text-decoration: none !important;
                padding: 10px 16px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 13px;
                transition: all 0.2s ease;
            ">
                {ad['btn_text']}
            </a>
        </div>
    </div>
    """
    
    await cl.Message(content=card_html).send()


@cl.on_chat_start
async def on_chat_start():
    """عند فتح المحادثة أول مرة"""
    await cl.Message(
        content="مرحباً بك! 🤖 أنا المساعد الذكي لمديرية تربية نينوى وجامعة الموصل. كيف يمكنني مساعدتك اليوم؟"
    ).send()
    
    # تحميل السلسلة وحفظها في جلسة المستخدم
    rag_chain = load_rag_chain()
    cl.user_session.set("rag_chain", rag_chain)
    
    # تهيئة عداد الإجابات للمستخدم
    cl.user_session.set("response_count", 0)


@cl.on_message
async def on_message(message: cl.Message):
    """عند استقبال أي سؤال من المستخدم"""
    rag_chain = cl.user_session.get("rag_chain")
    
    # إنشاء رسالة فارغة لدعم الـ Streaming
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # إرسال الإجابة بشكل تدريجي (Streaming)
        async for chunk in rag_chain.astream({"input": message.content}):
            if "answer" in chunk:
                await msg.stream_token(chunk["answer"])
                
        await msg.update()
        
        # 🟢 زيادة العداد بعد الإجابة الناجحة
        count = cl.user_session.get("response_count") + 1
        cl.user_session.set("response_count", count)
        
        # 🎯 إظهار الكارت الإعلاني كل 5 إجابات
        if count % 5 == 0:
            await send_fancy_ad_card()
            
    except Exception as e:
        msg.content = f"حدث خطأ أثناء معالجة الطلب: {str(e)}"
        await msg.update()
