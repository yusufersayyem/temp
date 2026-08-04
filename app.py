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

# ==================== 📢 قاعدة بيانات الإعلانات (Ads Database) ====================
# يمكنك مستقبلاً جلب هذه البيانات من قاعدة بيانات أو ملف JSON
ADS_DATABASE = [
    {
        "id": 1,
        "title": "🍕 مطعم الأصيل - الموصل",
        "description": "خصم خاص 15% لطلاب جامعة الموصل وكوادر التربية عند إبراز الهوية!",
        "image_url": "https://ik.imagekit.io/63rncvror/ad1.webp?updatedAt=1785601369756random=1", # استبدلها برابط صورة الإعلان Real URL
        "link": "https://instagram.com",
        "cta": "اطلب الآن عبر انستغرام"
    },
    {
        "id": 2,
        "title": "💻 شركة التقنية للحلول البرمجية",
        "description": "تصميم وتطوير المواقع والتطبيقات بأعلى جودة وبأسعار منافسة.",
        "image_url": "https://picsum.photos/400/200?random=2",
        "link": "https://wa.me/964000000000",
        "cta": "تواصل معنا عبر واتساب"
    }
]

def format_ad_card(ad: dict) -> str:
    """دالة مساعدة لإنشاء كارت إعلاني بتنسيق Markdown ملفت"""
    return (
        f"\n---\n"
        f"### {ad['title']}\n"
        f"![Ad Banner]({ad['image_url']})\n"
        f"{ad['description']}\n\n"
        f"👉 **[{ad['cta']}]({ad['link']})**\n"
        f"---"
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
    # 🟢 1. اختيار إعلان مميز (مثلاً الإعلان الأول) لعرضه ككارت عند البداية
    featured_ad = ADS_DATABASE[0]
    welcome_ad_card = format_ad_card(featured_ad)
    
    welcome_message = (
        "أهلاً بك! 👋\n"
        "المساعد الآلي للمديرية العامة لتربية نينوى وجامعة الموصل .. قم بطرح أي سؤال او استفسار لديك.\n"
        f"\n**رعاة الخدمة:**\n{welcome_ad_card}"
    )
    
    await cl.Message(content=welcome_message).send()
    
    # 🟢 2. تحميل الـ RAG Chain وتخزينها في جلسة المستخدم
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
        
        # 🟢 3. إرفاق إعلان عشوائي كبطاقة صغيرة في نهاية كل إجابة
        random_ad = random.choice(ADS_DATABASE)
        ad_footer = f"\n\n> 📢 **إعلان رعائي:** [{random_ad['title']}]({random_ad['link']}) - {random_ad['description']}"
        
        await msg.stream_token(ad_footer)
        await msg.update()
        
    except Exception as e:
        msg.content = f"⚠️ حدث خطأ أثناء معالجة الطلب: {str(e)}"
        await msg.update()
