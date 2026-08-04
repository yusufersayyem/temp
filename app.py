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

# ==================== 📢 قاعدة بيانات الشريط الإعلاني الرفيع ====================
# هيكلية مبسطة ومناسبة للعرض في شريط رفيع
ADS_DATABASE = [
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

def get_slim_banner_markdown(ad: dict) -> str:
    """
    دالة تقوم بتنسيق الإعلان على شكل شريط رفيع وأنيق (Slim Footer Banner)
    """
    return f"\n\n---\n` 📢 {ad['badge']} ` **[{ad['brand']}]({ad['link']})** — {ad['offer']} 🔗 *[تفاصيل أكثر]({ad['link']})*"

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
    # 🟢 1. إرسال رسالة الترحيب مع شريط إعلاني رفيع ومبسط في الأسفل
    welcome_ad = ADS_DATABASE[0]
    slim_banner = get_slim_banner_markdown(welcome_ad)
    
    welcome_message = (
        "أهلاً بك! 👋\n"
        "المساعد الآلي للمديرية العامة لتربية نينوى وجامعة الموصل .. قم بطرح أي سؤال أو استفسار لديك."
        f"{slim_banner}"
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
        
        # 🟢 3. اختيار إعلان عشوائي وإلحاقه كـ "شريط رفيع" في ذيل الرسالة (Footer Banner)
        random_ad = random.choice(ADS_DATABASE)
        banner_text = get_slim_banner_markdown(random_ad)
        
        await msg.stream_token(banner_text)
        await msg.update()
        
    except Exception as e:
        msg.content = f"⚠️ حدث خطأ أثناء معالجة الطلب: {str(e)}"
        await msg.update()
