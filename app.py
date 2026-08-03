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

# 📢 قاعدة بيانات الإعلانات
ADS_DATA = [
    {
        "title": "مطعم فاخر - عروض خاصة",
        "url": "https://voyager.mynu.app/restaurant/675af6c4fc92f8671caef3cc",
        "image_path": "https://ik.imagekit.io/63rncvror/ad1.webp?updatedAt=1785601369756"
    },
    {
        "title": "شركة نجمة الموصل",
        "url": "https://www.facebook.com/najmatalmosulco/",
        "image_path": "https://ik.imagekit.io/63rncvror/ad7.webp?updatedAt=1785601364077"
    },
    {
        "title": "جامعة النور الأهلية",
        "url": "https://alnoor.edu.iq/ar/",
        "image_path": "https://ik.imagekit.io/63rncvror/ad7.webp?updatedAt=1785601364077"
    }
]

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

# ==================== 📢 نظام الإعلانات المطور ====================

def get_ad_payload(ad_index: int):
    """توليد محتوى وعناصر الإعلان بناءً على الفهرس"""
    total_ads = len(ADS_DATA)
    current_index = ad_index % total_ads
    ad = ADS_DATA[current_index]

    content = (
        f"--- \n"
        f"✨ **رعاية إعلانية ({current_index + 1}/{total_ads})**\n\n"
        f"### 🔹 [{ad['title']}]({ad['url']})\n"
        f"👉 [اضغط هنا لزيارة التفاصيل والرابط الرسمي]({ad['url']})"
    )

    image_element = cl.Image(
        name=f"ad_img_{current_index}",
        url=ad["image_path"],
        display="inline"
    )

    actions = [
        cl.Action(
            name="change_ad", 
            value=str((current_index - 1) % total_ads), 
            label="◀ السابق"
        ),
        cl.Action(
            name="change_ad", 
            value=str((current_index + 1) % total_ads), 
            label="التالي ▶"
        )
    ]

    return content, [image_element], actions

@cl.action_callback("change_ad")
async def on_change_ad(action: cl.Action):
    """تحديث نفس رسالة الإعلان عند الضغط على أزرار التصفح دون إغراق الشات"""
    target_index = int(action.value)
    content, elements, actions = get_ad_payload(target_index)
    
    # الحصول على الرسالة الحالية وتحديث محتواها وأزرارها في مكانها
    message = action.for_id
    if message:
        msg = cl.Message(id=message, content=content, elements=elements, actions=actions)
        await msg.update()

# ==================== 💬 أحداث الشات ====================

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="مرحباً بك! 🤖 أنا المساعد الذكي لمديرية تربية نينوى وجامعة الموصل. كيف يمكنني مساعدتك اليوم؟"
    ).send()
    
    # تحميل الـ RAG Chain وتخزينها في جلسة المستخدم
    rag_chain = load_rag_chain()
    cl.user_session.set("rag_chain", rag_chain)
    cl.user_session.set("response_count", 0)

@cl.on_message
async def on_message(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")
    
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # بث الإجابة تدريجياً (Streaming)
        async for chunk in rag_chain.astream({"input": message.content}):
            if "answer" in chunk:
                await msg.stream_token(chunk["answer"])
                
        await msg.update()
        
        # زيادة عداد الإجابات
        count = cl.user_session.get("response_count") + 1
        cl.user_session.set("response_count", count)
        
        # عرض بطاقة إعلانية تفاعلية مستقيلّة كل إجابتين (مع اختيار عشوائي للبداية)
        if count % 2 == 0:
            random_start_index = random.randint(0, len(ADS_DATA) - 1)
            content, elements, actions = get_ad_payload(random_start_index)
            
            ad_msg = cl.Message(
                content=content,
                elements=elements,
                actions=actions
            )
            await ad_msg.send()
        
    except Exception as e:
        msg.content = f"⚠️ حدث خطأ أثناء معالجة الطلب: {str(e)}"
        await msg.update()
