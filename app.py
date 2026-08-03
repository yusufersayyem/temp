import os
import random
from dotenv import load_dotenv
import chainlit as cl

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

# 📢 قاعدة بيانات الإعلانات (يمكنك وضع روابط صورك المحلية أو رابط مباشر عبر الإنترنت)
ADS_DATA = [
    {
        "title": "مطعم فاخر - عروض خاصة",
        "url": "https://voyager.mynu.app/restaurant/675af6c4fc92f8671caef3cc",
        "image_path": "ads/ad1.webp"  # أو رابط مباشر "https://example.com/ad1.jpg"
    },
    {
        "title": "شركة نجمة الموصل",
        "url": "https://www.facebook.com/najmatalmosulco/",
        "image_path": "ads/ad2.webp"
    },
    {
        "title": "جامعة النور الأهلية",
        "url": "https://alnoor.edu.iq/ar/",
        "image_path": "ads/ad10.webp"
    }
]

def load_rag_chain():
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

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="مرحباً بك! 🤖 أنا المساعد الذكي لمديرية تربية نينوى وجامعة الموصل. كيف يمكنني مساعدتك اليوم؟"
    ).send()
    
    rag_chain = load_rag_chain()
    cl.user_session.set("rag_chain", rag_chain)
    
    # 🔢 عداد الإجابات المخصص لكل مستخدم
    cl.user_session.set("response_count", 0)

async def send_ad_card():
    """دالة لاختيار إعلان عشوائي وعرضه ككارت إعلاني أنيق"""
    ad = random.choice(ADS_DATA)
    
    # تجهيز العناصر المرفقة (الصورة)
    elements = []
    
    # إذا كانت الصورة ملفاً محلياً وموجوداً، أو رابئاً من الإنترنت
    if os.path.exists(ad["image_path"]):
        elements.append(
            cl.Image(name="ad_image", display="inline", path=ad["image_path"])
        )
    elif ad["image_path"].startswith("http"):
        elements.append(
            cl.Image(name="ad_image", display="inline", url=ad["image_path"])
        )
    
    ad_content = (
        f"📢 **إعلان رعاية المنصة:**\n\n"
        f"### [{ad['title']}]({ad['url']})\n"
        f"👉 [اضغط هنا للزيارة والتفاصيل]({ad['url']})"
    )
    
    await cl.Message(
        content=ad_content,
        elements=elements if elements else None
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")
    
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        async for chunk in rag_chain.astream({"input": message.content}):
            if "answer" in chunk:
                await msg.stream_token(chunk["answer"])
                
        await msg.update()
        
        # 🟢 زيادة العداد بعد الإجابة الناجحة
        count = cl.user_session.get("response_count") + 1
        cl.user_session.set("response_count", count)
        
        # 🎯 التحقق مما إذا اكتملت 5 إجابات
        if count % 2 == 0:
            await send_ad_card()
        
    except Exception as e:
        msg.content = f"حدث خطأ أثناء معالجة الطلب: {str(e)}"
        await msg.update()
