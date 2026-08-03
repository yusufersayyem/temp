import os
import random
import base64
from dotenv import load_dotenv
import chainlit as cl

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

# 📢 قاعدة بيانات الإعلانات
ADS_DATA = [
    {
        "title": "مطعم فاخر - عروض خاصة",
        "url": "https://voyager.mynu.app/restaurant/675af6c4fc92f8671caef3cc",
        "image_path": "ads/ad1.webp"
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
    cl.user_session.set("response_count", 0)

async def send_ad_card():
    """عرض الإعلانات باستخدام عنصر cl.Image المدمج في Chainlit لتجنب مشكلة الـ Base64"""
    selected_ads = random.sample(ADS_DATA, min(2, len(ADS_DATA)))
    
    elements = []
    ad_texts = []
    
    for idx, ad in enumerate(selected_ads):
        image_path = ad["image_path"]
        name = f"ad_image_{idx}"
        
        # إذا كانت الصورة محلياً وموجودة، نستخدم cl.Image
        if os.path.exists(image_path):
            elements.append(
                cl.Image(name=name, path=image_path, display="inline")
            )
        else:
            # إذا كان رابطاً خارجيًا أو غير موجود
            fallback_url = image_path if image_path.startswith("http") else "https://picsum.photos/400/200"
            elements.append(
                cl.Image(name=name, url=fallback_url, display="inline")
            )

        # تجهيز نص الكارت المنسق بـ Markdown
        ad_texts.append(
            f"### 📢 {ad['title']}\n"
            f"[👉 اضغط هنا لزيارة التفاصيل]({ad['url']})"
        )
    
    content_body = "\n---\n".join(ad_texts)
    full_content = f"**📢 عروض وإعلانات رعاية المنصة:**\n\n{content_body}"

    await cl.Message(
        content=full_content,
        elements=elements
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
        
        # 🟢 زيادة العداد
        count = cl.user_session.get("response_count") + 1
        cl.user_session.set("response_count", count)
        
        # 🎯 إظهار الإعلانات التفاعلية عند تكرار الشرط
        if count % 2 == 0:
            await send_ad_card()
        
    except Exception as e:
        msg.content = f"حدث خطأ أثناء معالجة الطلب: {str(e)}"
        await msg.update()
