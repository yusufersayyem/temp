import os
import random
from dotenv import load_dotenv
import chainlit as cl

# 🟢 استيراد Qdrant
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
    embedding_model = MistralAIEmbeddings(model="mistral-embed")
    
    # 🟢 الاتصال بـ Qdrant Cloud
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    
    # 🟢 تحميل الـ VectorStore من Qdrant
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

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="مرحباً بك! 🤖 أنا المساعد الذكي لمديرية تربية نينوى وجامعة الموصل. كيف يمكنني مساعدتك اليوم؟"
    ).send()
    
    rag_chain = load_rag_chain()
    cl.user_session.set("rag_chain", rag_chain)
    cl.user_session.set("response_count", 0)

async def send_ad_card():
    """عرض الإعلانات على شكل معرض متحرك/قابل للتمرير أفقيًا (Carousel)"""
    cards_html = ""
    for ad in ADS_DATA:
        cards_html += f"""
        <div style="
            flex: 0 0 200px;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 8px;
            background: #ffffff;
            box-shadow: 0 2px 5px rgba(0,0,0,0.08);
            text-align: center;
            box-sizing: border-box;
        ">
            <a href="{ad['url']}" target="_blank" style="text-decoration: none; color: inherit;">
                <img src="{ad['image_path']}" style="
                    width: 100%;
                    height: 100px;
                    object-fit: cover;
                    border-radius: 6px;
                    margin-bottom: 6px;
                " />
                <div style="
                    font-size: 13px;
                    font-weight: bold;
                    color: #0066cc;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                ">
                    {ad['title']}
                </div>
                <div style="
                    font-size: 11px;
                    color: #666;
                    margin-top: 3px;
                ">
                    🔗 اضغط للتفاصيل
                </div>
            </a>
        </div>
        """

    carousel_html = f"""
    <div style="margin-top: 10px; font-family: sans-serif;">
        <p style="font-weight: bold; font-size: 14px; margin-bottom: 8px; color: #333;">
            📢 <b>عروض ورعاية المنصة:</b>
        </p>
        <div style="
            display: flex;
            gap: 10px;
            overflow-x: auto;
            padding: 5px 2px 10px 2px;
            scroll-behavior: smooth;
            -webkit-overflow-scrolling: touch;
        ">
            {cards_html}
        </div>
    </div>
    """

    await cl.Message(content=carousel_html).send()

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
        
        count = cl.user_session.get("response_count") + 1
        cl.user_session.set("response_count", count)
        
        # عرض معرض الإعلانات كل إجابتين
        if count % 2 == 0:
            await send_ad_card()
        
    except Exception as e:
        msg.content = f"حدث خطأ أثناء معالجة الطلب: {str(e)}"
        await msg.update()
