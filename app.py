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

def get_image_src(image_path):
    """تحويل الصورة المحلية إلى Base64 أو إرجاع رابط URL المباشر"""
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode()
            ext = os.path.splitext(image_path)[1].replace(".", "").lower()
            mime_types = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
            mime_type = mime_types.get(ext, "image/jpeg")
            return f"data:{mime_type};base64,{encoded_string}"
        except Exception:
            return "https://picsum.photos/400/200"
    elif image_path.startswith("http"):
        return image_path
    return "https://picsum.photos/400/200"

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
    """عرض الإعلانات في شبكة كارتات أفقية/شبكية باستعمال CSS Flexbox"""
    # اختيار إعلانين كحد أقصى للظهور بجانب بعضهما، أو اختيار كل الإعلانات حسب رغبتك
    selected_ads = random.sample(ADS_DATA, min(2, len(ADS_DATA)))
    
    cards_html = ""
    for ad in selected_ads:
        img_src = get_image_src(ad["image_path"])
        cards_html += f"""
        <div style="
            flex: 1 1 180px;
            max-width: 220px;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            overflow: hidden;
            background: #ffffff;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        ">
            <div style="position: relative; height: 110px; overflow: hidden; background: #f1f5f9;">
                <img src="{img_src}" style="width: 100%; height: 100%; object-fit: cover; display: block;" />
            </div>
            <div style="padding: 10px; display: flex; flex-direction: column; flex-grow: 1; justify-content: space-between;">
                <h4 style="margin: 0 0 10px 0; color: #0f172a; font-size: 12px; font-weight: 700; line-height: 1.4;">{ad['title']}</h4>
                <a href="{ad['url']}" target="_blank" style="
                    display: block;
                    text-align: center;
                    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
                    color: #ffffff !important;
                    text-decoration: none !important;
                    padding: 6px 10px;
                    border-radius: 6px;
                    font-size: 11px;
                    font-weight: 600;
                ">
                    زيارة التفاصيل 👈
                </a>
            </div>
        </div>
        """

    grid_container = f"""
    <div style="
        direction: rtl;
        text-align: right;
        font-family: 'Cairo', system-ui, sans-serif;
        margin: 10px 0;
        width: 100%;
    ">
        <div style="font-size: 12px; color: #64748b; font-weight: bold; margin-bottom: 8px;">📢 عروض وإعلانات رعاية المنصة:</div>
        <div style="
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            align-items: stretch;
        ">
            {cards_html}
        </div>
    </div>
    """
    
    await cl.Message(content=grid_container).send()

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
        
        # 🎯 إظهار الإعلانات التفاعلية الشبكية عند تكرار الشرط
        if count % 2 == 0:
            await send_ad_card()
        
    except Exception as e:
        msg.content = f"حدث خطأ أثناء معالجة الطلب: {str(e)}"
        await msg.update()
