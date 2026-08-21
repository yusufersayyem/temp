import os
from dotenv import load_dotenv
import chainlit as cl

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_qdrant import QdrantVectorStore

# استيرادات آمنة وتتوافق مع أحدث إصدارات LangChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
COLLECTION_NAME = "my_pdf_documents"

def format_docs(docs):
    """دمج النصوص المسترجعة من Qdrant"""
    return "\n\n".join(doc.page_content for doc in docs)

@cl.on_chat_start
async def on_chat_start():
    msg = cl.Message(content="جاري الاتصال بالنموذج وقاعدة البيانات...")
    await msg.send()

    # 1. Embedding Model
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. Qdrant Retriever
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 3. LLM (Qwen 2.5 7B via Hugging Face API)
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        temperature=0.2,
        max_new_tokens=512
    )

    # 4. System Prompt
    template = """أنت مساعد ذكي ومؤدب متخصص في الإجابة على استفسارات تربية نينوى.
استخدم المعلومات الواردة في السياق المرفق فقط للإجابة على سؤال المستخدم.
إذا لم تجد الإجابة في السياق، قل بوضوح: 'عذراً، لا تتوفر هذه المعلومة ضمن بيانات تربية نينوى المتاحة حالياً.' ولا تقم بابتكار إجابات من عندك.

السياق المتاح:
{context}

السؤال: {question}
الإجابة:"""

    prompt = ChatPromptTemplate.from_template(template)

    # 5. Build RAG Chain using LCEL (تمنع خطأ ModuleNotFoundError تماماً)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    cl.user_session.set("rag_chain", rag_chain)
    
    msg.content = "أهلاً بك! أنا مساعدك الذكي الخاص بتربية نينوى. كيف يمكنني مساعدتك اليوم؟"
    await msg.update()

@cl.on_message
async def on_message(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")
    
    # استدعاء الـ Chain والحصول على النص مباشرة
    response_text = await rag_chain.ainvoke(message.content)
    
    await cl.Message(content=response_text).send()
