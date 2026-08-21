import os
from dotenv import load_dotenv
import chainlit as cl

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COLLECTION_NAME = "my_pdf_documents"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@cl.on_chat_start
async def on_chat_start():
    # 1. استخدام FastEmbed لنموذج bge-m3 (خفيف جداً على الذاكرة ومطابق لبياناتك)
    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-m3"
    )

    # 2. Qdrant Retriever
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 3. LLM عبر Groq (سريع جداً ودقيق)
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )

    # 4. Prompt Template
    template = """أنت مساعد ذكي ومؤدب متخصص في الإجابة على استفسارات تربية نينوى.
استخدم المعلومات الواردة في السياق المرفق فقط للإجابة على سؤال المستخدم.
إذا لم تجد الإجابة في السياق، قل بوضوح: 'عذراً، لا تتوفر هذه المعلومة ضمن بيانات تربية نينوى المتاحة حالياً.' ولا تقم بابتكار إجابات من عندك.

السياق المتاح:
{context}

السؤال: {question}
الإجابة:"""

    prompt = ChatPromptTemplate.from_template(template)

    # 5. Build LCEL Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    cl.user_session.set("rag_chain", rag_chain)
    await cl.Message(content="أهلاً بك! أنا مساعدك الذكي الخاص بتربية نينوى. كيف يمكنني مساعدتك اليوم؟").send()

@cl.on_message
async def on_message(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")
    
    msg = cl.Message(content="")
    await msg.send()

    # بث الإجابة بشكل تدريجي مريح وسريع
    async for chunk in rag_chain.astream(message.content):
        await msg.stream_token(chunk)
        
    await msg.update()
