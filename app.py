import os
import json
import chainlit as cl
from huggingface_hub import AsyncInferenceClient

# جلب المفاتيح والإعدادات
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
JSON_FILE_PATH = "qa_data.json"

client = AsyncInferenceClient(model=MODEL_ID, token=HF_TOKEN)

# تحميل ملف الـ JSON في الذاكرة عند بدء التشغيل (استهلاك لا يذكر للذاكرة)
def load_json_data():
    if os.path.exists(JSON_FILE_PATH):
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

qa_database = load_json_data()

# دالة للبحث البسيط في بيانات الـ JSON بناءً على الكلمات المفتاحية
def search_in_json(user_query, top_k=2):
    results = []
    query_words = set(user_query.lower().split())
    
    for item in qa_database:
        question = item.get("question", "").lower()
        full_qa = item.get("full_qa", "").lower()
        
        # حساب عدد الكلمات المشتركة بين سؤال المستخدم والقاعدة
        score = sum(1 for word in query_words if word in question or word in full_qa)
        
        if score > 0:
            results.append((score, item))
            
    # ترتيب النتائج من الأكثر تطابقاً إلى الأقل
    results.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in results[:top_k]]

@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "message_history",
        [{
            "role": "system", 
            "content": "أنت مساعد ذكي ومفيد. اعتمِد على السياق المرفق للإجابة عن أسئلة المستخدم بوضوح ودقة. إذا لم تجد الإجابة في السياق، أجب بما تعرفه بشكل عام."
        }]
    )

@cl.on_message
async def main(message: cl.Message):
    message_history = cl.user_session.get("message_history")

    # 1. البحث عن أفضل تطابق داخل ملف JSON
    matched_items = search_in_json(message.content, top_k=2)
    
    context_text = ""
    if matched_items:
        context_text = "\n\n".join([item.get("full_qa", "") for item in matched_items])

    # 2. دمج السياق المستخرج مع سؤال المستخدم
    if context_text:
        user_prompt_with_context = f"المعلومات المستخرجة من قاعدة البيانات:\n{context_text}\n\nسؤال المستخدم: {message.content}"
    else:
        user_prompt_with_context = message.content

    message_history.append({"role": "user", "content": user_prompt_with_context})

    msg = cl.Message(content="")
    await msg.send()

    try:
        # 3. إرسال النص مع السياق إلى نموذج Qwen
        stream = await client.chat_completion(
            messages=message_history,
            max_tokens=2048,
            temperature=0.3,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                await msg.stream_token(token)

        await msg.update()
        
        # حفظ الرسالة الأصلية بدون السياق لتنظيف السجل
        message_history[-1] = {"role": "user", "content": message.content}
        message_history.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("message_history", message_history)

    except Exception as e:
        msg.content = f"حدث خطأ أثناء الاتصال بالنظام: {str(e)}"
        await msg.update()
