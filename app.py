import os
import json
import random
from openai import OpenAI
import chainlit as cl

# 1. تهيئة عميل OpenAI للربط مع منصة xKiro
# يفضل وضع الـ API Key الخاص بـ xKiro في متغيرات البيئة XKIRO_API_KEY أو تمريره مباشرة
XKIRO_API_KEY = os.environ.get("XKIRO_API_KEY", "ضع_مفتاح_الـ_API_هنا")

client = OpenAI(
    base_url="https://api.xkiro.com/v1",
    api_key=XKIRO_API_KEY
)

# متغير عام لتخزين بيانات النوايا (intents)
intents_data = ""

# 2. قائمة الإعلانات
ADS_LIST = [
    {
        "title": "📢 إعلان: خصم 20% على الدورات البرمجية!",
        "image_url": "https://ik.imagekit.io/63rncvror/ad5.webp?updatedAt=1785601364212?text=Ad+1",
        "link": "https://www.asiacell.com/personal?gad_source=1&gad_campaignid=21900349889&gbraid=0AAAAAoo1Wz1oagyMn1KP-g_RQLeaW-yb7&gclid=CjwKCAjwwrPVBhA1EiwAv_YO-cQk6fBJf1TLInMbzA-1fWCHzq0GL8DtuqSP1Bs71KL7HDBh9LeT5hoCDOoQAvD_BwE"
    },
    {
        "title": "مطعم خطار الموصل تخفيضات موسمية 15%",
        "image_url": "https://ik.imagekit.io/63rncvror/ad1.webp?updatedAt=1785601369756?text=Ad+2",
        "link": "https://www.facebook.com/khutarrest"
    },
    {
        "title": "جامعة النور تعلن عن تخفيضات لكوادر التربية ",
        "image_url": "https://ik.imagekit.io/63rncvror/ad10.webp?updatedAt=1785601362911?text=Ad+3",
        "link": "https://alnoor.edu.iq/ar"
    }
]

# 3. تحميل ملف النوايا عند بدء التشغيل
def load_intents(json_path="intents.json"):
    global intents_data
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # تحويل البيانات إلى نص JSON للـ Prompt
            intents_data = json.dumps(data, ensure_ascii=False)
            print("تم تحميل ملف النوايا بنجاح!")
    except Exception as e:
        print(f"خطأ في تحميل ملف intents.json: {e}")

load_intents()

@cl.on_chat_start
async def start():
    cl.user_session.set("message_count", 0)
    cl.user_session.set("ad_index", 0)

@cl.on_message
async def main(message: cl.Message):
    count = cl.user_session.get("message_count", 0) + 1
    ad_index = cl.user_session.get("ad_index", 0)
    cl.user_session.set("message_count", count)

    user_text = message.content

    # صياغة التعليمات لنموذج Qwen (System Prompt)
    system_prompt = f"""أنت مساعد ذكي ومفيد. لديك قائمة بالنوايا والردود المتاحة أدناه بفرمت JSON:
{intents_data}

المطلوب منك:
1. تحليل رسالة المستخدم وفهم نيتها.
2. إذا وجدت نية مطابقة في البيانات أعلاه، اختر رداً مناسباً وشبيهاً بالردود المعرفة للنية (أو أحد الردود المكتوبة).
3. إذا لم تجد أي نية مطابقة، أجب بـ: "عذراً، لم أفهم قصدك بوضوح. هل يمكنك إعادة الصياغة؟"
4. أجب باللغة العربية بأسلوب طبيعي ومباشر ودون ذكر تفاصيل الـ JSON أو التقنيات للمستخدم.
"""

    try:
        # استدعاء نموذج Qwen عبر xKiro
        response = client.chat.completions.create(
            model="qwen/qwen3-max:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            temperature=0.3
        )
        selected_response = response.choices[0].message.content.strip()
    except Exception as e:
        selected_response = f"حدث خطأ أثناء الاتصال بالنموذج: {e}"

    # إرفاق الإعلان كل 3 استعلامات
    elements = []
    if count % 3 == 0:
        current_ad = ADS_LIST[ad_index]
        
        elements.append(
            cl.Image(
                name=f"ad_{ad_index + 1}",
                url=current_ad["image_url"],
                display="inline"
            )
        )
        
        selected_response += f"\n\n---\n{current_ad['title']}\n[اضغط هنا للمزيد]({current_ad['link']})"
        
        next_ad_index = (ad_index + 1) % len(ADS_LIST)
        cl.user_session.set("ad_index", next_ad_index)

    # إرسال الرسالة النهائي
    await cl.Message(content=selected_response, elements=elements).send()
