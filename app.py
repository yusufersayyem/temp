# 1. تعريف قائمة الإعلانات (يمكنك استخدام روابط صور خارجية CDN لتخفيف العبء عن Render)
ADS_LIST = [
    {
        "title": "📢 إعلان: خصم 20% على الدورات البرمجية!",
        "image_url": "https://via.placeholder.com/600x200?text=Ad+1",
        "link": "https://example.com/ad1"
    },
    {
        "title": "🚀 إعلان: استضف مشاريعك بسهولة مع خدماتنا السحابية!",
        "image_url": "https://via.placeholder.com/600x200?text=Ad+2",
        "link": "https://example.com/ad2"
    },
    {
        "title": "💡 إعلان: اشترك الآن في النشرة البرمجية اليومية!",
        "image_url": "https://via.placeholder.com/600x200?text=Ad+3",
        "link": "https://example.com/ad3"
    }
]

@cl.on_chat_start
async def start():
    # تهيئة العداد ومؤشر الإعلان لجلسة المستخدم
    cl.user_session.set("message_count", 0)
    cl.user_session.set("ad_index", 0)

@cl.on_message
async def main(message: cl.Message):
    # جلب القيم الحالية وتحديث العداد
    count = cl.user_session.get("message_count", 0) + 1
    ad_index = cl.user_session.get("ad_index", 0)
    cl.user_session.set("message_count", count)

    user_text = message.content
    
    # 2. منطق استخراج الرد عبر Cohere
    user_response = co.embed(
        texts=[user_text],
        model="embed-multilingual-v3.0",
        input_type="search_query"
    )
    
    user_embedding = np.array(user_response.embeddings[0])
    user_embedding = user_embedding / np.linalg.norm(user_embedding)
    
    similarities = np.dot(patterns_embeddings, user_embedding)
    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]
    
    THRESHOLD = 0.40
    if best_score >= THRESHOLD:
        matched_intent = intents_mapping[best_match_idx]
        selected_response = random.choice(matched_intent["responses"])
    else:
        selected_response = "عذراً، لم أفهم قصدك بوضوح. هل يمكنك إعادة الصياغة؟"

    # 3. التحقق من الشرط (كل 3 استعلامات)
    elements = []
    if count % 3 == 0:
        current_ad = ADS_LIST[ad_index]
        
        # إرفاق صورة الإعلان
        elements.append(
            cl.Image(
                name=f"ad_{ad_index + 1}",
                url=current_ad["image_url"],
                display="inline"
            )
        )
        
        # إضافة النص الخاص بالإعلان مع الرابط
        selected_response += f"\n\n---\n{current_ad['title']}\n[اضغط هنا للمزيد]({current_ad['link']})"
        
        # الانتقال للإعلان التالي (مع العودة للأول بعد انتهاء القائمة 0 -> 1 -> 2 -> 0)
        next_ad_index = (ad_index + 1) % len(ADS_LIST)
        cl.user_session.set("ad_index", next_ad_index)

    # 4. إرسال الرد النهائي
    await cl.Message(content=selected_response, elements=elements).send()
