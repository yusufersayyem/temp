import os
import json
import random
import cohere
import numpy as np
import chainlit as cl

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

patterns_list = []
intents_mapping = []
patterns_embeddings = None

def load_and_embed_intents(json_path="intents.json"):
    global patterns_list, intents_mapping, patterns_embeddings
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for intent in data["intents"]:
        tag = intent["tag"]
        responses = intent["responses"]
        # جلب مسار الصورة إن وجد
        image_path = intent.get("image", None)
        
        for pattern in intent["patterns"]:
            patterns_list.append(pattern)
            intents_mapping.append({
                "tag": tag,
                "responses": responses,
                "image": image_path
            })
            
    print("جاري استدعاء Cohere API لحساب متجهات الـ patterns...")
    response = co.embed(
        texts=patterns_list,
        model="embed-multilingual-v3.0",
        input_type="search_document"
    )
    
    embeddings_matrix = np.array(response.embeddings)
    patterns_embeddings = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    print("تم تجهيز متجهات الـ patterns بنجاح!")

load_and_embed_intents()

@cl.on_message
async def main(message: cl.Message):
    user_text = message.content
    
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
    elements = []
    
    if best_score >= THRESHOLD:
        matched_intent = intents_mapping[best_match_idx]
        selected_response = random.choice(matched_intent["responses"])
        
        # التأكد من وجود صورة مرافقة للـ Intent
        image_path = matched_intent.get("image")
        if image_path:
            # إذا كان الرابط أونلاين استخدم url، وإذا كان محلياً استخدم path
            if image_path.startswith("http"):
                elements.append(
                    cl.Image(name="intent_image", url=image_path, display="inline")
                )
            elif os.path.exists(image_path):
                elements.append(
                    cl.Image(name="intent_image", path=image_path, display="inline")
                )
    else:
        selected_response = "عذراً، لم أفهم قصدك بوضوح. هل يمكنك إعادة الصياغة؟"
        
    await cl.Message(content=selected_response, elements=elements).send()
