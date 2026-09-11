import os
import json
import random
import time
import requests
import numpy as np
import chainlit as cl
from sklearn.metrics.pairwise import cosine_similarity

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_TOKEN = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_embedding(texts, retries=3, delay=5):
    """دالة محسّنة مع خاصية إعادة المحاولة عند انتظار تحفيز النموذج"""
    payload = {"inputs": texts, "options": {"wait_for_model": True}}
    
    for attempt in range(retries):
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            # التعامل مع اختلاف الهياكل المرجعة من الـ API
            res_np = np.array(result)
            if res_np.ndim == 3:
                res_np = res_np.mean(axis=1) # متوسط التضمينات إذا كان النموذج يرجع Token Embeddings
            return res_np
            
        elif response.status_code == 503:
            # النموذج قيد التحميل، ننتظر قليلاً ثم نعيد المحاولة
            print(f"النموذج قيد التحميل... محاولة {attempt + 1} من {retries}")
            time.sleep(delay)
        else:
            # طباعة تفاصيل الخطأ في التيرمينال لمعرفته بدقة
            print(f"HF API Error: {response.status_code} - {response.text}")
            break
            
    raise Exception("فشل الاتصال بـ Hugging Face API بعد عدة محاولات.")
