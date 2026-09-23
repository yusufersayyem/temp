def get_embeddings_from_hf(texts, retries=3, delay=5):
    """استدعاء Hugging Face API مع دعم تنسيق Sentence Similarity للنموذج"""
    
    # تحويل النصوص إلى الهيكل الصحيح الذي يطلبه SentenceSimilarityPipeline
    # نضع النص الأول كـ source_sentence ونقارنه بباقي النصوص
    payload = {
        "inputs": {
            "source_sentence": texts[0],
            "sentences": texts
        },
        "options": {"wait_for_model": True}
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
            
            if response.status_code == 200:
                res_data = response.json()
                
                # إذا كانت الاستجابة قائمة أرقام مباشرة (Feature Extraction)
                if isinstance(res_data, list):
                    embeddings = np.array(res_data)
                else:
                    # في حال إرجاع استجابة بأسماء مفاتيح أخرى
                    embeddings = np.array(list(res_data.values()))

                # التأكد من صحة الأبعاد (2D Array)
                if len(embeddings.shape) == 1:
                    embeddings = np.expand_dims(embeddings, axis=0)

                # معايرة المتجهات (Normalization)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                return embeddings / norms
            else:
                print(f"محاولة {attempt + 1}: فشل الاستجابة ({response.status_code}) - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"محاولة {attempt + 1}: خطأ في الاتصال بالشبكة ({e})")
            
        if attempt < retries - 1:
            time.sleep(delay)

    raise RuntimeError("تعذر الاتصال بـ Hugging Face API بعد عدة محاولات.")
