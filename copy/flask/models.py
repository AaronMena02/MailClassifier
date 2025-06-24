from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Rutas a los modelos guardados (ajusta si están en otra carpeta)
modelo_sentiment_path = "modelo_sentiment"
modelo_priority_path = "modelo_priority"
modelo_category_path = "modelo_categoria"

# Carga tokenizers y modelos
tokenizer_sentiment = AutoTokenizer.from_pretrained(modelo_sentiment_path)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(modelo_sentiment_path)

tokenizer_priority = AutoTokenizer.from_pretrained(modelo_priority_path)
model_priority = AutoModelForSequenceClassification.from_pretrained(modelo_priority_path)

tokenizer_category = AutoTokenizer.from_pretrained(modelo_category_path)
model_category = AutoModelForSequenceClassification.from_pretrained(modelo_category_path)

print("Tamaño vocabulario del modelo sentiment:", model_sentiment.config.vocab_size)
print("Tamaño vocabulario del tokenizer sentiment:", tokenizer_sentiment.vocab_size)
print("Max position embeddings modelo sentiment:", model_sentiment.config.max_position_embeddings)

# Función predict corregida para truncar a max_position_embeddings
def predict(text, tokenizer, model):
    max_len = model.config.max_position_embeddings  # 130
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding='max_length',
        max_length=max_len
    )
    
    # Crear position_ids explícitos de 0 a max_len-1, evitando índices fuera de rango
    position_ids = torch.arange(max_len).unsqueeze(0)  # shape (1, max_len)
    inputs['position_ids'] = position_ids

    try:
        outputs = model(**inputs)
    except Exception as e:
        print("Error al ejecutar modelo:", e)
        raise e
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_idx = torch.argmax(probs, dim=1).item()
    return pred_idx, probs[0][pred_idx].item()


def classify_email(text):
    sentiment_idx, sentiment_prob = predict(text, tokenizer_sentiment, model_sentiment)
    priority_idx, priority_prob = predict(text, tokenizer_priority, model_priority)
    category_idx, category_prob = predict(text, tokenizer_category, model_category)
    
    # Mapas de índices a etiquetas (ajusta según tus modelos)
    sentiment_map = {0: "negativo", 1: "neutro", 2: "positivo"}
    priority_map = {0: "baja", 1: "media", 2: "alta"}
    category_map = {0: "queja", 1: "solicitud", 2: "comercial", 3: "otro"}

    return {
        "sentiment": sentiment_map.get(sentiment_idx, str(sentiment_idx)),
        "sentiment_prob": sentiment_prob,
        "priority": priority_map.get(priority_idx, str(priority_idx)),
        "priority_prob": priority_prob,
        "category": category_map.get(category_idx, str(category_idx)),
        "category_prob": category_prob,
    }
