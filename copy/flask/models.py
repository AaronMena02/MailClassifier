from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import json

# Rutas a los modelos guardados
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATHS = {
    "sentiment": os.path.join(BASE_DIR, "../sentiment_model"),
    "priority": os.path.join(BASE_DIR, "../priority_model"),
    "category": os.path.join(BASE_DIR, "../category_model")
}

# Función para cargar modelo, tokenizer y mappings desde config
def load_model_components(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    id2label = {int(k): v for k, v in config["id2label"].items()}
    
    return tokenizer, model, id2label

# Carga de todos los modelos
tokenizer_sentiment, model_sentiment, sentiment_map = load_model_components(MODELS_PATHS["sentiment"])
tokenizer_priority, model_priority, priority_map = load_model_components(MODELS_PATHS["priority"])
tokenizer_category, model_category, category_map = load_model_components(MODELS_PATHS["category"])


# Función de predicción
def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_idx].item()
    return pred_idx, confidence

"""def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    print("Input text:", text)
    print("Logits:", outputs.logits)
    print("Logits shape:", outputs.logits.shape)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_idx].item()
    return pred_idx, confidence"""


# Clasificación completa del correo
def classify_email(text):
    sentiment_idx, sentiment_prob = predict(text, tokenizer_sentiment, model_sentiment)
    priority_idx, priority_prob = predict(text, tokenizer_priority, model_priority)
    category_idx, category_prob = predict(text, tokenizer_category, model_category)
    
    return {
        "sentiment": sentiment_map.get(sentiment_idx, str(sentiment_idx)),
        "sentiment_prob": sentiment_prob,
        "priority": priority_map.get(priority_idx, str(priority_idx)),
        "priority_prob": priority_prob,
        "category": category_map.get(category_idx, str(category_idx)),
        "category_prob": category_prob
    }
