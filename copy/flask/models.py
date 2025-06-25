from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Rutas a los modelos guardados (ajusta si están en otra carpeta)
modelo_sentiment_path = "../sentiment_model"
modelo_priority_path = "../priority_model"
modelo_category_path = "../category_model"

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
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512  # Usar longitud estándar
    )
    
    # NO crear position_ids manualmente - deja que el modelo los maneje
    with torch.no_grad():
        outputs = model(**inputs)
    
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
    category_map = {0: "comercial", 1: "otro", 2: "queja", 3: "solicitud"}

    return {
        "sentiment": sentiment_map.get(sentiment_idx, str(sentiment_idx)),
        "sentiment_prob": sentiment_prob,
        "priority": priority_map.get(priority_idx, str(priority_idx)),
        "priority_prob": priority_prob,
        "category": category_map.get(category_idx, str(category_idx)),
        "category_prob": category_prob,
    }

# Ejemplos de correos reales (asunto + cuerpo)
test_emails = [
    """Asunto: Pedido recibido con retraso
    Hola equipo,
    Quería informar que mi pedido llegó con dos días de retraso, aunque el producto está en buen estado.
    Gracias.""",

    """Asunto: Producto defectuoso
    Estimados,
    Recibí el producto con un defecto y quisiera solicitar un reemplazo urgente.
    Espero una pronta respuesta.""",

    """Asunto: Excelente servicio
    Buenas tardes,
    Solo quería agradecer la rapidez y eficiencia en la entrega de mi pedido. Muy satisfecho con el servicio.""",

    """Asunto: Problemas con la factura
    Hola,
    He detectado cargos que no reconozco en la factura y necesito que se revise cuanto antes.
    Saludos.""",

    """Asunto: Consulta sobre producto
    Buen día,
    Me gustaría saber si el modelo X200 está disponible en color negro y si tiene garantía extendida.
    Gracias.""",

    """Asunto: Gracias por la ayuda
    Quiero expresar mi agradecimiento por la asistencia que me brindaron ayer. Todo fue excelente.""",

    """Asunto: Pedido incompleto
    Buenos días,
    El pedido que recibí está incompleto. Faltan varios artículos que ya he pagado.
    Espero una solución inmediata.""",

    """Asunto: Solicitud de información
    Estimados,
    Podrían enviarme los detalles técnicos del producto Z45, por favor?
    Quedo atento a su respuesta.""",

    """Asunto: Felicitaciones por el servicio
    Buen trabajo con el soporte técnico, me ayudaron rápido y resolvieron todo a la perfección.""",

    """Asunto: Incidencia no resuelta
    Llevo días esperando que resuelvan el problema con mi pedido, pero no recibo respuestas claras.
    Esto es muy frustrante."""
]

for email_text in test_emails:
    result = classify_email(email_text)
    print(f"Texto:\n{email_text}\n")
    print(f"Predicción:")
    print(f" Sentimiento: {result['sentiment']} (confianza: {result['sentiment_prob']:.2f})")
    print(f" Prioridad: {result['priority']} (confianza: {result['priority_prob']:.2f})")
    print(f" Categoría: {result['category']} (confianza: {result['category_prob']:.2f})\n")
    print("-" * 50)


