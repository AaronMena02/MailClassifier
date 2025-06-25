from flask import Flask, render_template, request, jsonify
from gmail_api import get_gmail_service, list_messages, get_message, get_plain_text
from models import classify_email, preprocesar_texto
from email.header import decode_header

app = Flask(__name__)

@app.route("/api/correos")
def api_correos():
    emails = obtener_emails_clasificados(max_results=10)
    return jsonify(emails)

# Función auxiliar para cargar y clasificar correos
def obtener_emails_clasificados(max_results=20):
    service = get_gmail_service()
    messages = list_messages(service, max_results=max_results)
    
    emails = []
    for m in messages:
        email_msg = get_message(service, m['id'])
        if email_msg:
            subject = decodificar_header(email_msg.get("Subject", "(sin asunto)"))
            body = get_plain_text(email_msg)
            
            if not body.strip():
                continue  # Evita clasificar correos vacíos

            entrada_modelo = preprocesar_texto(subject, body)
            classification = classify_email(entrada_modelo)

            emails.append({
                "from": decodificar_header(email_msg.get("From", "(sin remitente)")),
                "subject": subject,
                "text": body,
                "classification": classification
            })
    return emails

def decodificar_header(header_value):
    if not header_value:
        return ""
    decoded_fragments = decode_header(header_value)
    decoded_string = ''
    for fragment, encoding in decoded_fragments:
        if isinstance(fragment, bytes):
            try:
                decoded_string += fragment.decode(encoding or 'utf-8', errors='replace')
            except:
                decoded_string += fragment.decode('utf-8', errors='replace')
        else:
            decoded_string += fragment
    return decoded_string

# Ruta principal
@app.route('/')
def home():
    emails = obtener_emails_clasificados()
    return render_template('index.html', emails=emails)

# Ruta para filtros dinámicos
@app.route('/filtro/<tipo>/<valor>')
def filtrar(tipo, valor):
    if tipo not in ["sentimiento", "prioridad", "categoria"]:
        return "Filtro no válido", 400

    clave = {
        "sentimiento": "sentiment",
        "prioridad": "priority",
        "categoria": "category"
    }[tipo]

    emails = obtener_emails_clasificados(max_results=30)
    emails_filtrados = [e for e in emails if e["classification"].get(clave, "").lower() == valor.lower()]

    return render_template('index.html', emails=emails_filtrados)

if __name__ == '__main__':
    app.run(debug=True)
