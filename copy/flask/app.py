from flask import Flask, render_template
from gmail_api import get_gmail_service, list_messages, get_message, get_plain_text
from models import classify_email

app = Flask(__name__)

@app.route('/')
def home():
    service = get_gmail_service()
    messages = list_messages(service, max_results=5)
    
    emails = []
    for m in messages:
        email_msg = get_message(service, m['id'])
        if email_msg:
            text = get_plain_text(email_msg)
            classification = classify_email(text)
            emails.append({
                "from": email_msg["From"],
                "subject": email_msg["Subject"],
                "text": text,
                "classification": classification
            })
    
    return render_template('index.html', emails=emails)

if __name__ == '__main__':
    app.run(debug=True)
