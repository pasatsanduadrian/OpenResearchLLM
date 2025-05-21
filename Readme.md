# OpenResearchLLM

Web app pentru generare automată de rapoarte de cercetare cu LLM (Gemini/Mistral), web search și export PDF — totul cu un singur click.

---

## 🧰 Structura repository

- `requirements.txt` — lista completă de pachete necesare
- `.env.example` — model de configurare variabile de mediu
- `main.py`, `app.py`, `app2.py` — 3 variante de script principal, alege-o pe cea dorită
- `README.md` — acest ghid rapid

---

## ⚡ Instalare rapidă în Google Colab

### 1. Clonare repository & instalare dependențe

```python
!git clone https://github.com/pasatsanduadrian/OpenResearchLLM.git
%cd OpenResearchLLM
!pip install -r requirements.txt
```
### 2. Configurare variabile de mediu

Creează fișierul .env cu valorile tale (sau editează .env.example). Exemplu rapid în Colab/notebook:

```python
import os
with open(".env", "w") as f:
    f.write("""
GEMINI_API_KEY=YOUR_GEMINI_KEY
MISTRAL_API_KEY=YOUR_MISTRAL_KEY
PORT=5099
NGROK_TOKEN=YOUR_NGROK_TOKEN
NGROK_HOSTNAME=your-ngrok-subdomain.ngrok-free.app
""")
```

### 3. Asigură-te că requirements.txt include pachetele:

```python
flask
pyngrok
python-dotenv
mistralai
google-generativeai
duckduckgo-search
weasyprint
bs4
markdown
```

### 4. Alegerea și lansarea aplicației

Ai la dispoziție 3 variante de script:

    main.py

    app.py

    app2.py

Recomandare: testează-le și folosește-o pe cea care răspunde cel mai bine cerințelor tale.

Exemplu de rulare în Colab (sau terminal):

```python
!python app.py
```
### 5. Accesarea aplicației

După pornirea serverului, va apărea un link public ngrok (https://...ngrok-free.app).
Deschide-l în browser pentru a utiliza aplicația web.

#### 🔑 Variabile de mediu necesare (.env)

```python
GEMINI_API_KEY=your_gemini_key
MISTRAL_API_KEY=your_mistral_key
PORT=5099
NGROK_TOKEN=your_ngrok_token
NGROK_HOSTNAME=your-ngrok-subdomain.ngrok-free.app
```

    Cheile API se obțin de pe Google AI Studio (Gemini), Mistral, iar NGROK se obține de pe ngrok.com.

#### 📝 Funcționalități

    Căutare web automată pe baza întrebării date (DuckDuckGo, extragere text)

    Consultare surse, analiză, structurare raport (Gemini/Mistral LLM)

    Raport structurat, citări automate, export PDF   

    Interfață web modernă (Flask + Bootstrap)

    Integrare rapidă via Colab (rulează direct în notebook!)


#### 🚀 Quick Start (TL;DR)
```python
!git clone https://github.com/pasatsanduadrian/OpenResearchLLM.git
%cd OpenResearchLLM
!pip install -r requirements.txt
# Creează .env cu cheile tale (sau folosește .env.example)
!python app.py
```

#### Accesează linkul public ngrok generat și începe cercetarea!


