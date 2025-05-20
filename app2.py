import os
import time
import asyncio
import threading
import nest_asyncio
from flask import Flask, render_template_string, request, send_file
from pyngrok import ngrok
from dotenv import load_dotenv
from mistralai import Mistral
from google import genai
import aiohttp
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import io
import re
from markdown import markdown

try:
    from weasyprint import HTML  # Folosim WeasyPrint pentru generare PDF
except ImportError:
    HTML = None  # Avertizăm dacă lipsește la rulare

nest_asyncio.apply()
load_dotenv()

# --- Config ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
PORT = int(os.getenv("PORT", "5099"))
NGROK_TOKEN = os.getenv("NGROK_TOKEN")
NGROK_HOSTNAME = os.getenv("NGROK_HOSTNAME", None)

GEMINI_MODEL = "gemini-2.0-flash"
MISTRAL_MODEL = "mistral-large-latest"

# --- Rate Limiters ---
class AsyncRateLimiter:
    def __init__(self, min_interval_s):
        self.min_interval_s = min_interval_s
        self._last_call = 0
    async def wait(self):
        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self.min_interval_s:
            await asyncio.sleep(self.min_interval_s - elapsed)
        self._last_call = time.time()

gemini_async_limiter = AsyncRateLimiter(4.0)
mistral_async_limiter = AsyncRateLimiter(1.0)

# --- Init LLM clients ---
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

def combine_messages(messages):
    return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])

async def call_llm(messages):
    prompt = combine_messages(messages)
    try:
        await gemini_async_limiter.wait()
        resp = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=GEMINI_MODEL,
            contents=prompt
        )
        reply = resp.text.strip()
        if reply:
            return reply
        else:
            raise Exception("Empty Gemini reply")
    except Exception as e:
        print("[Fallback] Gemini failed, try Mistral:", e)
        try:
            await mistral_async_limiter.wait()
            mistral_resp = await asyncio.to_thread(
                mistral_client.chat.complete,
                model=MISTRAL_MODEL,
                messages=messages
            )
            return mistral_resp.choices[0].message.content.strip()
        except Exception as ex:
            print("[Error] Both LLMs failed:", ex)
            return "Error: Both LLMs failed."

# --- Async research pipeline ---
async def generate_search_queries(user_query, search_lang):
    prompt = (
        f"Ești un asistent de cercetare. Pentru întrebarea utilizatorului, generează până la patru interogări precise pentru web, în limba: {search_lang}."
        "\nRăspunde doar cu o listă Python de stringuri, exemplu: ['query1', 'query2']."
    )
    messages = [
        {"role": "system", "content": "Ești un asistent util și precis."},
        {"role": "user", "content": f"Întrebare: {user_query}\n\n{prompt}"}
    ]
    response = await call_llm(messages)
    clean = response.strip().replace('```python', '').replace('```', '').strip()
    try:
        queries = eval(clean)
        if isinstance(queries, list):
            return queries
        else:
            return []
    except Exception:
        return []

async def perform_search(query, max_results=8):
    try:
        def sync_search():
            with DDGS() as ddgs:
                return [r.get("href") for r in ddgs.text(query, max_results=max_results) if "href" in r]
        return await asyncio.to_thread(sync_search)
    except Exception:
        return []

async def fetch_text(session, url):
    try:
        async with session.get(url, timeout=10) as resp:
            if resp.status == 200:
                html = await resp.text()
                soup = BeautifulSoup(html, "html.parser")
                for s in soup(["script", "style"]): s.decompose()
                return soup.get_text(separator=" ", strip=True)
    except Exception:
        return ""
    return ""

async def evaluate_and_extract(user_query, search_query, page_text, report_lang):
    prompt = (
        f"Ești evaluator. Pentru întrebarea '{user_query}' și pagina web de mai jos, evaluează dacă e utilă. Dacă DA, extrage contextul relevant ca text simplu, răspunzând doar în limba: {report_lang}. "
        "Dacă nu e utilă, răspunde exact: Not Useful."
    )
    messages = [
        {"role": "system", "content": "Ești evaluator și extractor concis."},
        {"role": "user", "content": f"Întrebare: {user_query}\nCăutare: {search_query}\n\nConținut pagină (primele 20000 caractere):\n{page_text[:20000]}\n\n{prompt}"}
    ]
    resp = await call_llm(messages)
    cleaned = resp.strip().replace("```", "")
    if cleaned.lower() == "not useful":
        return None
    return cleaned

async def get_new_search_queries(user_query, previous_search_queries, all_contexts, search_lang):
    context_combined = "\n".join(all_contexts)
    prompt = (
        f"Ești un planificator analitic. Pe baza interogărilor de până acum (în {search_lang}) și a contextelor extrase, vezi dacă e nevoie de alte căutări. "
        "Dacă DA, propune până la patru noi interogări ca listă Python. Dacă NU, răspunde cu exact <done>."
    )
    messages = [
        {"role": "system", "content": "Planificator sistematic."},
        {"role": "user", "content": f"Întrebare: {user_query}\nCăutări precedente: {previous_search_queries}\n\nContext extras:\n{context_combined}\n\n{prompt}"}
    ]
    response = await call_llm(messages)
    if response.strip() == "<done>":
        return "<done>"
    clean = response.strip().replace('```python', '').replace('```', '').strip()
    try:
        queries = eval(clean)
        if isinstance(queries, list):
            return queries
    except Exception:
        pass
    return []

async def generate_final_report(user_query, all_contexts, report_lang, context_sources):
    bib_lines = [f"[{i+1}] {src}" for i, src in enumerate(context_sources)] if context_sources else []
    prompt = (
        f"Ești un cercetător și redactor. Pe baza contextului și a întrebării, scrie un raport structurat, cu secțiuni markdown: Titlu, Rezumat executiv, introducere, secțiuni, concluzie. "
        f"Redactează clar, formal, EXCLUSIV în limba: {report_lang}."
        " Pune referințele ca bullet list la final, format [1] url, [2] url etc, sub titlul 'Bibliografie'. "
        "NU include delimitatori de tipul ``` sau code fence."
        "\n\nReferințe pentru context:\n" + "\n".join(bib_lines)
    )
    context_combined = "\n".join(all_contexts)
    messages = [
        {"role": "system", "content": "Redactor raport profesionist."},
        {"role": "user", "content": f"Întrebare: {user_query}\n\nContext:\n{context_combined}\n\n{prompt}"}
    ]
    return await call_llm(messages)

# --- Async research orchestrator ---
async def full_research_pipeline(user_query, iter_limit, search_lang, report_lang, progress_cb=None):
    aggregated_contexts = []
    all_search_queries = []
    all_links = []
    all_links_flat = []
    iteration = 0
    progress = []

    async with aiohttp.ClientSession() as session:
        # Step 1: initial queries
        new_search_queries = await generate_search_queries(user_query, search_lang)
        if not new_search_queries:
            progress.append("LLM nu a returnat interogări de căutare.")
            return {"progress": progress, "report": "Niciun raport generat."}
        all_search_queries.extend(new_search_queries)

        while iteration < iter_limit:
            progress.append(f"Iterația {iteration+1}: căutare și extragere date...")
            if progress_cb: progress_cb(progress)

            # Step 2: search each query
            search_tasks = [perform_search(q) for q in new_search_queries]
            search_results = await asyncio.gather(*search_tasks)
            unique_links = {}
            for idx, links in enumerate(search_results):
                q = new_search_queries[idx]
                for link in links:
                    if link and link not in unique_links:
                        unique_links[link] = q

            all_links_flat.extend(unique_links.keys())

            # Step 3: fetch & extract each link
            link_tasks = [fetch_text(session, link) for link in unique_links]
            pages = await asyncio.gather(*link_tasks)
            context_tasks = [
                evaluate_and_extract(user_query, unique_links[link], pages[i], report_lang)
                for i, link in enumerate(unique_links)
            ]
            link_contexts = await asyncio.gather(*context_tasks)
            useful_contexts = [c for c in link_contexts if c]
            if useful_contexts:
                aggregated_contexts.extend(useful_contexts)
                progress.append(f"Context util extras: {len(useful_contexts)} surse.")
            else:
                progress.append("Niciun context util găsit.")
            if progress_cb: progress_cb(progress)

            # Step 4: more queries or finish?
            new_search_queries = await get_new_search_queries(user_query, all_search_queries, aggregated_contexts, search_lang)
            if new_search_queries == "<done>":
                progress.append("LLM a decis să oprească (suficiente date).")
                break
            elif new_search_queries:
                progress.append("LLM a propus noi interogări.")
                all_search_queries.extend(new_search_queries)
            else:
                progress.append("LLM nu a oferit alte interogări. Stop.")
                break
            iteration += 1

        # Step 5: final report
        progress.append("Se generează raportul final...")
        if progress_cb: progress_cb(progress)
        report = await generate_final_report(user_query, aggregated_contexts, report_lang, all_links_flat)
        progress.append("Gata.")
        if progress_cb: progress_cb(progress)
    return {"progress": progress, "report": report}

# --- Formatări vizuale și PDF ---
def clean_markdown(md):
    # Elimină TOATE code fences (``` sau ```markdown sau izolate)
    return re.sub(r'(```[\w\s]*)+', '', md)

def format_report_html(markdown_text):
    markdown_text = clean_markdown(markdown_text)
    # Sparge pe secțiuni principale (Titlu, etc)
    sections = re.split(r"(?m)^#+ ", markdown_text)
    headers = re.findall(r"(?m)^#+ (.+)$", markdown_text)
    blocks = []
    for idx, sec in enumerate(sections):
        if not sec.strip():
            continue
        title = headers[idx-1] if idx > 0 and idx-1 < len(headers) else None
        if title:
            blocks.append(f"<h3 class='mt-4 mb-2'>{title}</h3>")
        blocks.append(f"<div class='report-section mb-3'>{markdown(sec.strip())}</div>")
    return "\n".join(blocks)

def generate_pdf_from_html(html_content):
    if not HTML:
        raise RuntimeError("WeasyPrint nu e instalat.")
    pdf_io = io.BytesIO()
    HTML(string=html_content).write_pdf(pdf_io)
    pdf_io.seek(0)
    return pdf_io

# --- Flask app ---
app = Flask(__name__)
TASKS = {}

FORM_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>LLM Research Web App</title>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
<style>
body {background:#f7f8fa;}
.card {margin:50px auto; max-width:700px;}
#progressBox {background:#23272b;color:#fff;padding:12px 16px;border-radius:8px;margin-top:1.5rem;display:none}
</style>
</head>
<body>
<div class="card shadow">
 <div class="card-body">
   <h2 class="card-title mb-4">LLM Research Web Assistant</h2>
   <form id="main-form" method="POST" action="/start">
     <div class="form-group">
       <label for="query">Subiect / întrebare de cercetare</label>
       <textarea name="query" id="query" class="form-control" required rows="3"></textarea>
     </div>
     <div class="form-row">
        <div class="form-group col-md-6">
            <label for="search_lang">Limba pentru căutări pe web</label>
            <select name="search_lang" id="search_lang" class="form-control">
                <option value="română">Română</option>
                <option value="engleză">Engleză</option>
                <option value="germană">Germană</option>
                <option value="franceză">Franceză</option>
                <option value="spaniolă">Spaniolă</option>
                <option value="italiană">Italiană</option>
            </select>
        </div>
        <div class="form-group col-md-6">
            <label for="report_lang">Limba raportului final</label>
            <select name="report_lang" id="report_lang" class="form-control">
                <option value="română">Română</option>
                <option value="engleză">Engleză</option>
                <option value="germană">Germană</option>
                <option value="franceză">Franceză</option>
                <option value="spaniolă">Spaniolă</option>
                <option value="italiană">Italiană</option>
            </select>
        </div>
     </div>
     <div class="form-group">
       <label for="iterations">Număr maxim de iterații (default 2)</label>
       <input type="number" name="iterations" id="iterations" class="form-control" min="1" value="2"/>
     </div>
     <button type="submit" class="btn btn-primary">Start research</button>
   </form>
   <div id="progressBox"></div>
 </div>
</div>
<script>
document.getElementById('main-form').onsubmit = function() {
  document.getElementById('progressBox').style.display = "block";
  document.getElementById('progressBox').innerHTML = "Se procesează... Așteptați...";
};
</script>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Raport de cercetare</title>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
<style>
body{background:#f7f8fa;}
.card{margin-top:40px;}
.report-section {background:#fff;border-radius:7px;box-shadow:0 0 8px #dbe3ea;padding:18px;margin-bottom:1rem;}
h2, h3 {color:#274690;}
pre {background: #eef2fb;}
</style>
</head>
<body>
<div class="container mt-5">
  <div class="card shadow"><div class="card-body">
    <h4>Întrebare:</h4>
    <p><b>{{ query }}</b></p>
    <h5>Progres:</h5>
    <ul>{% for item in progress %}<li>{{item}}</li>{% endfor %}</ul>
    <h4 class="mt-4 mb-3">Raport final</h4>
    <div id="report">
      {{ report_html|safe }}
    </div>
    <form method="get" action="/download_pdf">
        <input type="hidden" name="task_id" value="{{ task_id }}"/>
        <button class="btn btn-success mt-3">Descarcă PDF</button>
    </form>
    <a href="/" class="btn btn-secondary mt-3 ml-2">Altă întrebare</a>
  </div></div>
</div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return FORM_HTML

@app.route("/start", methods=["POST"])
def start():
    query = request.form.get("query")
    iterations = int(request.form.get("iterations", 2))
    search_lang = request.form.get("search_lang", "română")
    report_lang = request.form.get("report_lang", "română")

    # Task id unic
    task_id = str(int(time.time() * 1000000))
    TASKS[task_id] = {"status": "running", "progress": [], "result": None, "html": None, "query": query}

    def progress_cb(prog):
        TASKS[task_id]["progress"] = list(prog)

    # Async run
    def run_task():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        res = loop.run_until_complete(
            full_research_pipeline(query, iterations, search_lang, report_lang, progress_cb=progress_cb)
        )
        TASKS[task_id]["status"] = "done"
        TASKS[task_id]["result"] = res
        TASKS[task_id]["html"] = format_report_html(res["report"])

    thread = threading.Thread(target=run_task)
    thread.start()

    return render_template_string("""
    <!DOCTYPE html>
    <html><head><title>Se procesează...</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>#progressBox{background:#23272b;color:#fff;padding:12px 16px;border-radius:8px;margin-top:1.5rem;}</style>
    <script>
    let interval = setInterval(async function(){
        let r = await fetch('/progress?task_id={{task_id}}');
        let data = await r.json();
        if(data.status === "done"){
            clearInterval(interval);
            window.location = "/result?task_id={{task_id}}";
        } else {
            document.getElementById('progressBox').innerHTML =
                '<b>Progres:</b><ul>' + data.progress.map(x=>'<li>'+x+'</li>').join('') + '</ul>';
        }
    }, 2000);
    </script>
    </head>
    <body>
    <div class="container mt-5">
      <div class="card shadow">
        <div class="card-body">
            <h3>Se procesează interogarea...</h3>
            <div id="progressBox">Așteptați...</div>
        </div>
      </div>
    </div>
    </body>
    </html>
    """, task_id=task_id)

@app.route("/progress")
def progress():
    task_id = request.args.get("task_id")
    t = TASKS.get(task_id, {})
    return {"status": t.get("status", "unknown"), "progress": t.get("progress", [])}

@app.route("/result")
def result():
    task_id = request.args.get("task_id")
    t = TASKS.get(task_id, {})
    html = t.get("html") or "<i>Nu există raport.</i>"
    res = t.get("result", {})
    query = t.get("query", "-")
    progress = res.get("progress", [])
    return render_template_string(RESULT_HTML, report_html=html, task_id=task_id, progress=progress, query=query)

@app.route("/download_pdf")
def download_pdf():
    task_id = request.args.get("task_id")
    t = TASKS.get(task_id, {})
    html = t.get("html", "")
    if not html or not HTML:
        return "Eroare PDF: Raport indisponibil sau lipsă weasyprint.", 400
    pdf_io = generate_pdf_from_html(html)
    return send_file(pdf_io, as_attachment=True, download_name="raport_llm.pdf", mimetype="application/pdf")

# --- Launch via ngrok (optional) ---
def run_app():
    if NGROK_TOKEN:
        ngrok.set_auth_token(NGROK_TOKEN)
        url = ngrok.connect(PORT, "http", hostname=NGROK_HOSTNAME)
        print("Public URL:", url)
    app.run(host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    run_app()
