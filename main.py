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
    from weasyprint import HTML
except ImportError:
    HTML = None  # Warn if missing

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

# --- Pipeline: Research, Extraction, Report ---
async def generate_search_queries(user_query, search_lang):
    prompt = (
        f"Generează până la 4 interogări precise pentru web, în limba: {search_lang}. "
        "Răspunde doar cu o listă Python de stringuri, ex: ['query1', 'query2']."
    )
    messages = [
        {"role": "system", "content": "Ești un asistent util și precis."},
        {"role": "user", "content": f"Întrebare: {user_query}\n{prompt}"}
    ]
    response = await call_llm(messages)
    clean = response.strip().replace('```python', '').replace('```', '').strip()
    try:
        queries = eval(clean)
        return queries if isinstance(queries, list) else []
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
    page_text_content = page_text[:20000] if page_text else ""
    prompt = (
        f"Pentru întrebarea '{user_query}' și pagina web de mai jos, evaluează dacă e utilă. "
        "Dacă DA, extrage contextul relevant ca text simplu, răspunzând doar în limba: {report_lang}. "
        "Dacă nu e utilă, răspunde exact: Not Useful."
    )
    messages = [
        {"role": "system", "content": "Evaluator și extractor concis."},
        {"role": "user", "content": f"Întrebare: {user_query}\nCăutare: {search_query}\nConținut pagină:\n{page_text_content}\n{prompt}"}
    ]
    resp = await call_llm(messages)
    cleaned = resp.strip().replace("```", "")
    return None if cleaned.lower() == "not useful" else cleaned

async def get_new_search_queries(user_query, previous_search_queries, all_contexts, search_lang):
    context_combined = "\n".join(all_contexts)
    prompt = (
        f"Pe baza interogărilor anterioare (în {search_lang}) și a contextelor extrase, vezi dacă mai sunt necesare alte căutări. "
        "Dacă DA, propune până la 4 noi interogări ca listă Python. Dacă NU, răspunde cu exact <done>."
    )
    messages = [
        {"role": "system", "content": "Planificator sistematic."},
        {"role": "user", "content": f"Întrebare: {user_query}\nCăutări precedente: {previous_search_queries}\nContext extras:\n{context_combined}\n{prompt}"}
    ]
    response = await call_llm(messages)
    if response.strip() == "<done>":
        return "<done>"
    clean = response.strip().replace('```python', '').replace('```', '').strip()
    try:
        queries = eval(clean)
        return queries if isinstance(queries, list) else []
    except Exception:
        return []

async def generate_final_report(user_query, all_contexts, report_lang, context_sources):
    bib_lines = [f"[{i+1}] {src}" for i, src in enumerate(context_sources)] if context_sources else []
    prompt = (
        f"Scrie un raport structurat, cu secțiuni markdown: Titlu, Rezumat executiv, introducere, secțiuni, concluzie."
        f" Redactează clar, formal, EXCLUSIV în limba: {report_lang}. "
        "Pentru fiecare sursă din lista de mai jos, inserează o citare [n] în text (măcar o dată), chiar dacă doar sumarizezi o idee sau menționezi sursa minimal."
        " Folosește formatare Markdown. La final, include secțiunea 'Bibliografie' cu lista surselor, o linie per referință, format [n] URL."
        " NU include delimitatori de tipul ``` sau code fence."
        "\n\nReferințe (folosește toate aceste surse, fiecare să apară citată o dată):\n" + "\n".join(bib_lines)
    )
    context_combined = "\n".join(all_contexts)
    messages = [
        {"role": "system", "content": "Redactor raport profesionist și precis."},
        {"role": "user", "content": f"Întrebare: {user_query}\n\nContext:\n{context_combined}\n\n{prompt}"}
    ]
    return await call_llm(messages)



async def full_research_pipeline(user_query, iter_limit, search_lang, report_lang, progress_cb=None):
    aggregated_contexts = []
    context_sources = []
    all_search_queries = []
    iteration = 0
    progress = []
    async with aiohttp.ClientSession() as session:
        new_search_queries = await generate_search_queries(user_query, search_lang)
        if not new_search_queries:
            progress.append("LLM nu a returnat interogări de căutare.")
            return {"progress": progress, "report": "Niciun raport generat."}
        all_search_queries.extend(new_search_queries)
        while iteration < iter_limit:
            progress.append(f"Iterația {iteration+1}: Căutare și extragere date...")
            if progress_cb: progress_cb(progress)
            search_tasks = [perform_search(q) for q in new_search_queries]
            search_results = await asyncio.gather(*search_tasks)
            unique_links = {}
            for idx, links in enumerate(search_results):
                q = new_search_queries[idx]
                for link in links:
                    if link and link not in unique_links:
                        unique_links[link] = q
            link_tasks = [fetch_text(session, link) for link in unique_links]
            pages = await asyncio.gather(*link_tasks)
            context_tasks = [
                evaluate_and_extract(user_query, unique_links[link], pages[i], report_lang)
                for i, link in enumerate(unique_links)
            ]
            link_contexts = await asyncio.gather(*context_tasks)
            useful_contexts_this_iter = []
            useful_sources_this_iter = []
            for i, c in enumerate(link_contexts):
                if c:
                    useful_contexts_this_iter.append(c)
                    useful_sources_this_iter.append(list(unique_links.keys())[i])
            if useful_contexts_this_iter:
                aggregated_contexts.extend(useful_contexts_this_iter)
                context_sources.extend(useful_sources_this_iter)
                progress.append(f"Context util extras: {len(useful_contexts_this_iter)} surse noi.")
            else:
                progress.append("Niciun context util găsit în această iterație.")
            if progress_cb: progress_cb(progress)
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
        progress.append("Se generează raportul final...")
        if progress_cb: progress_cb(progress)
        report = await generate_final_report(user_query, aggregated_contexts, report_lang, context_sources)
        progress.append("Gata.")
        if progress_cb: progress_cb(progress)
    return {"progress": progress, "report": report}

def format_report_html(markdown_text):
    # Remove all code fences
    markdown_text = re.sub(r'(```[\w\s]*\n)+', '', markdown_text)
    markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
    # Split into sections based on markdown headers
    sections = re.split(r"(?m)^(#{1,6}\s.*)$", markdown_text, flags=re.MULTILINE)
    html_parts = []
    current_header = None
    for i, part in enumerate(sections):
        if not part.strip():
            continue
        if re.match(r"^(#{1,6}\s.*)$", part, flags=re.MULTILINE):
            current_header = part.strip()
        else:
            if current_header:
                if current_header.startswith("# "):
                    html_parts.append(f"<h2 class='mt-4 mb-2 report-heading'>{current_header.replace('# ', '')}</h2>")
                elif current_header.startswith("## "):
                    html_parts.append(f"<h3 class='mt-3 mb-2 report-subheading'>{current_header.replace('## ', '')}</h3>")
                else:
                    html_parts.append(f"<div class='report-section-header'>{markdown(current_header.strip())}</div>")
                html_parts.append(f"<div class='report-section mb-3'>{markdown(part.strip(), extensions=['tables', 'fenced_code', 'footnotes', 'nl2br'])}</div>")
                current_header = None
            else:
                html_parts.append(f"<div class='report-section mb-3'>{markdown(part.strip(), extensions=['tables', 'fenced_code', 'footnotes', 'nl2br'])}</div>")
    return "\n".join(html_parts)

def generate_pdf_from_html(html_content, query_title="Raport LLM"):
    if not HTML:
        raise RuntimeError("WeasyPrint nu este instalat. Instalează cu pip install weasyprint.")
    pdf_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{query_title}</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: 'DejaVu Sans', Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; }}
            h1, h2, h3, h4, h5, h6 {{ color: #274690; margin-top: 1.5em; margin-bottom: 0.5em; }}
            h1 {{ font-size: 2.2em; text-align: center; margin-bottom: 1em; }}
            h2 {{ font-size: 1.8em; border-bottom: 2px solid #eee; padding-bottom: 5px; }}
            h3 {{ font-size: 1.4em; }}
            p {{ margin-bottom: 1em; }}
            ul {{ list-style-type: disc; margin-left: 20px; margin-bottom: 1em; }}
            ol {{ list-style-type: decimal; margin-left: 20px; margin-bottom: 1em; }}
            a {{ color: #007bff; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            .report-section {{ background: #f9f9f9; border-radius: 5px; padding: 15px; margin-bottom: 1em; border: 1px solid #ddd; }}
            .biblio-list {{ list-style: none; padding: 0; }}
            .biblio-list li {{ margin-bottom: 0.5em; }}
            .page-break {{ page-break-before: always; }}
        </style>
    </head>
    <body>
        <h1>{query_title}</h1>
        {html_content}
    </body>
    </html>
    """
    pdf_io = io.BytesIO()
    HTML(string=pdf_template, base_url=os.getcwd()).write_pdf(pdf_io)
    pdf_io.seek(0)
    return pdf_io

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
#progressBox {background:#23272b;color:#fff;padding:12px 16px;border-radius:8px;margin-top:1.5rem;display:none; overflow-y: auto; max-height: 200px;}
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
.container {margin-top:40px;}
.card {box-shadow:0 4px 8px rgba(0,0,0,0.1);}
.report-section {
    background:#fff;
    border-radius:7px;
    box-shadow
    :0 0 8px rgba(219,227,234,0.5); /* Lighter shadow */
    padding:20px; /* More padding */
    margin-bottom:1.5rem; /* More space between sections */
    border-left: 5px solid #274690; /* Accent border */
}
.report-heading, .report-subheading {color:#274690; font-weight: bold; margin-bottom: 0.8em;}
h1 {font-size: 2.2em; text-align: center; margin-bottom: 1em; color: #274690;}
h2 {font-size: 1.8em; border-bottom: 2px solid #eee; padding-bottom: 5px;}
h3 {font-size: 1.4em;}
pre {background: #eef2fb; padding: 10px; border-radius: 5px; overflow-x: auto;}
.biblio-list { list-style: none; padding: 0; }
.biblio-list li { margin-bottom: 0.5em; }
.biblio-list li b { color: #555; } /* Style for citation numbers */
</style>
</head>
<body>
<div class="container">
    <div class="card shadow">
        <div class="card-body">
            <h1 class="card-title text-center mb-4">Raport de Cercetare LLM</h1>
            <h4 class="mb-3">Întrebare:</h4>
            <p class="lead"><b>{{ query }}</b></p>
            <h5 class="mt-4 mb-3">Progresul cercetării:</h5>
            <ul class="list-group mb-4">
                {% for item in progress %}
                    <li class="list-group-item list-group-item-action">{{item}}</li>
                {% endfor %}
            </ul>
            <h2 class="mt-5 mb-4 text-center">Raport Final</h2>
            <div id="report-content">
                {{ report_html|safe }}
            </div>
            <hr class="my-5">
            <div class="d-flex justify-content-center">
                <form method="get" action="/download_pdf">
                    <input type="hidden" name="task_id" value="{{ task_id }}"/>
                    <button class="btn btn-success btn-lg mx-2">Descarcă PDF</button>
                </form>
                <a href="/" class="btn btn-secondary btn-lg mx-2">Cercetare nouă</a>
            </div>
        </div>
    </div>
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

    # Generate a unique task ID
    task_id = str(int(time.time() * 1000000))
    TASKS[task_id] = {"status": "running", "progress": [], "result": None, "html": None, "query": query, "search_lang": search_lang, "report_lang": report_lang}

    def progress_cb(prog):
        TASKS[task_id]["progress"] = list(prog)

    # Run the research pipeline in a separate thread
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
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css ">
    <style>#progressBox{background:#23272b;color:#fff;padding:12px 16px;border-radius:8px;margin-top:1.5rem; overflow-y: auto; max-height: 200px;}</style>
    <script>
    let interval = setInterval(async function(){
        let r = await fetch('/progress?task_id={{task_id}}');
        let data = await r.json();
        if(data.status === "done"){
            clearInterval(interval);
            window.location = "/result?task_id={{task_id}}";
        } else {
            document.getElementById('progressBox').innerHTML =
                '<b>Progres:</b><ul class="list-unstyled">' + data.progress.map(x=>'<li>'+x+'</li>').join('') + '</ul>';
        }
    }, 2000);
    </script>
    </head>
    <body>
    <div class="container mt-5">
        <div class="card shadow">
            <div class="card-body">
                <h3>Se procesează interogarea...</h3>
                <p>Acest proces poate dura câteva minute, în funcție de complexitatea întrebării și numărul de iterații.</p>
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
    progress_list = res.get("progress", [])
    search_lang = t.get("search_lang", "-")
    report_lang = t.get("report_lang", "-")
    return render_template_string(RESULT_HTML, report_html=html, task_id=task_id, progress=progress_list, query=query, search_lang=search_lang, report_lang=report_lang)

@app.route("/download_pdf")
def download_pdf():
    task_id = request.args.get("task_id")
    t = TASKS.get(task_id, {})
    html_content = t.get("html", "")
    query_title = f"Raport_{t.get('query', 'Cercetare')[:20].replace(' ', '_')}" # Simple title for PDF filename

    if not html_content or not HTML:
        return "Eroare PDF: Raport indisponibil sau librăria WeasyPrint nu este instalată. Vă rugăm să instalați WeasyPrint (pip install weasyprint).", 400

    try:
        pdf_io = generate_pdf_from_html(html_content, query_title=query_title)
        return send_file(pdf_io, as_attachment=True, download_name=f"{query_title}.pdf", mimetype="application/pdf")
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return f"Eroare la generarea PDF-ului: {e}", 500

# --- Launch via ngrok (optional) ---
def run_app():
    if NGROK_TOKEN:
        ngrok.set_auth_token(NGROK_TOKEN)
        url = ngrok.connect(PORT, "http", hostname=NGROK_HOSTNAME)
        print("Public URL:", url)
    app.run(host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    run_app()
