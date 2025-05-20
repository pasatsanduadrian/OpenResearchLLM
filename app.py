import os
import time
import asyncio
import threading
import nest_asyncio
from flask import Flask, render_template_string, request
from pyngrok import ngrok
from dotenv import load_dotenv
from mistralai import Mistral
from google import genai
import aiohttp
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

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

# --- Async Rate Limiter cu backoff la 429 ---
class AsyncRateLimiter:
    def __init__(self, min_interval_s):
        self.min_interval_s = min_interval_s
        self._last_call = 0
        self._lock = asyncio.Lock()
    async def wait(self):
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self.min_interval_s:
                await asyncio.sleep(self.min_interval_s - elapsed)
            self._last_call = time.time()

gemini_async_limiter = AsyncRateLimiter(4.0)
mistral_async_limiter = AsyncRateLimiter(1.5)

# --- Init LLM clients ---
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

def combine_messages(messages):
    return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])

# --- LLM fallback, cu retry exponential la 429 ---
async def call_llm(messages, max_retries=4, initial_wait=6):
    prompt = combine_messages(messages)
    # Gemini
    for attempt in range(max_retries):
        try:
            await gemini_async_limiter.wait()
            resp = await asyncio.to_thread(
                gemini_client.models.generate_content,
                model=GEMINI_MODEL,
                contents=prompt
            )
            reply = getattr(resp, "text", None)
            if reply and reply.strip():
                return reply.strip()
            raise Exception("Empty Gemini reply")
        except Exception as e:
            error_str = str(e)
            if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                print(f"[Gemini 429] Wait & retry ({attempt+1}) ...")
                await asyncio.sleep(initial_wait * (attempt + 1)) # backoff
            else:
                print("[Fallback] Gemini failed, try Mistral:", e)
                break  # Alt tip de eroare: nu încerca din nou
    # Mistral fallback
    for attempt in range(max_retries):
        try:
            await mistral_async_limiter.wait()
            mistral_resp = await asyncio.to_thread(
                mistral_client.chat.complete,
                model=MISTRAL_MODEL,
                messages=messages
            )
            reply = mistral_resp.choices[0].message.content.strip()
            if reply:
                return reply
            raise Exception("Empty Mistral reply")
        except Exception as ex:
            error_str = str(ex)
            if "429" in error_str or "rate limit" in error_str.lower():
                print(f"[Mistral 429] Wait & retry ({attempt+1}) ...")
                await asyncio.sleep(initial_wait * (attempt + 1))
            else:
                print("[Error] Both LLMs failed:", ex)
                break
    return "Error: Both LLMs failed."

# --- Async research pipeline ---
async def generate_search_queries(user_query):
    prompt = (
        "You are an expert research assistant. Given the user's query, generate up to four distinct, "
        "precise search queries that would help gather comprehensive information on the topic. "
        "Return only a Python list of strings, for example: ['query1', 'query2']."
    )
    messages = [
        {"role": "system", "content": "You are a helpful and precise research assistant."},
        {"role": "user", "content": f"User Query: {user_query}\n\n{prompt}"}
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

async def evaluate_and_extract(user_query, search_query, page_text):
    prompt = (
        "You are an expert evaluator. Given the user query and webpage content below, first determine if "
        "the webpage is useful. If it is useful, extract and return all relevant context as plain text. "
        "If not useful, reply exactly: Not Useful."
    )
    messages = [
        {"role": "system", "content": "You are a concise and expert evaluator and extractor."},
        {"role": "user", "content": f"User Query: {user_query}\nSearch Query: {search_query}\n\nWebpage Content (first 20000 chars):\n{page_text[:20000]}\n\n{prompt}"}
    ]
    resp = await call_llm(messages)
    cleaned = resp.strip().replace("```", "")
    if cleaned.lower() == "not useful":
        return None
    return cleaned

async def get_new_search_queries(user_query, previous_search_queries, all_contexts):
    context_combined = "\n".join(all_contexts)
    prompt = (
        "You are an analytical research assistant. Based on the original query, the search queries performed so far, "
        "and the extracted contexts from webpages, determine if further research is needed. "
        "If further research is needed, provide up to four new search queries as a Python list (e.g. ['q1','q2']). "
        "If no further research is needed, respond with exactly <done>."
    )
    messages = [
        {"role": "system", "content": "You are a systematic research planner."},
        {"role": "user", "content": f"User Query: {user_query}\nPrevious Search Queries: {previous_search_queries}\n\nExtracted Contexts:\n{context_combined}\n\n{prompt}"}
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

# --- Generare raport cu referințe și bibliografie ---
async def generate_final_report(user_query, all_contexts, sources):
    contexts_for_llm = []
    bib_entries = []
    for idx, (context, url) in enumerate(zip(all_contexts, sources)):
        ref = f"[{idx+1}]"
        contexts_for_llm.append(f"{ref} {context}")
        bib_entries.append(f"{ref} {url}")
    bib_section = "\n".join(bib_entries)
    context_combined = "\n\n".join(contexts_for_llm)
    prompt = (
        "You are an expert researcher and report writer. Based on the gathered contexts below and the original query, "
        "write a comprehensive, well-structured, and detailed report that addresses the query thoroughly. "
        "In your answer, use [N] to cite each piece of information based on its source context, where N is the number of the reference. "
        "At the end, add a separate section titled 'Bibliografie' listing all sources in the format [N] url. "
        "Use markdown for all sections: Title, Executive Summary, numbered sections, bullet lists, bold, tables if needed. "
        "Each section (title, executive summary, introduction, main sections, conclusion, bibliografie) must be in its own markdown block for easy HTML parsing. "
        "Return only markdown, no code fences or explanations.\n\n"
        f"Contexts:\n{context_combined}\n\nReferences:\n{bib_section}\n"
    )
    messages = [
        {"role": "system", "content": "You are a skilled report writer."},
        {"role": "user", "content": f"User Query: {user_query}\n\n{prompt}"}
    ]
    return await call_llm(messages)

# --- Async research orchestrator cu referințe ---
async def full_research_pipeline(user_query, iter_limit, progress_cb=None):
    aggregated_contexts = []
    context_sources = []
    all_search_queries = []
    iteration = 0
    progress = []

    async with aiohttp.ClientSession() as session:
        new_search_queries = await generate_search_queries(user_query)
        if not new_search_queries:
            progress.append("LLM did not return initial search queries.")
            return {"progress": progress, "report": "No report generated."}
        all_search_queries.extend(new_search_queries)

        while iteration < iter_limit:
            progress.append(f"Iteration {iteration+1}: searching and scraping...")
            if progress_cb: progress_cb(progress)
            search_tasks = [perform_search(q) for q in new_search_queries]
            search_results = await asyncio.gather(*search_tasks)
            unique_links = {}
            for idx, links in enumerate(search_results):
                q = new_search_queries[idx]
                for link in links:
                    if link and link not in unique_links:
                        unique_links[link] = q
            link_tasks = [
                fetch_text(session, link) for link in unique_links
            ]
            pages = await asyncio.gather(*link_tasks)
            context_tasks = [
                evaluate_and_extract(user_query, unique_links[link], pages[i])
                for i, link in enumerate(unique_links)
            ]
            link_contexts = await asyncio.gather(*context_tasks)
            for i, c in enumerate(link_contexts):
                if c:
                    aggregated_contexts.append(c)
                    context_sources.append(list(unique_links.keys())[i])
            if aggregated_contexts:
                progress.append(f"Found {len(aggregated_contexts)} useful contexts this iteration.")
            else:
                progress.append("No useful contexts found.")
            if progress_cb: progress_cb(progress)
            new_search_queries = await get_new_search_queries(user_query, all_search_queries, aggregated_contexts)
            if new_search_queries == "<done>":
                progress.append("LLM decided to stop searching (enough context).")
                break
            elif new_search_queries:
                progress.append("LLM provided new search queries.")
                all_search_queries.extend(new_search_queries)
            else:
                progress.append("LLM did not provide further queries. Stopping.")
                break
            iteration += 1
        progress.append("Generating final report via LLM.")
        if progress_cb: progress_cb(progress)
        report = await generate_final_report(user_query, aggregated_contexts, context_sources)
        progress.append("Done.")
        if progress_cb: progress_cb(progress)
    return {"progress": progress, "report": report}

# --- HTML vizual pentru raport (format markdown split pe secțiuni) ---
def format_report_html(markdown_text):
    import re
    from markdown import markdown
    # Înlătură duplicate block-level (ex: titluri duble de la LLM)
    # Sparge pe secțiuni principale (titlu, summary, etc)
    sections = re.split(r"(?m)^#+ ", markdown_text)
    headers = re.findall(r"(?m)^#+ (.+)$", markdown_text)
    blocks = []
    titles_seen = set()
    for idx, sec in enumerate(sections):
        if not sec.strip():
            continue
        title = headers[idx-1] if idx > 0 and idx-1 < len(headers) else None
        # Elimină duplicări de titlu/secțiuni
        if title and title.lower() in titles_seen:
            continue
        if title:
            blocks.append(f"<h3 class='mt-4 mb-2'>{title}</h3>")
            titles_seen.add(title.lower())
        blocks.append(f"<div class='report-section mb-3'>{markdown(sec.strip())}</div>")
    return "\n".join(blocks)

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
   <h2 class="card-title mb-4">Research Web Assistant (Gemini & Mistral)</h2>
   <form id="main-form" method="POST" action="/start">
     <div class="form-group">
       <label for="query">Your research query / topic</label>
       <textarea name="query" id="query" class="form-control" required rows="3"></textarea>
     </div>
     <div class="form-group">
       <label for="iterations">Max iterations (default 2)</label>
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
  document.getElementById('progressBox').innerHTML = "Starting...";
};
</script>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Research Report</title>
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
    <h4>Query:</h4>
    <p><b>{{ query }}</b></p>
    <h5>Progress:</h5>
    <ul>{% for item in progress %}<li>{{item}}</li>{% endfor %}</ul>
    <h4 class="mt-4 mb-3">Research Report</h4>
    <div id="report">
      {{ report_html|safe }}
    </div>
    <a href="/" class="btn btn-link mt-3">New research</a>
  </div></div>
</div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(FORM_HTML)

@app.route("/start", methods=["POST"])
def start():
    user_query = request.form.get("query", "").strip()
    iter_limit = int(request.form.get("iterations", 2))
    task_id = str(hash(user_query + str(iter_limit)))
    TASKS[task_id] = {"progress": [], "done": False, "result": "", "query": user_query}
    def progress_cb(proglist):
        TASKS[task_id]["progress"] = proglist.copy()
    def background_run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(full_research_pipeline(user_query, iter_limit, progress_cb))
        TASKS[task_id]["done"] = True
        TASKS[task_id]["result"] = result
    threading.Thread(target=background_run, daemon=True).start()
    return render_template_string("""
      <html><head>
      <meta http-equiv="refresh" content="1; URL=/progress?task_id={{task_id}}">
      </head>
      <body>
      <div style='margin:2em'>Task started. Loading progress page...</div>
      </body></html>
      """, task_id=task_id)

@app.route("/progress")
def progress():
    task_id = request.args.get("task_id")
    info = TASKS.get(task_id)
    if not info:
        return "Task not found."
    if not info["done"]:
        return render_template_string("""
            <html><head>
            <meta http-equiv="refresh" content="2; URL=/progress?task_id={{task_id}}">
            </head>
            <body>
            <h4>Progress:</h4>
            <ul>{% for item in progress %}<li>{{item}}</li>{% endfor %}</ul>
            <div>(refreshing...)</div>
            </body></html>
        """, progress=info["progress"], task_id=task_id)
    result = info["result"]
    report_html = format_report_html(result["report"])
    return render_template_string(RESULT_HTML,
        query=info.get("query", ""),
        progress=result["progress"],
        report_html=report_html)

def start_ngrok(app, port):
    if NGROK_TOKEN:
        ngrok.set_auth_token(NGROK_TOKEN)
        try:
            url = ngrok.connect(port, hostname=NGROK_HOSTNAME) if NGROK_HOSTNAME else ngrok.connect(port)
            print(f"Public URL: {url.public_url}")
        except Exception as e:
            print("ngrok error, fallback:", e)
            url = ngrok.connect(port)
            print(f"Public URL: {url.public_url}")
    else:
        print("No NGROK_TOKEN, local only.")
    app.run(port=port, debug=False)

if __name__ == "__main__":
    start_ngrok(app, PORT)
