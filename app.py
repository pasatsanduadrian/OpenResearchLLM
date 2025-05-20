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

gemini_async_limiter = AsyncRateLimiter(4.0)    # 15 RPM = 4s
mistral_async_limiter = AsyncRateLimiter(1.0)   # 1s pentru siguranță

# --- Init LLM clients ---
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

def combine_messages(messages):
    return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])

# --- LLM Fallback, cu rate limit ---
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

async def generate_final_report(user_query, all_contexts, context_sources):
    # Integrează referințe în raport cu [n] și listă la final
    context_refs = {}
    bib_lines = []
    for idx, src in enumerate(context_sources):
        ref_num = idx+1
        context_refs[src] = ref_num
        bib_lines.append(f"[{ref_num}] {src}")
    # Adaugă prompt explicit pentru citări și referințe
    prompt = (
        "You are an expert researcher and report writer. Based on the gathered contexts below and the original query, "
        "write a comprehensive, well-structured, and detailed report that addresses the query thoroughly. "
        "Cite sources in-line as [n] where relevant (see context source URLs below). At the end, include a section 'Bibliografie' as a markdown bullet list with one line per reference (in format [n] url)."
        "Use markdown for all sections: Title, Executive Summary, numbered sections, bullet lists, bold, tables if needed. "
        "Each section (title, executive summary, introduction, main sections, conclusion) must be in its own markdown block for easy HTML parsing. "
        "Return only markdown, no code fences or explanations."
        "\n\nContext sources:\n" + "\n".join(bib_lines)
    )
    messages = [
        {"role": "system", "content": "You are a skilled report writer."},
        {"role": "user", "content": f"User Query: {user_query}\n\nContexts:\n{chr(10).join(all_contexts)}\n\n{prompt}"}
    ]
    return await call_llm(messages)

# --- Async research orchestrator ---
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
            if progress_cb: progress_cb(progress, [], [])
            return {"progress": progress, "report": "No report generated."}
        all_search_queries.extend(new_search_queries)

        while iteration < iter_limit:
            progress.append(f"Iteration {iteration+1}: searching and scraping...")
            if progress_cb: progress_cb(progress, aggregated_contexts[-3:], context_sources[-3:])

            # Search each query
            search_tasks = [perform_search(q) for q in new_search_queries]
            search_results = await asyncio.gather(*search_tasks)
            unique_links = {}
            for idx, links in enumerate(search_results):
                q = new_search_queries[idx]
                for link in links:
                    if link and link not in unique_links:
                        unique_links[link] = q

            # Fetch & extract
            link_tasks = [
                fetch_text(session, link) for link in unique_links
            ]
            pages = await asyncio.gather(*link_tasks)
            context_tasks = [
                evaluate_and_extract(user_query, unique_links[link], pages[i])
                for i, link in enumerate(unique_links)
            ]
            link_contexts = await asyncio.gather(*context_tasks)
            useful_contexts = []
            useful_sources = []
            for i, c in enumerate(link_contexts):
                if c:
                    aggregated_contexts.append(c)
                    context_sources.append(list(unique_links.keys())[i])
                    useful_contexts.append(c)
                    useful_sources.append(list(unique_links.keys())[i])
            if useful_contexts:
                progress.append(f"Found {len(useful_contexts)} useful contexts this iteration.")
            else:
                progress.append("No useful contexts found.")
            if progress_cb: progress_cb(progress, aggregated_contexts[-3:], context_sources[-3:])

            # More queries or finish?
            new_search_queries = await get_new_search_queries(user_query, all_search_queries, aggregated_contexts)
            if new_search_queries == "<done>":
                progress.append("LLM decided to stop searching (enough context).")
                if progress_cb: progress_cb(progress, aggregated_contexts[-3:], context_sources[-3:])
                break
            elif new_search_queries:
                progress.append("LLM provided new search queries.")
                all_search_queries.extend(new_search_queries)
            else:
                progress.append("LLM did not provide further queries. Stopping.")
                if progress_cb: progress_cb(progress, aggregated_contexts[-3:], context_sources[-3:])
                break
            iteration += 1

        # Final report
        progress.append("Generating final report via LLM.")
        if progress_cb: progress_cb(progress, aggregated_contexts[-3:], context_sources[-3:])
        report = await generate_final_report(user_query, aggregated_contexts, context_sources)
        progress.append("Done.")
        if progress_cb: progress_cb(progress, aggregated_contexts[-3:], context_sources[-3:])
    return {"progress": progress, "report": report}

# --- HTML vizual pentru raport (format markdown split pe secțiuni, cu bibliografie curățată) ---
def format_report_html(markdown_text):
    import re
    from markdown import markdown

    # Extrage secțiunea "Bibliografie" (sau similar)
    bib_section = ""
    bib_block = re.search(r"(#+\s*Bibliografie\s*[\r\n]+)([\s\S]+)$", markdown_text, re.IGNORECASE)
    if bib_block:
        bib_section = bib_block.group(2)
        markdown_text = markdown_text[:bib_block.start()]
        # Extragem referințe gen [1] url sau linii cu linkuri
        bib_lines = re.findall(r"\[([0-9]+)\]\s*([^\s\[\]]+)", bib_section)
        bib_html = "<ul class='biblio-list'>"
        for n, url in bib_lines:
            bib_html += f"<li><b>[{n}]</b> <a href='{url}' target='_blank'>{url}</a></li>"
        bib_html += "</ul>"
    else:
        bib_html = ""

    # Curățare markdown redundanțe și fence
    markdown_text = re.sub(r'(```markdown[\r\n]*)+', '', markdown_text)
    markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
    markdown_text = re.sub(r'(\n#+ .+?)(\n#+ .+?)', r'\1', markdown_text)  # elimină titlurile consecutive duplicate

    # Spargem pe secțiuni (Titlu, Executive Summary etc.)
    section_blocks = re.split(r'(?m)^#+ (.+)$', markdown_text)
    html_parts = []
    for i in range(1, len(section_blocks), 2):
        header = section_blocks[i]
        body = section_blocks[i+1]
        if header.lower().startswith('bibliografie'):
            continue  # O tratăm separat
        html_parts.append(f"<h3 class='mt-4 mb-2'>{header.strip()}</h3>")
        html_parts.append(f"<div class='report-section mb-3'>{markdown(body.strip(), extensions=['tables', 'fenced_code', 'footnotes', 'nl2br'])}</div>")

    if bib_html:
        html_parts.append("<h3 class='mt-4 mb-2'>Bibliografie</h3>")
        html_parts.append(f"<div class='report-section mb-3'>{bib_html}</div>")

    return "\n".join(html_parts)

# --- Flask Web UI ---

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
.biblio-list li {margin-bottom:5px;}
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

from functools import partial

@app.route("/start", methods=["POST"])
def start():
    user_query = request.form.get("query", "").strip()
    iter_limit = int(request.form.get("iterations", 2))
    task_id = str(hash(user_query + str(iter_limit)))
    TASKS[task_id] = {"progress": [], "done": False, "result": "", "query": user_query, "contexts": [], "sources": []}
    def progress_cb(proglist, last_ctx=None, last_src=None):
        TASKS[task_id]["progress"] = proglist.copy()
        if last_ctx: TASKS[task_id]["contexts"] = last_ctx
        if last_src: TASKS[task_id]["sources"] = last_src
    def background_run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(full_research_pipeline(user_query, iter_limit, progress_cb))
        TASKS[task_id]["done"] = True
        TASKS[task_id]["result"] = result
    threading.Thread(target=background_run, daemon=True).start()
    return render_template_string("""
      <html>
      <head>
      <meta http-equiv="refresh" content="2; URL=/progress?task_id={{task_id}}">
      <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
      </head>
      <body>
      <div class='container mt-4'>
        <h4>Started research…</h4>
        <p><a href="/progress?task_id={{task_id}}">Click here if not redirected</a></p>
      </div>
      </body></html>
    """, task_id=task_id)

@app.route("/progress")
def progress():
    from markupsafe import escape
    task_id = request.args.get("task_id")
    info = TASKS.get(task_id)
    if not info:
        return "Task not found."
    if not info["done"]:
        partial_ctx = []
        last_contexts = info.get("contexts", [])
        last_sources = info.get("sources", [])
        if last_contexts:
            for i, ctx in enumerate(last_contexts[:3]):
                preview = ctx[:300] + ('...' if len(ctx) > 300 else '')
                url = last_sources[i] if i < len(last_sources) else ""
                partial_ctx.append(f"<div class='preview-block'><b>Context {i+1}:</b> <span style='color:#444'>{escape(preview)}</span> <br><i style='font-size:smaller'>Sursa: <a href='{escape(url)}' target='_blank'>{escape(url[:60])}...</a></i></div>")
        return render_template_string("""
            <html>
            <head>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <style>
              .preview-block {background: #f4f8fc; border-radius: 6px; margin: 6px 0; padding: 8px;}
            </style>
            <meta http-equiv="refresh" content="2; URL=/progress?task_id={{task_id}}">
            </head>
            <body>
            <div class='container mt-4'>
              <h4>Progress:</h4>
              <ul>{% for item in progress %}<li>{{item}}</li>{% endfor %}</ul>
              <h5>Preview contexte utile extrase (max 3):</h5>
              {{ ctx_html|safe }}
              <div>(refreshing...)</div>
            </div>
            </body></html>
        """, progress=info["progress"], ctx_html="".join(partial_ctx), task_id=task_id)
    result = info["result"]
    report_html = format_report_html(result["report"])
    return render_template_string(RESULT_HTML,
        query=info.get("query", ""),
        progress=result["progress"],
        report_html=report_html)

def run_flask():
    print(f"\n --- Running Flask app at http://127.0.0.1:{PORT}/ ---")
    app.run(port=PORT, debug=False, use_reloader=False)

if __name__ == "__main__":
    if NGROK_TOKEN:
        ngrok.set_auth_token(NGROK_TOKEN)
        url = ngrok.connect(PORT, "http", hostname=NGROK_HOSTNAME if NGROK_HOSTNAME else None)
        print("Public URL:", url)
    run_flask()
