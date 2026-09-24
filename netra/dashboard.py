"""Local companion dashboard (http://127.0.0.1:8765): live transcript, the image Netra saw,
the spoken answer, and per-stage latency with the accelerator that ran it.

Useful for sighted helpers and demos, and screen-reader friendly: answers land in an
ARIA live region, so NVDA / Narrator users can also type questions here.
Stdlib only (http.server + Server-Sent Events).
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from netra.events import bus
from netra.router import Intent

_clients: list[queue.Queue[str]] = []
_clients_lock = threading.Lock()
_state: dict[str, Any] = {"engines": {}, "image": b""}


def _on_event(kind: str, data: dict[str, Any]) -> None:
    if kind == "image":
        _state["image"] = data["jpeg"]
        data = {}
    if kind == "engines":
        _state["engines"] = data
    msg = f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    with _clients_lock:
        for q in _clients:
            q.put(msg)


def serve(assistant, port: int) -> ThreadingHTTPServer:
    _state["engines"] = dict(assistant.engines)
    bus.subscribe(_on_event)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the console for the assistant's own log
            pass

        def do_GET(self):
            if self.path == "/":
                self._send(200, "text/html; charset=utf-8", PAGE.encode())
            elif self.path.startswith("/image.jpg"):
                self._send(200, "image/jpeg", _state["image"] or b"")
            elif self.path == "/engines":
                self._send(200, "application/json", json.dumps(_state["engines"]).encode())
            elif self.path == "/events":
                self._stream()
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/ask" and body.get("text"):
                assistant.ask_text(body["text"])
            elif self.path == "/command":
                cmd = body.get("command", "")
                if cmd == "talk":
                    assistant.talk()
                elif cmd == "stop":
                    assistant.interrupt()
                elif cmd in Intent._value2member_map_:
                    assistant.run_command(Intent(cmd))
            self._send(204, "text/plain", b"")

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            q: queue.Queue[str] = queue.Queue()
            with _clients_lock:
                _clients.append(q)
            try:
                self.wfile.write(f"event: engines\ndata: {json.dumps(_state['engines'])}\n\n".encode())
                while True:
                    try:
                        msg = q.get(timeout=15)
                    except queue.Empty:
                        msg = ": keep-alive\n\n"
                    self.wfile.write(msg.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with _clients_lock:
                    _clients.remove(q)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="dashboard").start()
    return server


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Netra</title>
<style>
:root{--bg:#0b0d10;--panel:#15191f;--ink:#f4f6f8;--mute:#9aa4b2;--accent:#ffcc33;--ok:#4ade80;--npu:#60a5fa;--err:#f87171;--line:#2a313b}
@media (prefers-color-scheme:light){:root{--bg:#f7f7f5;--panel:#fff;--ink:#101418;--mute:#556070;--accent:#8a5a00;--ok:#15803d;--npu:#1d4ed8;--err:#b91c1c;--line:#dde1e6}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:18px/1.5 system-ui,Segoe UI,sans-serif}
header{display:flex;align-items:center;gap:14px;padding:16px 24px;border-bottom:1px solid var(--line)}
h1{font-size:26px;margin:0;letter-spacing:.5px}h1 span{color:var(--accent)}
#state{margin-left:auto;padding:6px 14px;border-radius:999px;border:2px solid var(--line);font-weight:700;text-transform:uppercase;font-size:14px}
#state.listening{border-color:var(--ok);color:var(--ok)}#state.thinking,#state.speaking{border-color:var(--npu);color:var(--npu)}
main{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:20px;padding:20px 24px}
@media (max-width:900px){main{grid-template-columns:1fr}}
section{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px}
h2{font-size:14px;text-transform:uppercase;letter-spacing:1px;color:var(--mute);margin:0 0 10px}
#you{font-size:22px;min-height:1.5em}#answer{font-size:24px;line-height:1.55;min-height:4em;white-space:pre-wrap}
img{width:100%;border-radius:10px;border:1px solid var(--line);background:#000;min-height:120px}
table{width:100%;border-collapse:collapse;font-size:16px}td{padding:6px 4px;border-bottom:1px solid var(--line)}
td.ms{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums;font-weight:700}td.eng{color:var(--mute)}
.npu{color:var(--npu);font-weight:700}
form{display:flex;gap:10px;margin-top:14px}input{flex:1;font:inherit;padding:12px;border-radius:10px;border:2px solid var(--line);background:var(--bg);color:var(--ink)}
button{font:inherit;font-weight:700;padding:12px 16px;border-radius:10px;border:2px solid var(--accent);background:transparent;color:var(--ink);cursor:pointer}
button:focus-visible,input:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
.row{display:flex;flex-wrap:wrap;gap:10px}#err{color:var(--err);min-height:1.2em}#ttfs{font-size:32px;font-weight:800;color:var(--accent)}
dl{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;margin:0;font-size:15px}dt{color:var(--mute)}
</style></head><body>
<header><h1>Netra <span>नेत्र</span></h1><div id="state" role="status">idle</div></header>
<main>
<div style="display:grid;gap:20px">
<section aria-labelledby="h-you"><h2 id="h-you">You said</h2><div id="you">Press Ctrl+Alt+Space and ask about your screen.</div></section>
<section aria-labelledby="h-ans"><h2 id="h-ans">Netra says</h2><div id="answer" aria-live="polite"></div><div id="err" role="alert"></div>
<form id="f"><label for="q" style="position:absolute;left:-9999px">Type a question</label><input id="q" placeholder="Type a question, e.g. what does this error say?" autocomplete="off"><button>Ask</button></form>
<div class="row" style="margin-top:12px">
<button data-c="talk">🎙 Talk</button><button data-c="describe_screen">Describe screen</button><button data-c="read_screen">Read screen</button><button data-c="pointer">Under mouse</button><button data-c="camera">Camera</button><button data-c="stop">Stop</button></div>
</section>
<section aria-labelledby="h-img"><h2 id="h-img">What Netra saw</h2><img id="img" alt="Last image Netra looked at"></section>
</div>
<div style="display:grid;gap:20px;align-content:start">
<section><h2>Time to first speech</h2><div id="ttfs">–</div></section>
<section aria-labelledby="h-st"><h2 id="h-st">Pipeline (last request)</h2><table id="stages"><tbody></tbody></table></section>
<section aria-labelledby="h-en"><h2 id="h-en">On-device engines</h2><dl id="engines"></dl></section>
</div></main>
<script>
const $=s=>document.querySelector(s);const rows={};
function engines(e){const dl=$('#engines');dl.innerHTML='';for(const[k,v]of Object.entries(e)){const dt=document.createElement('dt');dt.textContent=k.toUpperCase();const dd=document.createElement('dd');dd.textContent=v;if(/NPU/.test(v))dd.className='npu';dl.append(dt,dd)}}
const es=new EventSource('/events');
es.addEventListener('engines',m=>engines(JSON.parse(m.data)));
es.addEventListener('status',m=>{const s=JSON.parse(m.data).state;const el=$('#state');el.textContent=s;el.className=s});
es.addEventListener('transcript',m=>{$('#you').textContent=JSON.parse(m.data).text});
es.addEventListener('intent',m=>{const d=JSON.parse(m.data);if(d.text)$('#you').textContent=d.text;$('#err').textContent='';$('#stages tbody').innerHTML='';for(const k in rows)delete rows[k]});
es.addEventListener('answer',m=>{$('#answer').textContent=JSON.parse(m.data).text});
es.addEventListener('image',()=>{$('#img').src='/image.jpg?'+Date.now()});
es.addEventListener('error',m=>{if(m.data)$('#err').textContent=JSON.parse(m.data).message});
es.addEventListener('metric',m=>{const d=JSON.parse(m.data);if(d.name==='time_to_first_speech_ms')$('#ttfs').textContent=d.value+' ms'});
es.addEventListener('stage',m=>{const d=JSON.parse(m.data);let r=rows[d.name];if(!r){r=rows[d.name]=document.createElement('tr');$('#stages tbody').append(r)}
r.innerHTML='';const a=document.createElement('td');a.textContent=d.name;const b=document.createElement('td');b.className='eng'+(/NPU/.test(d.engine)?' npu':'');b.textContent=d.engine;const c=document.createElement('td');c.className='ms';c.textContent=d.ms+' ms';r.append(a,b,c)});
$('#f').addEventListener('submit',e=>{e.preventDefault();const t=$('#q').value.trim();if(!t)return;fetch('/ask',{method:'POST',body:JSON.stringify({text:t})});$('#q').value=''});
document.querySelectorAll('[data-c]').forEach(b=>b.addEventListener('click',()=>fetch('/command',{method:'POST',body:JSON.stringify({command:b.dataset.c})})));
fetch('/engines').then(r=>r.json()).then(engines);
</script></body></html>"""
