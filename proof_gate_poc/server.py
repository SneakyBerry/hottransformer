"""
HTTP server for the proof-gated transformer.

Endpoints:
  POST /prove     — search for a proof
  POST /check     — check a term against a type (Agda gate)
  GET  /health    — health check
  GET  /          — web UI

Usage:
  python -m proof_gate_poc.server
  python -m proof_gate_poc.server --port 8080 --checkpoint proof_gate_poc/checkpoints/model_final.pt
"""
from __future__ import annotations
import json
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

import torch

from .ir import Atom, Arrow, Prod, Sum, Type
from .type_checker import gate as type_check_gate
from .model import ProofGateTransformer
from .train import load_model
from .tokenizer import encode_problem, tokens_to_term, token_name, BOS, EOS, PAD
from .repl import parse_input, render_proof

# Global model reference
MODEL = None
CONFIG = None
DEVICE = None

HTML_UI = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Proof-Gated Transformer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'SF Mono', 'Fira Code', monospace; background: #0a0a0a; color: #e0e0e0; padding: 40px; max-width: 900px; margin: 0 auto; }
  h1 { color: #fff; margin-bottom: 8px; font-size: 24px; }
  .subtitle { color: #888; margin-bottom: 32px; font-size: 14px; }

  .tabs { display: flex; gap: 0; margin-bottom: 0; }
  .tab { padding: 10px 20px; background: #1a1a1a; border: 1px solid #333; border-bottom: none;
         color: #888; cursor: pointer; font-family: inherit; font-size: 14px; border-radius: 6px 6px 0 0; }
  .tab.active { background: #111; color: #fff; border-color: #444; }

  .panel { display: none; padding: 20px; background: #111; border: 1px solid #444; border-radius: 0 6px 6px 6px; margin-bottom: 16px; }
  .panel.active { display: block; }

  .input-row { display: flex; gap: 12px; margin-bottom: 12px; }
  input[type=text] { flex: 1; padding: 12px 16px; background: #1a1a1a; border: 1px solid #333; color: #fff;
                     font-family: inherit; font-size: 16px; border-radius: 6px; }
  input[type=text]:focus { outline: none; border-color: #4a9eff; }
  textarea { width: 100%; padding: 12px 16px; background: #1a1a1a; border: 1px solid #333; color: #fff;
             font-family: inherit; font-size: 14px; border-radius: 6px; resize: vertical; min-height: 160px; line-height: 1.5; }
  textarea:focus { outline: none; border-color: #4a9eff; }
  button { padding: 10px 20px; background: #4a9eff; color: #fff; border: none; font-family: inherit;
           font-size: 14px; cursor: pointer; border-radius: 6px; font-weight: bold; }
  button:hover { background: #3a8eef; }
  .btn-row { display: flex; gap: 8px; margin-top: 12px; }

  .result { margin-top: 16px; padding: 20px; background: #0d1117; border-radius: 8px; border: 1px solid #222;
            white-space: pre-wrap; line-height: 1.6; }
  .gate-open { border-color: #2ea043; }
  .gate-closed { border-color: #da3633; }
  .gate-label { font-weight: bold; font-size: 18px; margin-bottom: 12px; }
  .gate-open .gate-label { color: #2ea043; }
  .gate-closed .gate-label { color: #da3633; }
  .proof { color: #c9d1d9; }
  .formal { color: #888; margin-top: 12px; font-size: 13px; }
  .error-msg { color: #f85149; }

  .examples { margin-top: 24px; color: #666; font-size: 13px; }
  .examples code { color: #999; background: #1a1a1a; padding: 2px 6px; border-radius: 3px; cursor: pointer; }
  .examples code:hover { color: #fff; }
  .fp { margin-top: 16px; padding: 12px; background: #0d1117; border-radius: 6px; font-size: 13px; color: #888; }
  .fp b { color: #2ea043; }
</style>
</head><body>
<h1>Proof-Gated Transformer</h1>
<div class="subtitle">Type checker as architectural layer. Confabulation = 0 by construction.</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('stlc')">STLC Prover</div>
  <div class="tab" onclick="switchTab('agda')">Agda Gate</div>
</div>

<!-- STLC Panel -->
<div class="panel active" id="panel-stlc">
  <div class="input-row">
    <input type="text" id="query" placeholder="A -> B, A |- B"
      onkeydown="if(event.key==='Enter')proveHoles()">
    <button onclick="prove()">Prove (one-shot)</button>
    <button onclick="proveHoles()" style="background:#2ea043">Prove (holes)</button>
  </div>
  <div class="input-row" style="margin-top:8px">
    <label style="color:#888;font-size:13px;padding:8px 0">Tries:</label>
    <input type="text" id="tries" value="20" style="width:60px;flex:none;font-size:13px">
    <label style="color:#888;font-size:13px;padding:8px 0">Temp:</label>
    <input type="text" id="temp" value="0.5" style="width:60px;flex:none;font-size:13px">
  </div>
  <div class="examples">
    Examples (click to try):
    <code onclick="tryExample('A -> B, A |- B')">A -> B, A |- B</code>
    <code onclick="tryExample('A -> B, B -> C, A |- C')">A -> B, B -> C, A |- C</code>
    <code onclick="tryExample('(A&B), B -> C |- C')">(A&B), B -> C |- C</code>
    <code onclick="tryExample('(A|B), A -> C, B -> C |- C')">(A|B), A -> C, B -> C |- C</code>
    <code onclick="tryExample('|- A -> A')">|- A -> A</code>
    <code onclick="tryExample('A -> B, C |- B')">A -> B, C |- B</code>
  </div>
  <div id="result-stlc"></div>
</div>

<!-- Agda Panel -->
<div class="panel" id="panel-agda">
  <textarea id="agda-code" placeholder="module Check where

open import Agda.Builtin.Nat
open import Agda.Builtin.Equality

-- Write your proof here. If it type-checks, gate opens.
proof : 0 + 2 ≡ 2
proof = refl">module Check where

open import Agda.Builtin.Nat
open import Agda.Builtin.Equality

proof : 0 + 2 ≡ 2
proof = refl</textarea>
  <div class="btn-row">
    <button onclick="agdaCheck()">Type Check (Gate)</button>
    <button onclick="agdaExample('valid')" style="background:#333">Example: valid</button>
    <button onclick="agdaExample('invalid')" style="background:#333">Example: invalid</button>
    <button onclick="agdaExample('identity')" style="background:#333">Example: identity</button>
    <button onclick="agdaExample('nat')" style="background:#333">Example: nat proof</button>
  </div>
  <div id="result-agda"></div>
</div>

<div class="fp"><b>FP = 0.0%</b> — not by training. By theorem. The type checker cannot pass an invalid proof.</div>

<script>
function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('panel-' + tab).classList.add('active');
}

function tryExample(q) {
  document.getElementById('query').value = q;
  prove();
}

const agdaExamples = {
  valid: `module Check where

open import Agda.Builtin.Nat
open import Agda.Builtin.Equality

proof : 0 + 2 ≡ 2
proof = refl`,

  invalid: `module Check where

open import Agda.Builtin.Nat
open import Agda.Builtin.Equality

-- This should FAIL: 0 + 2 is not 3
proof : 0 + 2 ≡ 3
proof = refl`,

  identity: `module Check where

open import Agda.Builtin.Nat

id : Nat → Nat
id x = x

-- Apply identity
result : Nat
result = id 42`,

  nat: `module Check where

open import Agda.Builtin.Nat
open import Agda.Builtin.Equality

-- suc is injective (a basic theorem about naturals)
+-zero : (n : Nat) → n + 0 ≡ n
+-zero zero = refl
+-zero (suc n) = cong suc (+-zero n)
  where
    cong : {A B : Set} {x y : A} → (f : A → B) → x ≡ y → f x ≡ f y
    cong f refl = refl`
};

function agdaExample(name) {
  document.getElementById('agda-code').value = agdaExamples[name] || '';
}

async function prove() {
  const q = document.getElementById('query').value.trim();
  if (!q) return;
  const res = document.getElementById('result-stlc');
  res.className = 'result';
  res.innerHTML = '';

  // Use SSE for streaming proof steps
  try {
    const response = await fetch('/prove-stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: q, tries: parseInt(document.getElementById('tries')?.value||'20'), temperature: parseFloat(document.getElementById('temp')?.value||'0.5')})
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let steps = [];
    let foundProof = null;

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});

      const lines = buffer.split('\n');
      buffer = lines.pop();

      let eventType = '';
      for (const line of lines) {
        if (line.startsWith('event: ')) eventType = line.slice(7);
        else if (line.startsWith('data: ') && eventType) {
          const data = JSON.parse(line.slice(6));
          handleSSE(eventType, data, res, steps);
          if (eventType === 'proof_found') foundProof = data;
          eventType = '';
        }
      }
    }
  } catch(e) {
    res.className = 'result gate-closed';
    res.innerHTML = '<div class="gate-label">ERROR</div><div class="error-msg">' + esc(String(e)) + '</div>';
  }
}

function handleSSE(type, data, res, steps) {
  if (type === 'start') {
    res.className = 'result';
    res.innerHTML = '<div style="color:#888">Searching: ' + esc(data.premises) + ' ⊢ ' + esc(data.goal) + '</div>'
      + '<div id="progress" style="color:#555;margin-top:8px">Attempt 0/' + data.tries + '</div>';
  }
  else if (type === 'attempt') {
    const el = document.getElementById('progress');
    if (el) el.textContent = 'Attempt ' + data.n + '/' + data.total + '...';
  }
  else if (type === 'proof_found') {
    res.className = 'result gate-open';
    res.innerHTML = '<div class="gate-label">GATE OPEN ✓</div>'
      + '<div class="proof">' + esc(data.explanation) + '</div>'
      + '<div class="formal">Formal: ' + esc(data.term) + '  (' + data.length + ' tokens, attempt ' + data.attempt + ')</div>';
  }
  else if (type === 'done') {
    if (!data.gate_open) {
      res.className = 'result gate-closed';
      res.innerHTML = '<div class="gate-label">GATE CLOSED</div>'
        + '<div class="proof">No valid proof found.\nEither unprovable, or increase tries.</div>';
    }
  }
  else if (type === 'error') {
    res.className = 'result gate-closed';
    res.innerHTML = '<div class="gate-label">ERROR</div><div class="error-msg">' + esc(data.message) + '</div>';
  }
}

async function agdaCheck() {
  const code = document.getElementById('agda-code').value;
  if (!code.trim()) return;
  const res = document.getElementById('result-agda');
  res.className = 'result';
  res.innerHTML = 'Type checking with Agda...';
  try {
    const r = await fetch('/agda', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({code: code})
    });
    const data = await r.json();
    if (data.gate_open) {
      res.className = 'result gate-open';
      res.innerHTML = '<div class="gate-label">GATE OPEN ✓</div>'
        + '<div class="proof">Agda type checker accepts this code.\nEvery definition is verified.</div>';
    } else {
      res.className = 'result gate-closed';
      let msg = 'Agda type checker rejects this code.';
      if (data.error) msg += '\n\n' + data.error;
      res.innerHTML = '<div class="gate-label">GATE CLOSED ✗</div>'
        + '<div class="proof">' + esc(msg) + '</div>';
    }
  } catch(e) {
    res.className = 'result gate-closed';
    res.innerHTML = '<div class="gate-label">ERROR</div><div class="error-msg">' + esc(String(e)) + '</div>';
  }
}

async function proveHoles() {
  const q = document.getElementById('query').value.trim();
  if (!q) return;
  const res = document.getElementById('result-stlc');
  res.className = 'result';
  res.innerHTML = '<div id="holes-trace"></div>';

  try {
    const response = await fetch('/prove-holes', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: q, temperature: parseFloat(document.getElementById('temp')?.value||'0.5')})
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\n');
      buffer = lines.pop();

      let eventType = '';
      for (const line of lines) {
        if (line.startsWith('event: ')) eventType = line.slice(7);
        else if (line.startsWith('data: ') && eventType) {
          const data = JSON.parse(line.slice(6));
          handleHoleSSE(eventType, data, res);
          eventType = '';
        }
      }
    }
  } catch(e) {
    res.className = 'result gate-closed';
    res.innerHTML = '<div class="gate-label">ERROR</div><div class="error-msg">' + esc(String(e)) + '</div>';
  }
}

function handleHoleSSE(type, data, res) {
  const trace = document.getElementById('holes-trace');
  if (!trace) return;

  if (type === 'start') {
    trace.innerHTML = '<div style="color:#888;margin-bottom:12px">' + esc(data.premises) + ' ⊢ ' + esc(data.goal) + '</div>';
  }
  else if (type === 'hole') {
    const remaining = data.remaining_holes.map(h => '<span style="color:#4a9eff">?' + h.id + ':' + esc(h.goal) + '</span>').join(', ');
    trace.innerHTML += '<div style="margin:4px 0;color:#666;font-size:12px">Holes: ' + remaining + '</div>';
    trace.innerHTML += '<div style="color:#c9d1d9">Step ' + data.step + ': <span style="color:#4a9eff">?' + data.hole_id + '</span> : <b>' + esc(data.goal) + '</b>'
      + ' <span style="color:#555">(' + data.n_actions + ' options)</span></div>';
  }
  else if (type === 'fill') {
    let color = data.forced ? '#555' : '#2ea043';
    let label = data.forced ? 'forced' : 'chosen';
    trace.innerHTML += '<div style="color:' + color + ';margin-left:20px">→ ' + esc(data.action) + ' <span style="color:#444">[' + label + ']</span></div>';
    if (data.new_subgoals.length > 0) {
      const sgs = data.new_subgoals.map(g => '<span style="color:#da3633">?' + esc(g) + '</span>').join(', ');
      trace.innerHTML += '<div style="color:#888;margin-left:20px;font-size:12px">new holes: ' + sgs + '</div>';
    }
  }
  else if (type === 'stuck') {
    trace.innerHTML += '<div style="color:#da3633;margin-top:8px">Stuck at ?' + data.hole_id + ' : ' + esc(data.goal) + ' — no valid actions</div>';
    res.className = 'result gate-closed';
  }
  else if (type === 'done') {
    if (data.gate_open) {
      res.className = 'result gate-open';
      trace.innerHTML += '<hr style="border-color:#222;margin:12px 0">'
        + '<div class="gate-label">GATE OPEN ✓ (' + data.steps + ' steps)</div>'
        + '<div class="proof">' + esc(data.explanation) + '</div>'
        + '<div class="formal">Formal: ' + esc(data.term) + '</div>';
    } else {
      res.className = 'result gate-closed';
      trace.innerHTML += '<hr style="border-color:#222;margin:12px 0">'
        + '<div class="gate-label">GATE CLOSED</div>'
        + '<div style="color:#888">' + data.remaining_holes + ' holes remaining</div>';
    }
  }
}

function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>'); }
</script>
</body></html>"""


class ProofHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silent

    def _send(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, "application/json", json.dumps({"status": "ok", "model_loaded": MODEL is not None}))
        else:
            self._send(200, "text/html", HTML_UI)

    def do_POST(self):
        if self.path == "/prove":
            self._handle_prove()
        elif self.path == "/prove-stream":
            self._handle_prove_stream()
        elif self.path == "/prove-holes":
            self._handle_prove_holes()
        elif self.path == "/check":
            self._handle_check()
        elif self.path == "/agda":
            self._handle_agda()
        else:
            self._send(404, "application/json", json.dumps({"error": "not found"}))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        return json.loads(body)

    def _handle_prove(self):
        try:
            data = self._read_json()
            query = data.get("query", "")
            n_tries = data.get("tries", 10)
            temperature = data.get("temperature", 0.5)

            premises, goal = parse_input(query)

            if MODEL is None:
                self._send(503, "application/json", json.dumps({"error": "no model loaded"}))
                return

            results = _run_inference(MODEL, premises, goal, DEVICE, temperature, n_tries)

            if results:
                best = results[0]
                explanation = render_proof(best["term"], premises, goal)
                self._send(200, "application/json", json.dumps({
                    "gate_open": True,
                    "explanation": explanation,
                    "formal": str(best["term"]),
                    "proof_length": best["length"],
                    "n_proofs": len(results),
                    "premises": [str(p) for p in premises],
                    "goal": str(goal),
                }))
            else:
                self._send(200, "application/json", json.dumps({
                    "gate_open": False,
                    "message": f"No valid proof found in {n_tries} attempts.\nEither unprovable, or model needs more search.",
                    "premises": [str(p) for p in premises],
                    "goal": str(goal),
                }))

        except ValueError as e:
            self._send(400, "application/json", json.dumps({"error": f"Parse error: {str(e)}"}))
        except Exception as e:
            self._send(500, "application/json", json.dumps({"error": str(e)}))

    def _handle_check(self):
        try:
            data = self._read_json()
            term_str = data.get("term", "")
            type_str = data.get("type", "")
            context = data.get("context", "")

            from .agda_bridge import AgdaBridge
            bridge = AgdaBridge()
            valid = bridge.check_term(term_str, type_str, context=context)

            self._send(200, "application/json", json.dumps({
                "gate_open": valid,
                "term": term_str,
                "type": type_str,
            }))
        except Exception as e:
            self._send(500, "application/json", json.dumps({"error": str(e)}))

    def _handle_agda(self):
        """Type-check raw Agda code. Gate = does it compile?"""
        try:
            data = self._read_json()
            code = data.get("code", "")

            import subprocess
            import re as re_mod
            import os

            # Extract module name from code, or use default
            mod_match = re_mod.search(r'module\s+(\S+)', code)
            mod_name = mod_match.group(1) if mod_match else "WebCheck"

            # Write to project root with matching filename
            tmp_file = os.path.join(os.getcwd(), mod_name.replace(".", "/") + ".agda")
            os.makedirs(os.path.dirname(tmp_file) or ".", exist_ok=True)

            with open(tmp_file, 'w') as f:
                f.write(code)

            result = subprocess.run(
                ["agda", tmp_file],
                capture_output=True, text=True, timeout=30,
            )

            # Cleanup
            try:
                os.remove(tmp_file)
                os.remove(tmp_file.replace(".agda", ".agdai"))
            except OSError:
                pass

            if result.returncode == 0:
                self._send(200, "application/json", json.dumps({
                    "gate_open": True,
                }))
            else:
                error_msg = result.stderr.strip() or result.stdout.strip()
                # Clean up file paths in error messages
                error_msg = error_msg.replace(tmp_file, "Check.agda")
                self._send(200, "application/json", json.dumps({
                    "gate_open": False,
                    "error": error_msg,
                }))

        except subprocess.TimeoutExpired:
            self._send(200, "application/json", json.dumps({
                "gate_open": False,
                "error": "Type checking timed out (30s limit)",
            }))
        except Exception as e:
            self._send(500, "application/json", json.dumps({"error": str(e)}))

    def _handle_prove_stream(self):
        """SSE endpoint: stream proof search steps in real-time."""
        try:
            data = self._read_json()
            query = data.get("query", "")
            n_tries = data.get("tries", 20)
            temperature = data.get("temperature", 0.5)

            premises, goal = parse_input(query)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            def send_event(event_type, data_dict):
                line = f"event: {event_type}\ndata: {json.dumps(data_dict)}\n\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()

            prem_str = ", ".join(str(p) for p in premises)
            send_event("start", {"premises": prem_str, "goal": str(goal), "tries": n_tries})

            if MODEL is None:
                send_event("error", {"message": "No model loaded"})
                return

            MODEL.eval()
            ctx = tuple(premises)
            input_tokens = encode_problem(premises, goal)
            src = torch.tensor([input_tokens], dtype=torch.long, device=DEVICE)
            found = []
            seen = set()

            with torch.no_grad():
                for attempt in range(n_tries):
                    send_event("attempt", {"n": attempt + 1, "total": n_tries})

                    tokens, log_probs = MODEL.generate(
                        src, max_len=32, temperature=temperature, greedy=(attempt == 0))
                    seq = tokens[0].tolist()
                    eos_pos = len(seq)
                    for j, t in enumerate(seq):
                        if t == EOS:
                            eos_pos = j
                            break
                    seq = seq[:eos_pos]

                    term, _ = tokens_to_term(seq)
                    if term is None:
                        send_event("step", {"attempt": attempt + 1, "status": "parse_fail",
                                            "raw_tokens": [token_name(t) for t in seq]})
                        continue

                    term_str = str(term)
                    if term_str in seen:
                        send_event("step", {"attempt": attempt + 1, "status": "duplicate", "term": term_str})
                        continue
                    seen.add(term_str)

                    valid = type_check_gate(ctx, term, goal)
                    if valid:
                        explanation = render_proof(term, premises, goal)
                        send_event("proof_found", {
                            "attempt": attempt + 1,
                            "term": term_str,
                            "length": len(seq),
                            "explanation": explanation,
                        })
                        found.append({"term": term, "length": len(seq)})
                    else:
                        send_event("step", {"attempt": attempt + 1, "status": "gate_closed",
                                            "term": term_str})

            if found:
                best = found[0]
                send_event("done", {"gate_open": True, "n_proofs": len(found),
                                    "best": str(best["term"]), "best_length": best["length"]})
            else:
                send_event("done", {"gate_open": False, "message": "No valid proof found"})

        except ValueError as e:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            line = f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
            self.wfile.write(line.encode("utf-8"))
        except Exception as e:
            pass  # connection may be closed


    def _handle_prove_holes(self):
        """SSE: hole-by-hole proof construction. Streams each hole fill."""
        try:
            data = self._read_json()
            query = data.get("query", "")
            temperature = data.get("temperature", 0.5)

            premises, goal = parse_input(query)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            def send_event(event_type, data_dict):
                line = f"event: {event_type}\ndata: {json.dumps(data_dict)}\n\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()

            prem_str = ", ".join(str(p) for p in premises)
            send_event("start", {"premises": prem_str, "goal": str(goal)})

            from .guided_search import get_valid_actions, build_term
            from .hole_step import Hole, ProofState, fill_hole, reconstruct_term

            ctx = tuple(premises)
            state = ProofState(
                holes=[Hole(ctx=ctx, goal=goal, id=0)],
                tree={}, next_hole_id=1, root_hole_id=0,
            )

            for step in range(30):
                if not state.holes:
                    break

                hole = state.holes[0]
                actions = get_valid_actions(hole.ctx, hole.goal)

                holes_info = [{"id": h.id, "goal": str(h.goal)} for h in state.holes]
                send_event("hole", {
                    "step": step,
                    "hole_id": hole.id,
                    "goal": str(hole.goal),
                    "context": [f"v{i}:{hole.ctx[i]}" for i in range(len(hole.ctx))],
                    "n_actions": len(actions),
                    "actions": [a["description"] for a in actions],
                    "remaining_holes": holes_info,
                })

                if not actions:
                    send_event("stuck", {"hole_id": hole.id, "goal": str(hole.goal)})
                    break

                # Choose action: model if available, else first
                chosen = None
                if MODEL is not None and len(actions) > 1:
                    hole_input = encode_problem(list(hole.ctx), hole.goal)
                    hole_src = torch.tensor([hole_input], dtype=torch.long, device=DEVICE)
                    with torch.no_grad():
                        tokens, _ = MODEL.generate(hole_src, max_len=1,
                                                    temperature=temperature, greedy=False)
                    if tokens.numel() > 0:
                        idx = tokens[0, 0].item() % len(actions)
                        chosen = actions[idx]
                if chosen is None:
                    chosen = actions[0]

                send_event("fill", {
                    "step": step,
                    "hole_id": hole.id,
                    "action": chosen["description"],
                    "forced": len(actions) == 1,
                    "new_subgoals": [str(sg.goal) for sg in chosen["subgoals"]],
                })

                state = fill_hole(state, 0, chosen)

            term = reconstruct_term(state)
            valid = type_check_gate(ctx, term, goal) if term is not None else False

            if valid:
                explanation = render_proof(term, premises, goal)
                send_event("done", {
                    "gate_open": True,
                    "term": str(term),
                    "explanation": explanation,
                    "steps": step + 1,
                })
            else:
                send_event("done", {
                    "gate_open": False,
                    "remaining_holes": len(state.holes),
                })

        except ValueError as e:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n".encode())
        except Exception:
            pass


def _run_inference(model, premises, goal, device, temperature=0.5, n_tries=10):
    model.eval()
    ctx = tuple(premises)
    input_tokens = encode_problem(premises, goal)
    src = torch.tensor([input_tokens], dtype=torch.long, device=device)
    results = []
    seen = set()

    with torch.no_grad():
        for attempt in range(n_tries):
            tokens, log_probs = model.generate(
                src, max_len=32, temperature=temperature, greedy=(attempt == 0))
            seq = tokens[0].tolist()
            eos_pos = len(seq)
            for j, t in enumerate(seq):
                if t == EOS:
                    eos_pos = j
                    break
            seq = seq[:eos_pos]
            term, _ = tokens_to_term(seq)
            if term is None:
                continue
            term_str = str(term)
            if term_str in seen:
                continue
            seen.add(term_str)
            if type_check_gate(ctx, term, goal):
                results.append({"term": term, "tokens": seq, "length": len(seq)})

    results.sort(key=lambda r: r["length"])
    return results


def main():
    global MODEL, CONFIG, DEVICE

    parser = argparse.ArgumentParser(description="Proof-Gated Transformer HTTP Server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--checkpoint", default="proof_gate_poc/checkpoints/model_final.pt")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    try:
        MODEL, CONFIG, _, _ = load_model(args.checkpoint)
        DEVICE = CONFIG.get_device()
        MODEL.eval()
        n_params = sum(p.numel() for p in MODEL.parameters())
        print(f"Model loaded: {n_params:,} parameters on {DEVICE}")
    except FileNotFoundError:
        print(f"No checkpoint found at {args.checkpoint} — running without model")

    server = HTTPServer((args.host, args.port), ProofHandler)
    print(f"\nServer running at http://{args.host}:{args.port}")
    print(f"Share with: http://<your-ip>:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")
        server.server_close()


if __name__ == "__main__":
    main()
