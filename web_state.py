#!/usr/bin/env python3
"""
web_state.py
Tiny Flask dashboard that exposes lr_state.json.
Runs in its own process so the bot is not blocked.
"""

import json
import os
from pathlib import Path
from flask import Flask, render_template_string

STATE_FILE = Path("lr_state.json")
PORT = int(os.getenv("PORT", 8080))          # Railway injects PORT

app = Flask(__name__)

TMPL = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Kraken-Futures LR Bot – state</title>
  <style>
    body{font-family:system-ui, sans-serif; margin:2rem; background:#111; color:#eee;}
    h1{color:#0f0;}
    pre{background:#222; padding:1rem; border-radius:6px; overflow-x:auto;}
  </style>
</head>
<body>
  <h1>lr_live.py state</h1>
  <pre>{{state}}</pre>
  <p>Last updated: {{updated}}</p>
</body>
</html>
"""

@app.route("/")
def index():
    if STATE_FILE.exists():
        st = json.loads(STATE_FILE.read_text())
        pretty = json.dumps(st, indent=2, sort_keys=True)
    else:
        pretty = "{}"
    return render_template_string(TMPL, state=pretty, updated=pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))

def run_web():
    # Suppress Flask default logging unless you want it
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=PORT, threaded=True)

if __name__ == "__main__":
    run_web()
