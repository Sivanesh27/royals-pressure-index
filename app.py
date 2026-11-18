"""Royals Pressure Index API.  Serves the dashboard and a small JSON API.
Run locally:  python app.py      Production:  gunicorn app:app"""
import json, os
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, abort

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(ROOT, "static")
app = Flask(__name__, static_folder=STATIC, static_url_path="")
with open(os.path.join(STATIC, "data.json")) as fh:
    DATA = json.load(fh)
PLAYERS = {r: {p["id"]: p for p in DATA["players"][r]} for r in ("bat", "bowl")}


@app.after_request
def cors(resp):                      # lets the Vercel front end call this API from another domain
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Cache-Control"] = "public, max-age=60" if request.path.startswith("/api/") else resp.headers.get("Cache-Control", "")
    return resp


@app.get("/api/health")              # point UptimeRobot here
def health():
    return jsonify(status="ok", matches=DATA["meta"]["matches"], deliveries=DATA["meta"]["deliveries"])


@app.get("/api/data")
def data():
    return jsonify(DATA)


@app.get("/api/final")
def final():
    return jsonify({r: [dict(rank=p["rank"], id=p["id"], name=p["name"], score=p["score"]) for p in DATA["players"][r][:10]] for r in ("bat", "bowl")})


@app.get("/api/player/<role>/<int:pid>")
def player(role, pid):
    p = PLAYERS.get(role, {}).get(pid)
    if not p:
        abort(404)
    return jsonify(dict(p, log=DATA["logs"].get(f"{role}{pid}", [])))


@app.get("/api/rank")
def rank():
    """Re-rank with your own definition of pressure.
    /api/rank?role=bat&cut=75&wP=45&wI=20&wC=20&wD=15&min_balls=100&min_inns=10&n=10"""
    role = request.args.get("role", "bat")
    cut = request.args.get("cut", "75")
    if role not in PLAYERS or cut not in DATA["whatif"][role]:
        abort(400, "role must be bat|bowl and cut one of " + ",".join(DATA["whatif"]["bat"]))
    g = lambda k, d: float(request.args.get(k, d))
    w = np.array([g("wP", 45), g("wI", 20), g("wC", 20), g("wD", 15)])
    mb, mi, n = g("min_balls", 100), g("min_inns", 10), int(g("n", 10))
    items = [(i, v) for i, v in DATA["whatif"][role][cut].items() if v[5] >= mb and v[6] >= mi]
    if len(items) < 12:
        abort(400, "too few players pass the minimums")
    M = np.array([[v[0], v[3], v[4], v[2]] for _, v in items])          # P, I, C, D
    Z = (M - M.mean(0)) / np.where(M.std(0) == 0, 1, M.std(0))
    s = Z @ (w / (w.sum() or 1))
    order = np.argsort(-s)[:n]
    return jsonify([dict(rank=k + 1, id=int(items[j][0]), name=DATA["names"][items[j][0]], score=round(float(s[j]), 3),
                         pressure_balls=items[j][1][5], innings=items[j][1][6]) for k, j in enumerate(order)])


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
