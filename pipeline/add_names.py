import os
"""Adds the id->name lookup the dashboard needs and writes static/data.json."""
import json, os, pandas as pd
out = os.environ.get("OUT_DIR", "pipeline/out/")
d = json.load(open(out + "data.json")); e = pd.read_pickle(out + "effects.pkl")
ids = {i for r in d["whatif"] for c in d["whatif"][r] for i in d["whatif"][r][c]}
d["names"] = {i: str(e.name.get(int(i), "?")) for i in ids}
json.dump(d, open("static/data.json", "w"), separators=(",", ":")); print("wrote static/data.json")
