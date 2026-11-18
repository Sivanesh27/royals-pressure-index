# Deploying the Royals Pressure Index

Layout: `static/` is the dashboard (plain HTML, no build step). `app.py` is a small Flask API that also serves the dashboard.
Plan: Vercel hosts the page, Render hosts the API, UptimeRobot keeps Render awake. If the API is down the page falls back to `static/data.json`, so it never shows blank.

## 0. Try it locally
    pip install -r requirements.txt
    python app.py            # open http://localhost:8000

## 1. GitHub
1. Create an empty repository on github.com (for example `royals-pressure-index`).
2. In this folder:
       git init && git add . && git commit -m "Pressure index dashboard"
       git branch -M main
       git remote add origin https://github.com/<you>/royals-pressure-index.git
       git push -u origin main
   Do not commit the raw data pack (`raw_data/` is already in .gitignore).

## 2. Render (API)
1. render.com > New > Web Service > connect the repo. (Or New > Blueprint, which reads `render.yaml`.)
2. Runtime Python, build `pip install -r requirements.txt`, start `gunicorn app:app --workers 1 --threads 4 --timeout 60`.
3. Health check path `/api/health`. Free plan.
4. When it is live, open `https://<name>.onrender.com/api/health`. You should see `{"status":"ok",...}`.
   Other endpoints: `/api/final`, `/api/rank?role=bat&cut=75&wP=45&wI=20&wC=20&wD=15`, `/api/player/bat/<id>`.
   The free plan sleeps after about 15 minutes idle; the first request then takes up to a minute.

## 3. Vercel (dashboard)
1. Edit `static/config.js`:  `window.API_BASE="https://<name>.onrender.com";` then commit and push.
2. vercel.com > Add New > Project > import the repo. Framework preset: Other. Output directory: `static` (already set in `vercel.json`). No build command.
3. Deploy. That URL is the dashboard link for the Unstop form.

## 4. UptimeRobot
1. uptimerobot.com > Add New Monitor > HTTP(s).
2. URL `https://<name>.onrender.com/api/health`, interval 5 minutes.
3. This keeps the free Render service from sleeping and emails you if it goes down.

## 5. Database
Not needed. All results are precomputed into a 1 MB JSON file. If you later want to store visitor input, Neon (serverless Postgres) is the easy free option: create a project, copy the connection string into a Render environment variable `DATABASE_URL`, and add `psycopg[binary]` to requirements.

## Photos
Put licensed photos in `static/photos/<player_id>.jpg`. The ids are in `static/data.json` under `players`. Without a file the page shows initials.

## Rebuilding the numbers
    DATA_DIR="/path/to/Data Pack/" sh run_pipeline.sh
Needs pandas, numpy, scikit-learn, scipy, matplotlib, reportlab.
