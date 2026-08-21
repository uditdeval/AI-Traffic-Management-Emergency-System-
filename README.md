# TRAFIQ — Full-stack AI Traffic Management System

This package turns the uploaded TRAFIQ dashboard from a simulated browser demo into a frontend + FastAPI backend.

## Project structure

```text
TRAFIQ_full_project/
├── frontend/
│   └── index.html
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── .env.example
├── data/
│   └── uploads/
├── run_backend.bat
├── run_backend.sh
└── README.md
```

## What is now real

- FastAPI REST API
- SQLite persistence
- Junction CRUD (currently create + list)
- Video upload
- Background analysis jobs
- Job progress polling
- Real Ultralytics YOLO vehicle detection/tracking
- Vehicle classes: car, motorcycle, bus, truck
- Vehicle counts
- Approximate queue length
- Approximate speed
- Density classification
- Signal green-time recommendation
- Analysis history
- Dashboard API
- Settings API
- CORS for connecting GitHub Pages frontend to the backend

## Important limitation

The standard COCO YOLO model does **not** have an `ambulance` class. The emergency banner in the original UI therefore must not be treated as real emergency-vehicle detection. To make that feature real, train/use a custom emergency-vehicle model and add it to the backend.

Likewise, queue length and speed are estimates based on image geometry. For a serious deployment, calibrate each camera with road geometry/homography and a known distance.

## Run locally on Windows

1. Open a terminal in this project folder.
2. Run:

```bat
run_backend.bat
```

3. The first YOLO run downloads the model weights automatically.
4. Open the API docs:

`http://127.0.0.1:8000/docs`

5. Open `frontend/index.html` in a browser.

The frontend is configured to call:

`http://127.0.0.1:8000`

## GitHub Pages + backend

GitHub Pages can host the frontend, but it cannot run this Python/FastAPI backend. Keep `frontend/index.html` on GitHub Pages and deploy `backend/` to a Python-capable service/server.

Before public deployment, change:

```js
window.TRAFIQ_API_BASE = 'http://127.0.0.1:8000';
```

to your deployed HTTPS API URL.

## Security before public launch

This starter backend is designed for a college project/demo. Before making the API public, add:

- authentication and role-based access
- restricted CORS to your exact frontend domain
- rate limiting
- file-content validation
- reverse proxy + HTTPS
- persistent production database
- object storage for large videos
- job queue (Redis/Celery/RQ) for multiple simultaneous analyses
- camera authentication
- audit logging

Do not put database passwords, API keys, or server credentials into the frontend or GitHub repository.
