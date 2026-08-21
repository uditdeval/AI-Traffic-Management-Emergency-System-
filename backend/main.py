from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "trafiq.db"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8n.pt")
CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.35"))
SAMPLE_EVERY = max(1, int(os.getenv("SAMPLE_EVERY", "2")))
PIXELS_PER_METER = float(os.getenv("PIXELS_PER_METER", "10"))
DEFAULT_THRESHOLD = int(os.getenv("HIGH_DENSITY_THRESHOLD", "50"))

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

app = FastAPI(
    title="TRAFIQ API",
    version="1.0.0",
    description="Backend for the TRAFIQ AI Smart Traffic Management System.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to your GitHub Pages URL in production.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_LOCK = threading.Lock()
MODEL_LOCK = threading.Lock()
MODEL = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with DB_LOCK:
        conn = db()
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS junctions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                online INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                junction_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                created_at TEXT NOT NULL,
                frames_processed INTEGER NOT NULL DEFAULT 0,
                duration_seconds REAL NOT NULL DEFAULT 0,
                total_vehicles INTEGER NOT NULL DEFAULT 0,
                unique_tracks INTEGER NOT NULL DEFAULT 0,
                car INTEGER NOT NULL DEFAULT 0,
                motorcycle INTEGER NOT NULL DEFAULT 0,
                bus INTEGER NOT NULL DEFAULT 0,
                truck INTEGER NOT NULL DEFAULT 0,
                density TEXT NOT NULL DEFAULT 'LOW',
                queue_length_m REAL NOT NULL DEFAULT 0,
                avg_speed_kmh REAL NOT NULL DEFAULT 0,
                avg_wait_seconds REAL NOT NULL DEFAULT 0,
                recommended_green_seconds INTEGER NOT NULL DEFAULT 20,
                emergency_detected INTEGER NOT NULL DEFAULT 0,
                emergency_type TEXT,
                emergency_arm TEXT,
                FOREIGN KEY(junction_id) REFERENCES junctions(id)
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                message TEXT NOT NULL DEFAULT '',
                error TEXT,
                result_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY CHECK(id=1),
                high_density_alerts INTEGER NOT NULL DEFAULT 1,
                emergency_priority INTEGER NOT NULL DEFAULT 1,
                camera_health_alerts INTEGER NOT NULL DEFAULT 1,
                weekly_analytics_email INTEGER NOT NULL DEFAULT 0,
                high_density_threshold INTEGER NOT NULL DEFAULT 50
            );
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO settings(id, high_density_threshold) VALUES(1, ?)",
            (DEFAULT_THRESHOLD,),
        )

        default_junctions = [
            ("CAM-04", "MG Road & 5th Ave"),
            ("CAM-01", "Station Road Circle"),
            ("CAM-02", "College Junction"),
            ("CAM-03", "Ring Road North"),
            ("CAM-05", "Market Square"),
        ]
        for jid, name in default_junctions:
            conn.execute(
                "INSERT OR IGNORE INTO junctions(id,name,online,created_at) VALUES(?,?,1,?)",
                (jid, name, now_iso()),
            )
        conn.commit()
        conn.close()


init_db()


class JunctionCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)


class SettingsUpdate(BaseModel):
    high_density_alerts: bool = True
    emergency_priority: bool = True
    camera_health_alerts: bool = True
    weekly_analytics_email: bool = False
    high_density_threshold: int = Field(default=50, ge=10, le=200)


def get_settings() -> dict[str, Any]:
    conn = db()
    row = conn.execute("SELECT * FROM settings WHERE id=1").fetchone()
    conn.close()
    if not row:
        return {
            "high_density_alerts": True,
            "emergency_priority": True,
            "camera_health_alerts": True,
            "weekly_analytics_email": False,
            "high_density_threshold": DEFAULT_THRESHOLD,
        }
    return {
        "high_density_alerts": bool(row["high_density_alerts"]),
        "emergency_priority": bool(row["emergency_priority"]),
        "camera_health_alerts": bool(row["camera_health_alerts"]),
        "weekly_analytics_email": bool(row["weekly_analytics_email"]),
        "high_density_threshold": int(row["high_density_threshold"]),
    }


def get_model():
    global MODEL
    if YOLO is None:
        raise RuntimeError(
            "Ultralytics is not installed. Run: pip install -r requirements.txt"
        )
    with MODEL_LOCK:
        if MODEL is None:
            MODEL = YOLO(YOLO_MODEL_NAME)
    return MODEL


def classify_density(total: int, threshold: int) -> str:
    if total >= threshold:
        return "HIGH"
    if total >= max(0, threshold - 20):
        return "MEDIUM"
    return "LOW"


def recommended_green(density: str, total: int) -> int:
    if density == "HIGH":
        return min(90, max(60, 60 + int(total * 0.45)))
    if density == "MEDIUM":
        return min(60, max(35, 35 + int(total * 0.25)))
    return min(35, max(20, 20 + int(total * 0.10)))


def make_result(
    *,
    frames_processed: int,
    duration_seconds: float,
    latest_counts: dict[str, int],
    unique_tracks: int,
    queue_length_m: float,
    avg_speed_kmh: float,
    avg_wait_seconds: float,
    threshold: int,
) -> dict[str, Any]:
    total = sum(latest_counts.values())
    density = classify_density(total, threshold)
    return {
        "frames_processed": frames_processed,
        "duration_seconds": round(duration_seconds, 2),
        "total_vehicles": total,
        "unique_tracks": unique_tracks,
        "counts": latest_counts,
        "density": density,
        "queue_length_m": round(queue_length_m, 1),
        "avg_speed_kmh": round(avg_speed_kmh, 1),
        "avg_wait_seconds": round(avg_wait_seconds, 1),
        "recommended_green_seconds": recommended_green(density, total),
        "emergency": {"detected": False, "type": None, "arm": None},
    }


def analyze_video(path: Path, progress) -> dict[str, Any]:
    """
    Real video analysis using Ultralytics YOLO tracking.

    COCO's standard classes cover car, motorcycle, bus and truck. Ambulance/fire
    truck are NOT separate COCO classes, so emergency detection is left as a
    hook for a custom emergency-vehicle model rather than being falsely claimed.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count else 0.0

    model = get_model()
    threshold = get_settings()["high_density_threshold"]

    unique_ids: set[int] = set()
    latest_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
    history: list[int] = []
    queue_samples: list[float] = []
    speed_samples: list[float] = []
    track_positions: dict[int, tuple[float, float, int]] = {}

    frame_index = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % SAMPLE_EVERY != 0:
            frame_index += 1
            continue

        processed += 1
        h, w = frame.shape[:2]
        stop_y = int(h * 0.62)

        try:
            with MODEL_LOCK:
                results = model.track(
                    frame,
                    persist=True,
                    classes=list(VEHICLE_CLASSES.keys()),
                    conf=CONFIDENCE,
                    verbose=False,
                )
        except Exception:
            # Some model/runtime combinations do not support persistence
            # cleanly; a normal detection pass still gives useful counts.
            with MODEL_LOCK:
                results = model.predict(
                    frame,
                    classes=list(VEHICLE_CLASSES.keys()),
                    conf=CONFIDENCE,
                    verbose=False,
                )

        counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        frame_queue = 0.0
        frame_speeds = []

        if results:
            r = results[0]
            boxes = getattr(r, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                ids = (
                    boxes.id.cpu().numpy().astype(int)
                    if getattr(boxes, "id", None) is not None
                    else np.arange(len(xyxy)) + processed * 100000
                )

                for box, c, tid in zip(xyxy, cls, ids):
                    label = VEHICLE_CLASSES.get(int(c))
                    if not label:
                        continue
                    counts[label] += 1
                    unique_ids.add(int(tid))

                    x1, y1, x2, y2 = map(float, box)
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    # Queue approximation: vehicles before the stop line.
                    if cy < stop_y:
                        distance_px = max(0.0, stop_y - cy)
                        frame_queue = max(frame_queue, distance_px / PIXELS_PER_METER)

                    old = track_positions.get(int(tid))
                    if old and processed > old[2]:
                        dx = cx - old[0]
                        dy = cy - old[1]
                        pixels = (dx * dx + dy * dy) ** 0.5
                        dt = (processed - old[2]) * SAMPLE_EVERY / fps
                        if dt > 0:
                            speed_kmh = (pixels / PIXELS_PER_METER) / dt * 3.6
                            if 0 <= speed_kmh <= 120:
                                frame_speeds.append(speed_kmh)
                    track_positions[int(tid)] = (cx, cy, processed)

        latest_counts = counts
        total = sum(counts.values())
        history.append(total)
        history = history[-15:]
        queue_samples.append(frame_queue)
        if frame_speeds:
            speed_samples.extend(frame_speeds[-20:])

        progress_value = int(min(95, (frame_index / max(frame_count, 1)) * 95))
        progress(progress_value, f"Running YOLO detection · frame {frame_index}/{frame_count}")

        frame_index += 1

    cap.release()

    avg_speed = float(np.mean(speed_samples)) if speed_samples else 0.0
    density = classify_density(sum(latest_counts.values()), threshold)
    # Convert stopped/slow observations into a rough waiting-time estimate.
    avg_wait = (
        max(0.0, min(180.0, 90.0 - avg_speed * 2.0))
        if avg_speed
        else (80.0 if density == "HIGH" else 40.0 if density == "MEDIUM" else 15.0)
    )

    result = make_result(
        frames_processed=processed,
        duration_seconds=duration,
        latest_counts=latest_counts,
        unique_tracks=len(unique_ids),
        queue_length_m=float(np.mean(queue_samples[-30:])) if queue_samples else 0.0,
        avg_speed_kmh=avg_speed,
        avg_wait_seconds=avg_wait,
        threshold=threshold,
    )
    result["history"] = history
    result["progress"] = 100
    return result


def update_job(job_id: str, *, status=None, progress=None, message=None, error=None, result=None):
    with DB_LOCK:
        conn = db()
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            conn.close()
            return
        conn.execute(
            """
            UPDATE jobs SET
              status=COALESCE(?,status),
              progress=COALESCE(?,progress),
              message=COALESCE(?,message),
              error=COALESCE(?,error),
              result_json=COALESCE(?,result_json),
              updated_at=?
            WHERE id=?
            """,
            (
                status,
                progress,
                message,
                error,
                json.dumps(result) if result is not None else None,
                now_iso(),
                job_id,
            ),
        )
        conn.commit()
        conn.close()


def run_job(job_id: str, file_path: Path, junction_id: str, filename: str):
    try:
        update_job(job_id, status="processing", progress=1, message="Preparing video")

        def progress(p: int, msg: str):
            update_job(job_id, progress=p, message=msg)

        result = analyze_video(file_path, progress)

        analysis_id = str(uuid.uuid4())
        c = result["counts"]
        with DB_LOCK:
            conn = db()
            conn.execute(
                """
                INSERT INTO analyses(
                  id,junction_id,filename,created_at,frames_processed,duration_seconds,
                  total_vehicles,unique_tracks,car,motorcycle,bus,truck,density,
                  queue_length_m,avg_speed_kmh,avg_wait_seconds,recommended_green_seconds,
                  emergency_detected,emergency_type,emergency_arm
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    analysis_id,
                    junction_id,
                    filename,
                    now_iso(),
                    result["frames_processed"],
                    result["duration_seconds"],
                    result["total_vehicles"],
                    result["unique_tracks"],
                    c["car"],
                    c["motorcycle"],
                    c["bus"],
                    c["truck"],
                    result["density"],
                    result["queue_length_m"],
                    result["avg_speed_kmh"],
                    result["avg_wait_seconds"],
                    result["recommended_green_seconds"],
                    0,
                    None,
                    None,
                ),
            )
            conn.commit()
            conn.close()

        result["analysis_id"] = analysis_id
        update_job(job_id, status="completed", progress=100, message="Analysis complete", result=result)
    except Exception as exc:
        update_job(job_id, status="failed", message="Analysis failed", error=str(exc))
    finally:
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.get("/")
def root():
    return {
        "name": "TRAFIQ API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "trafiq-backend",
        "time": now_iso(),
        "yolo_available": YOLO is not None,
        "model": YOLO_MODEL_NAME,
    }


@app.get("/api/junctions")
def list_junctions():
    conn = db()
    rows = conn.execute("SELECT * FROM junctions ORDER BY name").fetchall()
    output = []
    for row in rows:
        latest = conn.execute(
            "SELECT * FROM analyses WHERE junction_id=? ORDER BY created_at DESC LIMIT 1",
            (row["id"],),
        ).fetchone()
        latest_data = None
        if latest:
            latest_data = {
                "total_vehicles": latest["total_vehicles"],
                "density": latest["density"],
                "queue_length_m": latest["queue_length_m"],
                "avg_speed_kmh": latest["avg_speed_kmh"],
                "avg_wait_seconds": latest["avg_wait_seconds"],
                "recommended_green_seconds": latest["recommended_green_seconds"],
            }
        output.append({
            "id": row["id"],
            "name": row["name"],
            "online": bool(row["online"]),
            "created_at": row["created_at"],
            "latest": latest_data,
        })
    conn.close()
    return output


@app.post("/api/junctions")
def create_junction(payload: JunctionCreate):
    jid = "CAM-" + str(uuid.uuid4())[:6].upper()
    with DB_LOCK:
        conn = db()
        conn.execute(
            "INSERT INTO junctions(id,name,online,created_at) VALUES(?,?,1,?)",
            (jid, payload.name.strip(), now_iso()),
        )
        conn.commit()
        conn.close()
    return {"id": jid, "name": payload.name.strip(), "online": True}


@app.get("/api/dashboard")
def dashboard(junction_id: str = "CAM-04"):
    conn = db()
    row = conn.execute(
        "SELECT * FROM analyses WHERE junction_id=? ORDER BY created_at DESC LIMIT 1",
        (junction_id,),
    ).fetchone()
    if not row:
        conn.close()
        return {
            "junction_id": junction_id,
            "total_vehicles": 0,
            "counts": {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0},
            "density": "LOW",
            "queue_length_m": 0,
            "avg_speed_kmh": 0,
            "avg_wait_seconds": 0,
            "recommended_green_seconds": 20,
            "history": [],
            "emergency": {"detected": False},
        }

    history_rows = conn.execute(
        """
        SELECT total_vehicles FROM analyses
        WHERE junction_id=? ORDER BY created_at DESC LIMIT 15
        """,
        (junction_id,),
    ).fetchall()
    conn.close()

    return {
        "junction_id": junction_id,
        "total_vehicles": row["total_vehicles"],
        "counts": {
            "car": row["car"],
            "motorcycle": row["motorcycle"],
            "bus": row["bus"],
            "truck": row["truck"],
        },
        "density": row["density"],
        "queue_length_m": row["queue_length_m"],
        "avg_speed_kmh": row["avg_speed_kmh"],
        "avg_wait_seconds": row["avg_wait_seconds"],
        "recommended_green_seconds": row["recommended_green_seconds"],
        "history": list(reversed([r["total_vehicles"] for r in history_rows])),
        "emergency": {
            "detected": bool(row["emergency_detected"]),
            "type": row["emergency_type"],
            "arm": row["emergency_arm"],
        },
        "last_analysis": row["created_at"],
    }


@app.post("/api/analyze")
async def create_analysis_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    junction_id: str = Form("CAM-04"),
):
    if not file.filename:
        raise HTTPException(400, "No file selected.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise HTTPException(400, "Unsupported video format.")

    size_limit = MAX_UPLOAD_MB * 1024 * 1024
    job_id = str(uuid.uuid4())
    target = UPLOAD_DIR / f"{job_id}{suffix}"

    total = 0
    with target.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > size_limit:
                out.close()
                target.unlink(missing_ok=True)
                raise HTTPException(413, f"Video exceeds {MAX_UPLOAD_MB} MB.")
            out.write(chunk)

    with DB_LOCK:
        conn = db()
        conn.execute(
            "INSERT INTO jobs(id,status,progress,message,created_at,updated_at) VALUES(?,?,?,?,?,?)",
            (job_id, "queued", 0, "Video uploaded — queued for AI analysis", now_iso(), now_iso()),
        )
        conn.commit()
        conn.close()

    background_tasks.add_task(run_job, job_id, target, junction_id, file.filename)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    conn = db()
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Job not found.")
    result = json.loads(row["result_json"]) if row["result_json"] else None
    return {
        "job_id": row["id"],
        "status": row["status"],
        "progress": row["progress"],
        "message": row["message"],
        "error": row["error"],
        "result": result,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


@app.get("/api/analyses")
def list_analyses(limit: int = 20, junction_id: Optional[str] = None):
    limit = max(1, min(limit, 100))
    conn = db()
    if junction_id:
        rows = conn.execute(
            "SELECT * FROM analyses WHERE junction_id=? ORDER BY created_at DESC LIMIT ?",
            (junction_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "junction_id": r["junction_id"],
            "filename": r["filename"],
            "created_at": r["created_at"],
            "frames_processed": r["frames_processed"],
            "duration_seconds": r["duration_seconds"],
            "total_vehicles": r["total_vehicles"],
            "unique_tracks": r["unique_tracks"],
            "density": r["density"],
            "queue_length_m": r["queue_length_m"],
            "avg_speed_kmh": r["avg_speed_kmh"],
            "avg_wait_seconds": r["avg_wait_seconds"],
            "recommended_green_seconds": r["recommended_green_seconds"],
            "counts": {
                "car": r["car"],
                "motorcycle": r["motorcycle"],
                "bus": r["bus"],
                "truck": r["truck"],
            },
        }
        for r in rows
    ]


@app.get("/api/settings")
def read_settings():
    return get_settings()


@app.put("/api/settings")
def write_settings(payload: SettingsUpdate):
    with DB_LOCK:
        conn = db()
        conn.execute(
            """
            UPDATE settings SET
              high_density_alerts=?,
              emergency_priority=?,
              camera_health_alerts=?,
              weekly_analytics_email=?,
              high_density_threshold=?
            WHERE id=1
            """,
            (
                int(payload.high_density_alerts),
                int(payload.emergency_priority),
                int(payload.camera_health_alerts),
                int(payload.weekly_analytics_email),
                payload.high_density_threshold,
            ),
        )
        conn.commit()
        conn.close()
    return payload.model_dump()


@app.get("/api/videos/{filename}")
def get_uploaded_video(filename: str):
    # Optional endpoint for future playback; current analysis deletes uploads after processing.
    path = UPLOAD_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(404, "Video not found.")
    return FileResponse(path)
