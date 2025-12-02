from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Literal, List

from fastapi import FastAPI, Depends, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, Field

from sqlalchemy import (
    create_engine,
    Integer,
    String,
    DateTime,
    ForeignKey,
    LargeBinary,
    select,
    func,
    delete,
    create_engine, Integer, String, DateTime, ForeignKey,
    select, func, Float
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
    Session,
)
from sqlalchemy.exc import IntegrityError
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from fastapi.responses import FileResponse

from sqlalchemy import inspect, text
import uuid
import shutil
import re

RETAIN_DAYS = int(os.getenv("UPLOAD_RETAIN_DAYS", "1"))  # keep only today by default

# ---------- Paths & DB ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent

STATIC_DIR = ROOT / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"

DB_URL = os.getenv("DATABASE_URL")
if DB_URL:
    # Normalize scheme for psycopg v3
    if DB_URL.startswith("postgres://"):
        DB_URL = DB_URL.replace("postgres://", "postgresql+psycopg://", 1)
else:
    DB_URL = f"sqlite:///{ROOT.parent / 'geoguessr.db'}"

# SQLite needs special connect args; Postgres does not
connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite:///") else {}
engine = create_engine(
    DB_URL, echo=False, future=True, pool_pre_ping=True, connect_args=connect_args
)
SessionLocal = sessionmaker(
    bind=engine, autoflush=False, autocommit=False, future=True
)


# ---------- ORM Models ---------------------------------------------------------
class Base(DeclarativeBase):
    pass


class Player(Base):
    __tablename__ = "players"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), index=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    entries: Mapped[List["ScoreEntry"]] = relationship(back_populates="player")


class Board(Base):
    __tablename__ = "boards"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    slug: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    owner_player_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("players.id"), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    owner_player_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("players.id"), nullable=True
    )


class ScoreEntry(Base):
    __tablename__ = "score_entries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.id", ondelete="CASCADE"), index=True
    )
    played_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    round1: Mapped[int] = mapped_column(Integer)
    round2: Mapped[int] = mapped_column(Integer)
    round3: Mapped[int] = mapped_column(Integer)
    total_score: Mapped[int] = mapped_column(Integer)
    board_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("boards.id"), index=True, nullable=True
    )

    # screenshots (per round) – values are URL paths like "/images/<id>"
    screenshot_r1_path: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    screenshot_r2_path: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    screenshot_r3_path: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )

    # legacy single screenshot (kept for backward compat; optional)
    screenshot_path: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )

    player: Mapped[Player] = relationship(back_populates="entries")

    # One GeoGame per ScoreEntry (GeoGuessr import)
    geogame: Mapped[Optional["GeoGame"]] = relationship(
        back_populates="score_entry",
        uselist=False,
        cascade="all, delete-orphan",
    )


class GeoGame(Base):
    """
    One imported GeoGuessr game per ScoreEntry (per player+board+day).
    Stores overall distance etc.
    """
    __tablename__ = "geogames"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    score_entry_id: Mapped[int] = mapped_column(
        ForeignKey("score_entries.id", ondelete="CASCADE"),
        unique=True,
        index=True,
    )
    geoguessr_game_id: Mapped[Optional[str]] = mapped_column(
        String(80), nullable=True
    )
    total_distance_m: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    score_entry: Mapped[ScoreEntry] = relationship(back_populates="geogame")
    rounds: Mapped[List["GeoRound"]] = relationship(
        back_populates="game", cascade="all, delete-orphan"
    )


class GeoRound(Base):
    """
    One row per round with guess & target coordinates and distance.
    """
    __tablename__ = "georounds"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(
        ForeignKey("geogames.id", ondelete="CASCADE"), index=True
    )
    round_index: Mapped[int] = mapped_column(Integer)  # 1..3

    guess_lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    guess_lng: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    target_lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    target_lng: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    distance_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    game: Mapped[GeoGame] = relationship(back_populates="rounds")

class ImageBlob(Base):
    __tablename__ = "images"

    # 32-character hex UUID string
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    mime_type: Mapped[str] = mapped_column(String(100))
    data: Mapped[bytes] = mapped_column(LargeBinary)


# Create tables if they don't exist
Base.metadata.create_all(bind=engine)


def ensure_round_shot_columns():
    insp = inspect(engine)
    cols = {c["name"] for c in insp.get_columns("score_entries")}
    stmts = []
    if "screenshot_r1_path" not in cols:
        stmts.append(
            "ALTER TABLE score_entries ADD COLUMN screenshot_r1_path VARCHAR(255)"
        )
    if "screenshot_r2_path" not in cols:
        stmts.append(
            "ALTER TABLE score_entries ADD COLUMN screenshot_r2_path VARCHAR(255)"
        )
    if "screenshot_r3_path" not in cols:
        stmts.append(
            "ALTER TABLE score_entries ADD COLUMN screenshot_r3_path VARCHAR(255)"
        )

    if stmts:
        with engine.begin() as conn:
            for sql in stmts:
                conn.exec_driver_sql(
                    sql
                )  # one statement at a time (works on SQLite & Postgres)


ensure_round_shot_columns()

def ensure_distance_columns():
    insp = inspect(engine)
    cols = {c['name'] for c in insp.get_columns('score_entries')}
    stmts = []
    if 'distance_r1_m' not in cols:
        stmts.append("ALTER TABLE score_entries ADD COLUMN distance_r1_m FLOAT")
    if 'distance_r2_m' not in cols:
        stmts.append("ALTER TABLE score_entries ADD COLUMN distance_r2_m FLOAT")
    if 'distance_r3_m' not in cols:
        stmts.append("ALTER TABLE score_entries ADD COLUMN distance_r3_m FLOAT")

    if stmts:
        with engine.begin() as conn:
            for sql in stmts:
                conn.exec_driver_sql(sql)

def ensure_se_column():
    insp = inspect(engine)
    cols = [c["name"] for c in insp.get_columns("score_entries")]
    if "screenshot_path" not in cols:
        with engine.begin() as conn:
            try:
                conn.execute(
                    text("ALTER TABLE score_entries ADD COLUMN screenshot_path VARCHAR(255)")
                )
            except Exception:
                pass


def cleanup_old_uploads():
    """
    Legacy filesystem cleanup: old /static/uploads/YYYYMMDD folders.
    Safe to keep around for local dev or older data.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date()
    for p in UPLOAD_DIR.iterdir():
        # only touch YYYYMMDD folders
        if p.is_dir() and re.fullmatch(r"\d{8}", p.name):
            try:
                d = datetime.strptime(p.name, "%Y%m%d").date()
            except ValueError:
                continue
            # delete anything not within the retention window
            if (today - d).days >= RETAIN_DAYS:
                shutil.rmtree(p, ignore_errors=True)

def ensure_geoguessr_columns():
    """
    Add distance/coordinate columns for GeoGuessr imports if they don't exist yet.
    Safe on both SQLite and Postgres.
    """
    insp = inspect(engine)
    existing = {c['name'] for c in insp.get_columns('score_entries')}
    # Use a generic FLOAT type that works on both SQLite and Postgres.
    float_type = "DOUBLE PRECISION" if engine.dialect.name.startswith("postgres") else "FLOAT"

    to_add = {
        "distance_r1_m": float_type,
        "distance_r2_m": float_type,
        "distance_r3_m": float_type,
        "total_distance_m": float_type,
        "guess_r1_lat": float_type,
        "guess_r1_lng": float_type,
        "target_r1_lat": float_type,
        "target_r1_lng": float_type,
        "guess_r2_lat": float_type,
        "guess_r2_lng": float_type,
        "target_r2_lat": float_type,
        "target_r2_lng": float_type,
        "guess_r3_lat": float_type,
        "guess_r3_lng": float_type,
        "target_r3_lat": float_type,
        "target_r3_lng": float_type,
    }

    stmts = []
    for col, typ in to_add.items():
        if col not in existing:
            stmts.append(f"ALTER TABLE score_entries ADD COLUMN {col} {typ}")

    if stmts:
        with engine.begin() as conn:
            for sql in stmts:
                conn.exec_driver_sql(sql)


def cleanup_old_images():
    """
    Delete images older than RETAIN_DAYS from the DB.
    With RETAIN_DAYS=1 this keeps roughly 'today' and removes older.
    """
    if RETAIN_DAYS <= 0:
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=RETAIN_DAYS)
    with SessionLocal() as db:
        db.execute(delete(ImageBlob).where(ImageBlob.created_at < cutoff))
        db.commit()


ensure_se_column()
ensure_geoguessr_columns()
cleanup_old_uploads()
cleanup_old_images()


# Ensure a Global board exists (stats-only)
def ensure_global_board() -> None:
    with SessionLocal() as db:
        exists = db.execute(select(Board).where(Board.slug == "global")).scalar_one_or_none()
        if not exists:
            db.add(Board(name="Global", slug="global", created_at=datetime.now(timezone.utc)))
            db.commit()


ensure_global_board()

# ---------- App & Templates ----------------------------------------------------
app = FastAPI(title="GeoGuesser Leaderboard", version="0.6.3")
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
app.add_middleware(SessionMiddleware, secret_key="change-me-please-very-secret")

static_dir = ROOT / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------- Schemas ------------------------------------------------------------
class SubmitEntryIn(BaseModel):
    player_name: str = Field(..., min_length=1, max_length=80)
    round1: int = Field(..., ge=0, le=5000)
    round2: int = Field(..., ge=0, le=5000)
    round3: int = Field(..., ge=0, le=5000)
    board_slug: Optional[str] = Field(default=None)  # API only; UI uses path param



class GeoGuessrRoundIn(BaseModel):
    score: int = Field(..., ge=0, le=5000)
    distance_m: Optional[float] = Field(default=None, ge=0)
    guess_lat: Optional[float] = None
    guess_lng: Optional[float] = None
    target_lat: Optional[float] = None
    target_lng: Optional[float] = None


class GeoGuessrImportIn(BaseModel):
    player_name: str = Field(..., min_length=1, max_length=80)
    board_slug: str = Field(..., min_length=1, max_length=120)
    total_score: Optional[int] = Field(default=None, ge=0)
    total_distance_m: Optional[float] = Field(default=None, ge=0)
    game_id: Optional[str] = None
    # extension will send an ISO timestamp (captured_at)
    played_at: Optional[datetime] = None
    rounds: List[GeoGuessrRoundIn]


# ---------- Utilities ----------------------------------------------------------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def start_of_today_utc(now: datetime) -> datetime:
    return datetime(now.year, now.month, now.day, tzinfo=timezone.utc)


def start_of_week_utc(now: datetime) -> datetime:
    monday = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) - timedelta(
        days=now.weekday()
    )
    return datetime(monday.year, monday.month, monday.day, tzinfo=timezone.utc)


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return s or "board"


def current_user(request: Request, db: Session) -> Optional[Player]:
    email = request.session.get("user_email")
    if not email:
        return None
    return (
        db.execute(select(Player).where(func.lower(Player.email) == email.lower()))
        .scalar_one_or_none()
    )


def get_all_boards(db: Session) -> list[dict]:
    rows = db.execute(select(Board.id, Board.name, Board.slug).order_by(Board.name.asc())).all()
    return [{"id": r.id, "name": r.name, "slug": r.slug} for r in rows]


def get_board_by_slug(db: Session, slug: str) -> Optional[Board]:
    return db.execute(select(Board).where(Board.slug == slug)).scalar_one_or_none()


def ensure_unique_slug(db: Session, base_slug: str) -> str:
    slug = base_slug
    i = 2
    while db.execute(select(Board.id).where(Board.slug == slug)).first():
        slug = f"{base_slug}-{i}"
        i += 1
    return slug


def fetch_player_entries(db: Session, player_id: int) -> list[dict]:
    # Build a "day" key that works on SQLite and Postgres
    dialect = db.get_bind().dialect.name
    if "sqlite" in dialect:
        day_expr = func.strftime("%Y-%m-%d", ScoreEntry.played_at)
    else:
        day_expr = func.date_trunc("day", ScoreEntry.played_at)

    # Pick the ORIGINAL entry per day = the one with the smallest id for that day
    first_of_day = (
        select(func.min(ScoreEntry.id).label("entry_id"))
        .where(ScoreEntry.player_id == player_id)
        .group_by(day_expr)
        .subquery("first_of_day")
    )

    rows = db.execute(
        select(
            ScoreEntry.played_at,
            ScoreEntry.round1,
            ScoreEntry.round2,
            ScoreEntry.round3,
            ScoreEntry.total_score,
            Board.name.label("board_name"),
            Board.slug.label("board_slug"),
        )
        .join(first_of_day, ScoreEntry.id == first_of_day.c.entry_id)
        .join(Board, ScoreEntry.board_id == Board.id, isouter=True)
        .order_by(ScoreEntry.played_at.desc())
    ).all()

    out: list[dict] = []
    for r in rows:
        dt: datetime = r.played_at
        out.append(
            {
                "played_at": dt,
                "day": dt.date().isoformat(),
                "time": dt.strftime("%H:%M"),
                "round1": int(r.round1 or 0),
                "round2": int(r.round2 or 0),
                "round3": int(r.round3 or 0),
                "total": int(r.total_score or 0),
                "board_name": r.board_name or "—",
                "board_slug": r.board_slug or "",
            }
        )
    return out


def query_leaderboard(
    db: Session, board_id: Optional[int], period: Literal["all", "today", "week"]
) -> list[dict]:
    now = utcnow()
    since: Optional[datetime] = None
    if period == "today":
        since = start_of_today_utc(now)
    elif period == "week":
        since = start_of_week_utc(now)

    # Dialect-neutral "day" key
    dialect = db.get_bind().dialect.name
    if "sqlite" in dialect:
        day_expr = func.strftime("%Y-%m-%d", ScoreEntry.played_at)
    else:
        day_expr = func.date_trunc("day", ScoreEntry.played_at)

    # 1) Collapse duplicates: one row per player per day (across boards)
    day_rows = (
        select(
            Player.id.label("pid"),
            Player.name.label("player_name"),
            day_expr.label("day_key"),
            func.max(ScoreEntry.total_score).label("day_total"),
            func.max(ScoreEntry.round1).label("r1"),
            func.max(ScoreEntry.round2).label("r2"),
            func.max(ScoreEntry.round3).label("r3"),
        )
        .join(ScoreEntry, ScoreEntry.player_id == Player.id)
    )
    if board_id is not None:
        day_rows = day_rows.where(ScoreEntry.board_id == board_id)
    if since is not None:
        day_rows = day_rows.where(ScoreEntry.played_at >= since)

    day_rows = day_rows.group_by("pid", "player_name", "day_key").subquery("per_day")

    # 2) Aggregate per player over the selected period
    agg = select(
        day_rows.c.pid,
        day_rows.c.player_name,
        func.count().label("entries"),  # distinct days
        func.coalesce(func.sum(day_rows.c.day_total), 0).label("total_score"),
        func.avg(day_rows.c.day_total).label("avg_score"),  # average per day
        func.max(day_rows.c.r1).label("max_r1"),
        func.max(day_rows.c.r2).label("max_r2"),
        func.max(day_rows.c.r3).label("max_r3"),
        func.coalesce(func.sum(day_rows.c.r1), 0).label("sum_r1"),
        func.coalesce(func.sum(day_rows.c.r2), 0).label("sum_r2"),
        func.coalesce(func.sum(day_rows.c.r3), 0).label("sum_r3"),
    ).group_by(day_rows.c.pid, day_rows.c.player_name).order_by(
        func.sum(day_rows.c.day_total).desc()
    )

    rows = db.execute(agg).all()

    out: list[dict] = []
    for idx, r in enumerate(rows, start=1):
        best_round = max(
            int(r.max_r1 or 0), int(r.max_r2 or 0), int(r.max_r3 or 0)
        )
        avg_score = float(r.avg_score or 0.0)
        avg_round = avg_score / 3.0  # for Today table
        out.append(
            {
                "rank": idx,
                "player_id": r.pid,
                "player_name": r.player_name,
                "count": int(r.entries or 0),
                "total_score": int(r.total_score or 0),
                "average_score": avg_score,
                "average_round": avg_round,
                "max_round": best_round,
                "round1": int(r.sum_r1 or 0),
                "round2": int(r.sum_r2 or 0),
                "round3": int(r.sum_r3 or 0),
            }
        )

    return out


# ---------- Routes -------------------------------------------------------------
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    ico = static_dir / "favicon.ico"
    svg = static_dir / "favicon.svg"
    if ico.exists():
        return FileResponse(ico, media_type="image/x-icon")
    return FileResponse(svg, media_type="image/svg+xml")


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/board/global", status_code=307)


@app.get("/board/{slug}", response_class=HTMLResponse)
async def board_leaderboard(slug: str, request: Request, db: Session = Depends(get_db)):
    board = get_board_by_slug(db, slug) if slug != "global" else None
    # Select board in session (for navbar Submit)
    request.session["current_board_slug"] = slug if slug != "global" else None

    rows_all = query_leaderboard(
        db, None if slug == "global" else (board.id if board else None), "all"
    )
    rows_week = query_leaderboard(
        db, None if slug == "global" else (board.id if board else None), "week"
    )
    rows_today = query_leaderboard(
        db, None if slug == "global" else (board.id if board else None), "today"
    )

    me = current_user(request, db)
    boards = get_all_boards(db)
    return templates.TemplateResponse(
        "leaderboard.html",
        {
            "request": request,
            "rows_all": rows_all,
            "rows_week": rows_week,
            "rows_today": rows_today,
            "me": me,
            "board": board,  # None means Global
            "is_global": slug == "global",
            "boards": boards,
            "saved": request.query_params.get("saved") == "1",
        },
    )


@app.get("/boards", response_class=HTMLResponse)
async def boards_page(request: Request, db: Session = Depends(get_db)):
    me = current_user(request, db)
    boards = get_all_boards(db)
    return templates.TemplateResponse(
        "boards.html", {"request": request, "me": me, "boards": boards}
    )


@app.post("/boards")
async def create_board(
    request: Request, db: Session = Depends(get_db), name: str = Form(...)
):
    me = current_user(request, db)
    if not me:
        return RedirectResponse(url="/login?next=%2Fboards", status_code=303)
    slug = ensure_unique_slug(db, slugify(name))
    board = Board(
        name=name.strip(),
        slug=slug,
        owner_player_id=me.id if me else None,
        created_at=utcnow(),
    )
    db.add(board)
    db.commit()
    # Select that board immediately
    request.session["current_board_slug"] = slug
    return RedirectResponse(url=f"/board/{slug}", status_code=303)


# ----- Board-scoped submit (no dropdown) --------------------------------------
@app.get("/board/{slug}/submit", response_class=HTMLResponse)
async def submit_form_for_board(
    slug: str, request: Request, db: Session = Depends(get_db)
):
    me = current_user(request, db)
    if not me:
        return templates.TemplateResponse("login_required.html", {"request": request})
    if slug == "global":
        return RedirectResponse(url="/board/global", status_code=303)

    board = get_board_by_slug(db, slug)
    if not board:
        raise HTTPException(404, "Board not found")

    today = start_of_today_utc(utcnow())

    # already on this board today?
    exists_here = db.execute(
        select(ScoreEntry.id).where(
            ScoreEntry.player_id == me.id,
            ScoreEntry.board_id == board.id,
            ScoreEntry.played_at >= today,
        )
    ).first()
    if exists_here:
        # normal limit message
        return templates.TemplateResponse(
            "submit_board.html",
            {"request": request, "me": me, "limit_reached": True, "board": board},
        )

    # submitted somewhere else today? -> reuse automatically
    any_today = (
        db.execute(
            select(ScoreEntry)
            .where(ScoreEntry.player_id == me.id, ScoreEntry.played_at >= today)
            .order_by(ScoreEntry.played_at.desc())
            .limit(1)
        )
        .scalars()
        .first()
    )

    if any_today:
        copy = ScoreEntry(
            player_id=me.id,
            played_at=any_today.played_at,
            round1=any_today.round1,
            round2=any_today.round2,
            round3=any_today.round3,
            total_score=any_today.total_score,
            board_id=board.id,
            screenshot_r1_path=any_today.screenshot_r1_path,
            screenshot_r2_path=any_today.screenshot_r2_path,
            screenshot_r3_path=any_today.screenshot_r3_path,
            screenshot_path=any_today.screenshot_path,  # legacy carry-over
        )
        db.add(copy)
        db.commit()
        request.session["current_board_slug"] = slug
        return RedirectResponse(url=f"/board/{slug}?saved=1&reused=1", status_code=303)

    # no entry yet today -> show form for first entry
    return templates.TemplateResponse(
        "submit_board.html",
        {"request": request, "me": me, "limit_reached": False, "board": board},
    )


@app.post("/board/{slug}/submit")
async def submit_post_for_board(
    slug: str,
    request: Request,
    round1: int = Form(...),
    round2: int = Form(...),
    round3: int = Form(...),
    screenshot_r1: Optional[str] = Form(None),
    screenshot_r2: Optional[str] = Form(None),
    screenshot_r3: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    me = current_user(request, db)
    if not me:
        return RedirectResponse(
            url=f"/login?next=%2Fboard%2F{slug}%2Fsubmit", status_code=303
        )
    if slug == "global":
        return RedirectResponse(url="/board/global", status_code=303)

    board = get_board_by_slug(db, slug)
    if not board:
        raise HTTPException(404, "Board not found")

    today = start_of_today_utc(utcnow())

    # block duplicates on this board
    exists_here = db.execute(
        select(ScoreEntry.id).where(
            ScoreEntry.player_id == me.id,
            ScoreEntry.board_id == board.id,
            ScoreEntry.played_at >= today,
        )
    ).first()
    if exists_here:
        return RedirectResponse(url=f"/board/{slug}/submit?limit=1", status_code=303)

    # if already submitted somewhere else today, reuse that instead of new values
    any_today = (
        db.execute(
            select(ScoreEntry)
            .where(ScoreEntry.player_id == me.id, ScoreEntry.played_at >= today)
            .order_by(ScoreEntry.played_at.desc())
            .limit(1)
        )
        .scalars()
        .first()
    )

    if any_today:
        copy = ScoreEntry(
            player_id=me.id,
            played_at=any_today.played_at,
            round1=any_today.round1,
            round2=any_today.round2,
            round3=any_today.round3,
            total_score=any_today.total_score,
            board_id=board.id,
            screenshot_r1_path=any_today.screenshot_r1_path,
            screenshot_r2_path=any_today.screenshot_r2_path,
            screenshot_r3_path=any_today.screenshot_r3_path,
            screenshot_path=any_today.screenshot_path,  # legacy carry-over
        )
        db.add(copy)
        db.commit()
        request.session["current_board_slug"] = slug
        return RedirectResponse(url=f"/board/{slug}?saved=1&reused=1", status_code=303)

    # first submission of the day -> accept the form values
    try:
        r1 = int(round1)
        r2 = int(round2)
        r3 = int(round3)
    except Exception:
        raise HTTPException(400, detail="Scores must be integers")
    if any(x < 0 or x > 5000 for x in (r1, r2, r3)):
        raise HTTPException(400, detail="Scores must be between 0 and 5000")

    total = r1 + r2 + r3
    entry = ScoreEntry(
        player_id=me.id,
        played_at=utcnow(),
        round1=r1,
        round2=r2,
        round3=r3,
        total_score=total,
        board_id=board.id,
        screenshot_r1_path=screenshot_r1 or None,
        screenshot_r2_path=screenshot_r2 or None,
        screenshot_r3_path=screenshot_r3 or None,
    )

    db.add(entry)
    db.commit()
    request.session["current_board_slug"] = slug
    return RedirectResponse(url=f"/board/{slug}?saved=1", status_code=303)


# ---------- Auth ---------------------------------------------------------------
@app.get("/register", response_class=HTMLResponse)
async def register_form(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/register")
async def register_post(
    request: Request,
    db: Session = Depends(get_db),
    email: str = Form(...),
    name: str = Form(...),
):
    email = email.strip()
    name = name.strip()
    if not email or "@" not in email:
        raise HTTPException(400, detail="Valid email required")
    if not name:
        raise HTTPException(400, detail="Name is required")

    # Try by email
    player = (
        db.execute(select(Player).where(func.lower(Player.email) == email.lower()))
        .scalar_one_or_none()
    )
    if player:
        if player.name != name:
            player.name = name
            db.add(player)
            db.commit()
    else:
        # Reuse by name if legacy UNIQUE(name) existed in older DBs
        by_name = (
            db.execute(select(Player).where(func.lower(Player.name) == name.lower()))
            .scalar_one_or_none()
        )
        if by_name:
            by_name.email = email
            db.add(by_name)
            db.commit()
            player = by_name
        else:
            try:
                player = Player(email=email, name=name)
                db.add(player)
                db.commit()
            except IntegrityError:
                db.rollback()
                existing = (
                    db.execute(
                        select(Player).where(func.lower(Player.name) == name.lower())
                    )
                    .scalar_one_or_none()
                )
                if existing:
                    existing.email = email
                    db.add(existing)
                    db.commit()
                    player = existing
                else:
                    raise

    request.session["user_email"] = email
    request.session["user_name"] = player.name
    return RedirectResponse(url="/board/global", status_code=303)


@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login_post(
    request: Request, db: Session = Depends(get_db), email: str = Form(...)
):
    email = email.strip()
    player = (
        db.execute(select(Player).where(func.lower(Player.email) == email.lower()))
        .scalar_one_or_none()
    )
    if not player:
        return RedirectResponse(url=f"/register?email={email}", status_code=303)
    request.session["user_email"] = email
    request.session["user_name"] = player.name
    next_url = request.query_params.get("next") or "/board/global"
    return RedirectResponse(url=next_url, status_code=303)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/board/global", status_code=303)


# ---------- API ----------------------------------------------------------------
@app.get("/api/leaderboard")
async def api_leaderboard(
    period: Literal["all", "today", "week"] = "all",
    board_slug: str = "global",
    db: Session = Depends(get_db),
):
    if board_slug == "global":
        return query_leaderboard(db, None, period)
    board = get_board_by_slug(db, board_slug)
    if not board:
        raise HTTPException(404, "Board not found")
    return query_leaderboard(db, board.id, period)


@app.post("/api/submit_entry")
async def api_submit_entry(payload: SubmitEntryIn, db: Session = Depends(get_db)):
    """API still accepts board_slug, but rejects 'global'."""
    player = (
        db.execute(
            select(Player).where(func.lower(Player.name) == payload.player_name.lower())
        )
        .scalar_one_or_none()
    )
    if not player:
        player = Player(name=payload.player_name)
        db.add(player)
        db.flush()

    today = start_of_today_utc(utcnow())
    exists = db.execute(
        select(ScoreEntry.id).where(
            ScoreEntry.player_id == player.id, ScoreEntry.played_at >= today
        )
    ).first()
    if exists:
        raise HTTPException(409, detail="Already submitted today")

    if not payload.board_slug or payload.board_slug == "global":
        raise HTTPException(400, "Submissions must target a specific board")

    board = get_board_by_slug(db, payload.board_slug)
    if not board:
        raise HTTPException(400, "Board not found")

    total = int(payload.round1) + int(payload.round2) + int(payload.round3)
    entry = ScoreEntry(
        player_id=player.id,
        played_at=utcnow(),
        round1=int(payload.round1),
        round2=int(payload.round2),
        round3=int(payload.round3),
        total_score=total,
        board_id=board.id,
    )
    db.add(entry)
    db.commit()
    return {"ok": True, "entry_id": entry.id, "total": total}



@app.post("/api/geoguessr/import")
async def api_geoguessr_import(
    payload: GeoGuessrImportIn,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Import a GeoGuessr daily game with per-round coordinates.

    - Uses the currently logged-in GeoLead user (session), NOT the GeoGuessr nick.
    - Rejects imports where the game is not from "today" (UTC).
    - Upserts ScoreEntry for today (player + board).
    - Upserts GeoGame + GeoRound rows for map visualization.
    """

    # 0) Logged-in GeoLead user
    player = current_user(request, db)
    if not player:
        raise HTTPException(status_code=401, detail="Login required to import scores")

    # 1) Validate board
    if not payload.board_slug or payload.board_slug == "global":
        raise HTTPException(400, "Submissions must target a specific board")

    board = get_board_by_slug(db, payload.board_slug)
    if not board:
        raise HTTPException(400, "Board not found")

    # 2) Validate that the game is from "today"
    now = utcnow()
    today_start = start_of_today_utc(now)

    if payload.played_at is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "GeoGuessr data is missing a timestamp. "
                "Please refresh the GeoGuessr results page and try again."
            ),
        )

    # Normalize to UTC
    if payload.played_at.tzinfo is None:
        game_dt_utc = payload.played_at.replace(tzinfo=timezone.utc)
    else:
        game_dt_utc = payload.played_at.astimezone(timezone.utc)

    game_day_start = start_of_today_utc(game_dt_utc)

    if game_day_start != today_start:
        raise HTTPException(
            status_code=400,
            detail=(
                "Last GeoGuessr game is not from today. "
                "Finish today’s daily game first, then try again."
            ),
        )

    # 3) Compute totals if they weren't provided
    round_scores = [int(r.score or 0) for r in payload.rounds]
    while len(round_scores) < 3:
        round_scores.append(0)

    total_score = (
        int(payload.total_score)
        if payload.total_score is not None
        else sum(round_scores[:3])
    )

    total_distance_m = (
        float(payload.total_distance_m)
        if payload.total_distance_m is not None
        else sum(float(r.distance_m or 0.0) for r in payload.rounds)
    )

    # 4) Upsert ScoreEntry for today (per player+board)
    entry = (
        db.execute(
            select(ScoreEntry)
            .where(
                ScoreEntry.player_id == player.id,
                ScoreEntry.board_id == board.id,
                ScoreEntry.played_at >= today_start,
            )
            .limit(1)
        )
        .scalars()
        .first()
    )

    if entry:
        entry.round1 = round_scores[0]
        entry.round2 = round_scores[1]
        entry.round3 = round_scores[2]
        entry.total_score = total_score
        entry.played_at = now
    else:
        entry = ScoreEntry(
            player_id=player.id,
            played_at=now,
            round1=round_scores[0],
            round2=round_scores[1],
            round3=round_scores[2],
            total_score=total_score,
            board_id=board.id,
        )
        db.add(entry)
        db.flush()  # get entry.id

    # 5) Upsert GeoGame
    game = (
        db.execute(
            select(GeoGame).where(GeoGame.score_entry_id == entry.id).limit(1)
        )
        .scalars()
        .first()
    )

    if not game:
        game = GeoGame(
            score_entry_id=entry.id,
            geoguessr_game_id=payload.game_id,
            total_distance_m=total_distance_m,
        )
        db.add(game)
        db.flush()
    else:
        game.geoguessr_game_id = payload.game_id or game.geoguessr_game_id
        game.total_distance_m = total_distance_m

    # 6) Replace GeoRound rows for this game
    for existing in list(game.rounds):
        db.delete(existing)
    game.rounds.clear()

    for idx, r in enumerate(payload.rounds, start=1):
        gr = GeoRound(
            game_id=game.id,
            round_index=idx,
            guess_lat=r.guess_lat,
            guess_lng=r.guess_lng,
            target_lat=r.target_lat,
            target_lng=r.target_lng,
            distance_m=r.distance_m,
            score=int(r.score or 0),
        )
        db.add(gr)

    db.commit()

    return {
        "ok": True,
        "entry_id": entry.id,
        "total_score": total_score,
        "total_distance_m": total_distance_m,
    }

@app.get("/health")
async def health():
    return {"ok": True, "time": utcnow().isoformat()}


@app.get("/me", response_class=HTMLResponse)
async def my_stats(request: Request, db: Session = Depends(get_db)):
    me = current_user(request, db)
    if not me:
        return templates.TemplateResponse("login_required.html", {"request": request})
    entries = fetch_player_entries(db, me.id)
    return templates.TemplateResponse(
        "user_history.html",
        {"request": request, "me": me, "player": me, "entries": entries},
    )


@app.get("/player/{player_id}", response_class=HTMLResponse)
async def player_stats(player_id: int, request: Request, db: Session = Depends(get_db)):
    player = db.get(Player, player_id)
    if not player:
        raise HTTPException(404, "Player not found")
    entries = fetch_player_entries(db, player.id)
    me = current_user(request, db)
    return templates.TemplateResponse(
        "user_history.html",
        {"request": request, "me": me, "player": player, "entries": entries},
    )


@app.get("/boards/new", response_class=HTMLResponse)
async def new_board_form(request: Request, db: Session = Depends(get_db)):
    me = current_user(request, db)
    if not me:
        # only logged-in users can create boards
        return RedirectResponse(url="/login?next=%2Fboards%2Fnew", status_code=303)
    return templates.TemplateResponse("board_new.html", {"request": request, "me": me})


@app.post("/api/paste_image")
async def paste_image(request: Request, db: Session = Depends(get_db)):
    # Purge old images first (DB + any legacy folders)
    cleanup_old_uploads()
    cleanup_old_images()

    form = await request.form()
    f: UploadFile | None = form.get("image")  # JS sends Clipboard image here
    if not f:
        raise HTTPException(400, "No image in form-data field 'image'")

    ctype = (f.content_type or "").lower()
    if ctype not in ("image/png", "image/jpeg", "image/webp"):
        raise HTTPException(400, "Unsupported type")

    data = await f.read()
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(413, "Image too large (max 5 MB)")

    # Store image bytes in DB
    image_id = uuid.uuid4().hex  # 32-char hex string
    img = ImageBlob(id=image_id, mime_type=ctype, data=data)
    db.add(img)
    db.commit()

    # The returned "url" is used by the frontend as screenshot_*_path
    url = f"/images/{image_id}"
    return {"ok": True, "url": url}


@app.get("/images/{image_id}")
async def get_image(image_id: str, db: Session = Depends(get_db)):
    img = db.get(ImageBlob, image_id)
    if not img:
        raise HTTPException(404, "Image not found")
    return Response(content=img.data, media_type=img.mime_type)


@app.get("/board/{slug}/today", response_class=HTMLResponse)
async def todays_round(slug: str, request: Request, db: Session = Depends(get_db)):
    me = current_user(request, db)
    if not me:
        return RedirectResponse(
            url="/login?next=%2Fboard%2F" + slug + "%2Ftoday", status_code=303
        )

    if slug == "global":
        raise HTTPException(404, "Pick a specific board.")

    board = get_board_by_slug(db, slug)
    if not board:
        raise HTTPException(404, "Board not found")

    today = start_of_today_utc(utcnow())

    # gate: only users who submitted to THIS board today may view
    allowed = db.execute(
        select(ScoreEntry.id).where(
            ScoreEntry.player_id == me.id,
            ScoreEntry.board_id == board.id,
            ScoreEntry.played_at >= today,
        )
    ).first()
    if not allowed:
        return templates.TemplateResponse(
            "today_locked.html", {"request": request, "me": me, "board": board}
        )

    # 1) Base rows (scores)
    rows = db.execute(
        select(
            ScoreEntry.id.label("entry_id"),
            Player.name,
            ScoreEntry.round1,
            ScoreEntry.round2,
            ScoreEntry.round3,
            ScoreEntry.total_score,
            ScoreEntry.screenshot_r1_path.label("shot1"),
            ScoreEntry.screenshot_r2_path.label("shot2"),
            ScoreEntry.screenshot_r3_path.label("shot3"),
            ScoreEntry.screenshot_path.label("legacy_shot"),  # fallback
            ScoreEntry.played_at,
        )
        .join(Player, Player.id == ScoreEntry.player_id)
        .where(ScoreEntry.board_id == board.id, ScoreEntry.played_at >= today)
        .order_by(ScoreEntry.total_score.desc())
    ).all()

    posts = [
        {
            "name": r.name,
            "r1": int(r.round1),
            "r2": int(r.round2),
            "r3": int(r.round3),
            "total": int(r.total_score),
            "img1": r.shot1 or r.legacy_shot,
            "img2": r.shot2,
            "img3": r.shot3,
            "time": r.played_at.strftime("%H:%M"),
        }
        for r in rows
    ]
    entry_ids = [r.entry_id for r in rows]

    # 2) Geo data per entry
    geo_by_entry: dict[int, dict] = {}
    if entry_ids:
        geo_rows = db.execute(
            select(
                GeoGame.score_entry_id,
                GeoGame.total_distance_m,
                GeoRound.round_index,
                GeoRound.score,
                GeoRound.distance_m,
                GeoRound.guess_lat,
                GeoRound.guess_lng,
                GeoRound.target_lat,
                GeoRound.target_lng,
            )
            .join(GeoRound, GeoRound.game_id == GeoGame.id, isouter=True)
            .where(GeoGame.score_entry_id.in_(entry_ids))
        ).all()

        for gr in geo_rows:
            bucket = geo_by_entry.setdefault(
                gr.score_entry_id,
                {"total_distance_m": float(gr.total_distance_m or 0.0), "rounds": {}},
            )
            if gr.round_index is not None:
                bucket["rounds"][gr.round_index] = {
                    "score": int(gr.score or 0),
                    "distance_m": float(gr.distance_m or 0.0) if gr.distance_m is not None else None,
                    "guess_lat": gr.guess_lat,
                    "guess_lng": gr.guess_lng,
                    "target_lat": gr.target_lat,
                    "target_lng": gr.target_lng,
                }

    # 3) Build posts list for template
    posts: list[dict] = []
    for r in rows:
        geo = geo_by_entry.get(r.entry_id, {})
        rounds_geo = geo.get("rounds", {})
        total_distance_m = geo.get("total_distance_m")

        def build_round(idx: int, score_value: int) -> dict:
            rg = rounds_geo.get(idx) or {}
            return {
                "index": idx,
                "score": score_value,
                "distance_m": rg.get("distance_m"),
                "guess_lat": rg.get("guess_lat"),
                "guess_lng": rg.get("guess_lng"),
                "target_lat": rg.get("target_lat"),
                "target_lng": rg.get("target_lng"),
                "has_map": rg.get("guess_lat") is not None and rg.get("target_lat") is not None,
            }

        posts.append(
            {
                "name": r.name,
                "r1": int(r.round1 or 0),
                "r2": int(r.round2 or 0),
                "r3": int(r.round3 or 0),
                "total": int(r.total_score or 0),
                "total_distance_m": total_distance_m,
                "total_distance_km": (total_distance_m / 1000.0)
                if total_distance_m
                else None,
                "time": r.played_at.strftime("%H:%M"),
                "rounds": [
                    build_round(1, int(r.round1 or 0)),
                    build_round(2, int(r.round2 or 0)),
                    build_round(3, int(r.round3 or 0)),
                ],
            }
        )

    return templates.TemplateResponse(
        "today_round.html",
        {"request": request, "me": me, "board": board, "posts": posts, "is_global": False},
        {
            "request": request,
            "me": me,
            "board": board,
            "posts": posts,
            "is_global": False,
        },
    )

@app.get("/privacy/extension", response_class=HTMLResponse)
async def extension_privacy(request: Request):
    """
    Privacy policy for the GeoLead ⇄ GeoGuessr Bridge browser extension.
    """
    last_updated = datetime.now(timezone.utc).date().isoformat()
    return templates.TemplateResponse(
        "privacy_extension.html",
        {
            "request": request,
            "last_updated": last_updated,
        },
    )

@app.get("/privacy/extension", response_class=HTMLResponse)
async def extension_privacy(request: Request):
    """
    Privacy policy for the GeoLead ⇄ GeoGuessr Bridge browser extension.
    """
    last_updated = datetime.now(timezone.utc).date().isoformat()
    return templates.TemplateResponse(
        "privacy_extension.html",
        {
            "request": request,
            "last_updated": last_updated,
        },
    )
