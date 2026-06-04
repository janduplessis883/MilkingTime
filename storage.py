from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


DB_DIR = Path(".milkingtime")
DB_PATH = DB_DIR / "milkingtime.db"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Storage:
    def __init__(self, path: Path = DB_PATH) -> None:
        DB_DIR.mkdir(exist_ok=True)
        self.path = path
        self._init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists workers (
                    id text primary key,
                    full_name text not null,
                    phone text,
                    hourly_rate real not null,
                    active integer not null default 1,
                    created_at text not null
                );

                create table if not exists face_embeddings (
                    id text primary key,
                    worker_id text not null,
                    embedding_json text not null,
                    created_at text not null,
                    foreign key(worker_id) references workers(id)
                );

                create table if not exists shifts (
                    id text primary key,
                    worker_id text not null,
                    checked_in_at text not null,
                    checked_out_at text,
                    checkin_lat real,
                    checkin_lon real,
                    checkout_lat real,
                    checkout_lon real,
                    auto_checkout integer not null default 0,
                    outside_since text,
                    notes text,
                    foreign key(worker_id) references workers(id)
                );

                create table if not exists settings (
                    key text primary key,
                    value text not null
                );
                """
            )
            defaults = {
                "farm_lat": "-33.9249",
                "farm_lon": "18.4241",
                "farm_radius_m": "500",
                "grace_minutes": "10",
                "overtime_multiplier": "1.5",
            }
            for key, value in defaults.items():
                conn.execute(
                    "insert or ignore into settings (key, value) values (?, ?)",
                    (key, value),
                )

    def get_settings(self) -> dict[str, str]:
        with self.connect() as conn:
            rows = conn.execute("select key, value from settings").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def update_settings(self, settings: dict[str, Any]) -> None:
        with self.connect() as conn:
            for key, value in settings.items():
                conn.execute(
                    """
                    insert into settings (key, value) values (?, ?)
                    on conflict(key) do update set value = excluded.value
                    """,
                    (key, str(value)),
                )

    def create_worker(
        self,
        full_name: str,
        phone: str,
        hourly_rate: float,
        embedding: list[float],
    ) -> str:
        worker_id = str(uuid4())
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                insert into workers (id, full_name, phone, hourly_rate, created_at)
                values (?, ?, ?, ?, ?)
                """,
                (worker_id, full_name, phone, hourly_rate, now),
            )
            conn.execute(
                """
                insert into face_embeddings (id, worker_id, embedding_json, created_at)
                values (?, ?, ?, ?)
                """,
                (str(uuid4()), worker_id, json.dumps(embedding), now),
            )
        return worker_id

    def list_workers(self, include_inactive: bool = False) -> list[dict]:
        query = "select * from workers"
        if not include_inactive:
            query += " where active = 1"
        query += " order by full_name"
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(query).fetchall()]

    def list_worker_embeddings(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                select w.id, w.full_name, w.hourly_rate, e.embedding_json
                from workers w
                join face_embeddings e on e.worker_id = w.id
                where w.active = 1
                """
            ).fetchall()
        return [
            {
                "id": row["id"],
                "full_name": row["full_name"],
                "hourly_rate": row["hourly_rate"],
                "embedding": json.loads(row["embedding_json"]),
            }
            for row in rows
        ]

    def get_open_shift(self, worker_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                select * from shifts
                where worker_id = ? and checked_out_at is null
                order by checked_in_at desc
                limit 1
                """,
                (worker_id,),
            ).fetchone()
        return dict(row) if row else None

    def check_in(self, worker_id: str, lat: float, lon: float) -> str:
        existing = self.get_open_shift(worker_id)
        if existing:
            return existing["id"]

        shift_id = str(uuid4())
        with self.connect() as conn:
            conn.execute(
                """
                insert into shifts (id, worker_id, checked_in_at, checkin_lat, checkin_lon)
                values (?, ?, ?, ?, ?)
                """,
                (shift_id, worker_id, utc_now_iso(), lat, lon),
            )
        return shift_id

    def check_out(
        self,
        worker_id: str,
        lat: float | None,
        lon: float | None,
        auto_checkout: bool = False,
        notes: str | None = None,
    ) -> None:
        shift = self.get_open_shift(worker_id)
        if not shift:
            return
        with self.connect() as conn:
            conn.execute(
                """
                update shifts
                set checked_out_at = ?,
                    checkout_lat = ?,
                    checkout_lon = ?,
                    auto_checkout = ?,
                    notes = ?
                where id = ?
                """,
                (
                    utc_now_iso(),
                    lat,
                    lon,
                    1 if auto_checkout else 0,
                    notes,
                    shift["id"],
                ),
            )

    def mark_outside_status(self, shift_id: str, outside_since: str | None) -> None:
        with self.connect() as conn:
            conn.execute(
                "update shifts set outside_since = ? where id = ?",
                (outside_since, shift_id),
            )

    def list_shifts(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                select s.*, w.full_name, w.hourly_rate
                from shifts s
                join workers w on w.id = s.worker_id
                order by s.checked_in_at desc
                """
            ).fetchall()
        return [dict(row) for row in rows]


def supabase_configured() -> bool:
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"))

