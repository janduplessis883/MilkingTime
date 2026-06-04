from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from supabase import Client, create_client


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


DEFAULT_SETTINGS = {
    "farm_lat": "-33.9249",
    "farm_lon": "18.4241",
    "farm_radius_m": "500",
    "grace_minutes": "10",
    "overtime_multiplier": "1.5",
}


class SupabaseStorage:
    def __init__(self, url: str, key: str) -> None:
        self.client: Client = create_client(url, key)
        self._ensure_default_settings()

    def _ensure_default_settings(self) -> None:
        existing = self.get_settings()
        missing = [
            {"key": key, "value": value}
            for key, value in DEFAULT_SETTINGS.items()
            if key not in existing
        ]
        if missing:
            self.client.table("settings").upsert(missing).execute()

    def get_settings(self) -> dict[str, str]:
        rows = self.client.table("settings").select("key,value").execute().data
        return {row["key"]: row["value"] for row in rows}

    def update_settings(self, settings: dict[str, Any]) -> None:
        rows = [{"key": key, "value": str(value)} for key, value in settings.items()]
        self.client.table("settings").upsert(rows).execute()

    def create_worker(
        self,
        full_name: str,
        phone: str,
        hourly_rate: float,
        embedding: list[float],
    ) -> str:
        worker_id = str(uuid4())
        now = utc_now_iso()
        self.client.table("workers").insert(
            {
                "id": worker_id,
                "full_name": full_name,
                "phone": phone,
                "hourly_rate": hourly_rate,
                "active": True,
                "created_at": now,
            }
        ).execute()
        self.client.table("face_embeddings").insert(
            {
                "id": str(uuid4()),
                "worker_id": worker_id,
                "embedding": embedding,
                "created_at": now,
            }
        ).execute()
        return worker_id

    def list_workers(self, include_inactive: bool = False) -> list[dict]:
        query = self.client.table("workers").select("*").order("full_name")
        if not include_inactive:
            query = query.eq("active", True)
        return query.execute().data

    def update_worker_hourly_rate(self, worker_id: str, hourly_rate: float) -> None:
        self.client.table("workers").update({"hourly_rate": hourly_rate}).eq(
            "id", worker_id
        ).execute()

    def replace_worker_embedding(self, worker_id: str, embedding: list[float]) -> None:
        self.client.table("face_embeddings").delete().eq("worker_id", worker_id).execute()
        self.client.table("face_embeddings").insert(
            {
                "id": str(uuid4()),
                "worker_id": worker_id,
                "embedding": embedding,
                "created_at": utc_now_iso(),
            }
        ).execute()

    def list_worker_embeddings(self) -> list[dict]:
        rows = (
            self.client.table("face_embeddings")
            .select("embedding, workers(id, full_name, hourly_rate, active)")
            .execute()
            .data
        )
        workers = []
        for row in rows:
            worker = row.get("workers")
            if not worker or not worker.get("active"):
                continue
            workers.append(
                {
                    "id": worker["id"],
                    "full_name": worker["full_name"],
                    "hourly_rate": worker["hourly_rate"],
                    "embedding": row["embedding"],
                }
            )
        return workers

    def get_open_shift(self, worker_id: str) -> dict | None:
        rows = (
            self.client.table("shifts")
            .select("*")
            .eq("worker_id", worker_id)
            .is_("checked_out_at", "null")
            .order("checked_in_at", desc=True)
            .limit(1)
            .execute()
            .data
        )
        return rows[0] if rows else None

    def check_in(self, worker_id: str, lat: float, lon: float) -> str:
        existing = self.get_open_shift(worker_id)
        if existing:
            return existing["id"]

        shift_id = str(uuid4())
        self.client.table("shifts").insert(
            {
                "id": shift_id,
                "worker_id": worker_id,
                "checked_in_at": utc_now_iso(),
                "checkin_lat": lat,
                "checkin_lon": lon,
                "auto_checkout": False,
            }
        ).execute()
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
        self.client.table("shifts").update(
            {
                "checked_out_at": utc_now_iso(),
                "checkout_lat": lat,
                "checkout_lon": lon,
                "auto_checkout": auto_checkout,
                "notes": notes,
            }
        ).eq("id", shift["id"]).execute()

    def mark_outside_status(self, shift_id: str, outside_since: str | None) -> None:
        self.client.table("shifts").update({"outside_since": outside_since}).eq(
            "id", shift_id
        ).execute()

    def list_shifts(self) -> list[dict]:
        rows = (
            self.client.table("shifts")
            .select("*, workers(full_name, hourly_rate)")
            .order("checked_in_at", desc=True)
            .execute()
            .data
        )
        shifts = []
        for row in rows:
            worker = row.pop("workers", {}) or {}
            row["full_name"] = worker.get("full_name", "Unknown worker")
            row["hourly_rate"] = worker.get("hourly_rate", 0)
            shifts.append(row)
        return shifts


def supabase_configured(secrets: dict | None = None) -> bool:
    if secrets:
        return bool(secrets.get("SUPABASE_URL") and secrets.get("SUPABASE_KEY"))
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"))
