from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

try:
    from streamlit_geolocation import streamlit_geolocation
except ImportError:
    streamlit_geolocation = None

from face_engine import average_embeddings, best_match, image_to_embedding
from geofence import GeoFence, is_inside_geofence
from storage import SupabaseStorage, supabase_configured, utc_now_iso


st.set_page_config(page_title="MilkingTime", page_icon="MT", layout="wide")


def get_supabase_secrets() -> dict[str, str]:
    try:
        connection = st.secrets.get("connections", {}).get("supabase", {})
        url = connection.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
        key = connection.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY")
    except Exception:
        return {}
    if not url or not key:
        return {}
    return {"SUPABASE_URL": url, "SUPABASE_KEY": key}


def get_manager_pin() -> str:
    try:
        app_config = st.secrets.get("app", {})
        return app_config.get(
            "MANAGER_PIN",
            st.secrets.get("MANAGER_PIN", os.getenv("MANAGER_PIN", "1234")),
        )
    except Exception:
        return os.getenv("MANAGER_PIN", "1234")


@st.cache_resource
def init_storage(supabase_url: str | None, supabase_key: str | None):
    if not supabase_url or not supabase_key:
        raise RuntimeError(
            "Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY to Streamlit secrets."
        )
    return SupabaseStorage(supabase_url, supabase_key)


supabase_secrets = get_supabase_secrets()
storage_error = None
try:
    storage = init_storage(
        supabase_secrets.get("SUPABASE_URL"),
        supabase_secrets.get("SUPABASE_KEY"),
    )
except Exception as exc:
    storage_error = str(exc)
    storage = None


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def money(value: float) -> str:
    return f"R {value:,.2f}"


def extract_browser_location(location: object) -> tuple[float, float, float | None] | None:
    if not isinstance(location, dict):
        return None
    lat = location.get("latitude")
    lon = location.get("longitude")
    accuracy = location.get("accuracy")
    if lat is None or lon is None:
        return None
    return float(lat), float(lon), float(accuracy) if accuracy is not None else None


def render_geofence_map(lat: float, lon: float) -> None:
    map_df = pd.DataFrame([{"lat": lat, "lon": lon}])
    st.map(map_df, latitude="lat", longitude="lon", zoom=15, size=80)


def current_fence() -> GeoFence:
    if storage is None:
        raise RuntimeError("Storage is not available.")
    settings = storage.get_settings()
    return GeoFence(
        center_lat=float(settings["farm_lat"]),
        center_lon=float(settings["farm_lon"]),
        radius_m=float(settings["farm_radius_m"]),
        grace_minutes=int(float(settings["grace_minutes"])),
    )


def recognize_from_camera(label: str) -> tuple[dict | None, float, list[float] | None]:
    photo = st.camera_input(label)
    if not photo:
        return None, 0.0, None

    embedding = image_to_embedding(photo.getvalue())
    if storage is None:
        return None, 0.0, embedding
    worker, score = best_match(embedding, storage.list_worker_embeddings())
    return worker, score, embedding


def apply_geofence_status(worker_id: str, lat: float, lon: float) -> str:
    fence = current_fence()
    open_shift = storage.get_open_shift(worker_id)
    if not open_shift:
        return "No active shift."

    inside, distance = is_inside_geofence(lat, lon, fence)
    if inside:
        if open_shift.get("outside_since"):
            storage.mark_outside_status(open_shift["id"], None)
        return f"Inside farm fence ({distance:.0f} m from center)."

    now = datetime.now(timezone.utc)
    outside_since = parse_dt(open_shift.get("outside_since"))
    if outside_since is None:
        storage.mark_outside_status(open_shift["id"], utc_now_iso())
        return (
            f"Outside farm fence ({distance:.0f} m from center). "
            f"Grace period started: {fence.grace_minutes} minutes."
        )

    if now - outside_since >= timedelta(minutes=fence.grace_minutes):
        storage.check_out(
            worker_id,
            lat,
            lon,
            auto_checkout=True,
            notes="Auto checkout after leaving geo-fence grace period.",
        )
        return "Auto checked out after leaving the farm geo-fence."

    remaining = timedelta(minutes=fence.grace_minutes) - (now - outside_since)
    minutes_left = max(0, int(remaining.total_seconds() // 60))
    return f"Outside farm fence. Grace period remaining: about {minutes_left} minutes."


def enrollment_screen() -> None:
    if storage is None:
        st.error("Supabase storage is not available. Check the setup message at the top of the app.")
        return

    st.subheader("Worker self-enrollment")
    st.caption("Capture three face images so the app can build one recognition vector.")

    with st.form("worker_details"):
        full_name = st.text_input("Full name")
        phone = st.text_input("Phone number")
        hourly_rate = st.number_input("Hourly rate (South African Rand)", min_value=0.0, value=50.0, step=5.0)
        submitted = st.form_submit_button("Save details")

    if submitted:
        if not full_name.strip():
            st.error("Please enter a worker name.")
            return
        st.session_state["enrolling_worker"] = {
            "full_name": full_name.strip(),
            "phone": phone.strip(),
            "hourly_rate": hourly_rate,
            "embeddings": [],
        }

    details = st.session_state.get("enrolling_worker")
    if not details:
        return

    st.info(f"Enrolling {details['full_name']}. Capture {3 - len(details['embeddings'])} more image(s).")
    photo = st.camera_input("Enrollment face capture")
    if photo and st.button("Use this capture"):
        details["embeddings"].append(image_to_embedding(photo.getvalue()))
        st.session_state["enrolling_worker"] = details
        st.rerun()

    if len(details["embeddings"]) >= 3:
        embedding = average_embeddings(details["embeddings"])
        worker_id = storage.create_worker(
            details["full_name"],
            details["phone"],
            float(details["hourly_rate"]),
            embedding,
        )
        del st.session_state["enrolling_worker"]
        st.success(f"Enrollment complete. Worker ID: {worker_id}")


def worker_clock_screen() -> None:
    if storage is None:
        st.error("Supabase storage is not available. Check the setup message at the top of the app.")
        return

    st.subheader("Worker clock in / out")
    st.caption("Use face scan and current location to manage a shift.")

    col_a, col_b = st.columns(2)
    with col_a:
        lat = st.number_input("Current latitude", value=float(storage.get_settings()["farm_lat"]), format="%.6f")
    with col_b:
        lon = st.number_input("Current longitude", value=float(storage.get_settings()["farm_lon"]), format="%.6f")

    worker, score, _ = recognize_from_camera("Worker face scan")
    if not worker:
        if score:
            st.warning(f"No confident match. Best score: {score:.2f}")
        return

    st.success(f"Recognized {worker['full_name']} with score {score:.2f}.")
    open_shift = storage.get_open_shift(worker["id"])

    if open_shift:
        st.info("Current status: checked in.")
        status = apply_geofence_status(worker["id"], lat, lon)
        st.write(status)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Manual check out", type="primary"):
                storage.check_out(worker["id"], lat, lon)
                st.success("Checked out.")
                st.rerun()
        with col_b:
            if st.button("Refresh geo-fence status"):
                st.rerun()
    else:
        fence = current_fence()
        inside, distance = is_inside_geofence(lat, lon, fence)
        if not inside:
            st.error(f"You are outside the farm geo-fence ({distance:.0f} m from center).")
            return
        if st.button("Check in", type="primary"):
            storage.check_in(worker["id"], lat, lon)
            st.success("Checked in.")
            st.rerun()


def shifts_dataframe() -> pd.DataFrame:
    if storage is None:
        return pd.DataFrame()

    shifts = storage.list_shifts()
    if not shifts:
        return pd.DataFrame()

    rows = []
    for shift in shifts:
        start = parse_dt(shift["checked_in_at"])
        end = parse_dt(shift["checked_out_at"]) or datetime.now(timezone.utc)
        if not start:
            continue
        hours = max(0.0, (end - start).total_seconds() / 3600)
        weekend_hours = hours if start.weekday() in (5, 6) else 0.0
        regular_hours = hours - weekend_hours
        rows.append(
            {
                "worker": shift["full_name"],
                "checked_in_at": start,
                "checked_out_at": end if shift["checked_out_at"] else None,
                "hours": hours,
                "regular_hours": regular_hours,
                "weekend_overtime_hours": weekend_hours,
                "hourly_rate": shift["hourly_rate"],
                "auto_checkout": bool(shift["auto_checkout"]),
            }
        )
    return pd.DataFrame(rows)


def manager_screen() -> None:
    if storage is None:
        st.error("Supabase storage is not available. Check the setup message at the top of the app.")
        return

    st.subheader("Manager")
    pin = st.text_input("Admin PIN", type="password")
    if pin != get_manager_pin():
        st.warning("Enter the manager PIN to continue.")
        return

    settings = storage.get_settings()
    st.markdown("#### Farm geo-fence")
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    with col_a:
        farm_lat = st.number_input("Center latitude", value=float(settings["farm_lat"]), format="%.6f")
    with col_b:
        farm_lon = st.number_input("Center longitude", value=float(settings["farm_lon"]), format="%.6f")
    with col_c:
        radius = st.number_input("Radius meters", min_value=25.0, value=float(settings["farm_radius_m"]), step=25.0)
    with col_d:
        grace = st.number_input("Grace minutes", min_value=1, value=int(float(settings["grace_minutes"])), step=1)
    with col_e:
        overtime_multiplier = st.number_input(
            "Overtime multiplier",
            min_value=1.0,
            value=float(settings["overtime_multiplier"]),
            step=0.25,
        )

    if st.button("Save geo-fence settings"):
        storage.update_settings(
            {
                "farm_lat": farm_lat,
                "farm_lon": farm_lon,
                "farm_radius_m": radius,
                "grace_minutes": grace,
                "overtime_multiplier": overtime_multiplier,
            }
        )
        st.success("Settings saved.")

    st.markdown("#### Geo-fence centre")
    render_geofence_map(farm_lat, farm_lon)

    if streamlit_geolocation is None:
        st.info("Install streamlit-geolocation to capture the browser location.")
    else:
        browser_location = streamlit_geolocation()
        parsed_location = extract_browser_location(browser_location)
        if parsed_location:
            browser_lat, browser_lon, accuracy = parsed_location
            accuracy_text = f" Accuracy: {accuracy:.0f} m." if accuracy is not None else ""
            st.write(
                f"Browser location: {browser_lat:.6f}, {browser_lon:.6f}.{accuracy_text}"
            )
            if st.button("Set geo-fence centre to browser location", type="primary"):
                storage.update_settings(
                    {
                        "farm_lat": browser_lat,
                        "farm_lon": browser_lon,
                        "farm_radius_m": radius,
                        "grace_minutes": grace,
                        "overtime_multiplier": overtime_multiplier,
                    }
                )
                st.success("Geo-fence centre updated from browser location.")
                st.rerun()
        else:
            st.caption("Use the location button above to capture this browser's current position.")

    workers = storage.list_workers(include_inactive=True)
    st.markdown("#### Workers")
    if workers:
        st.dataframe(pd.DataFrame(workers), use_container_width=True, hide_index=True)
    else:
        st.info("No workers enrolled yet.")

    st.markdown("#### Weekly summaries")
    df = shifts_dataframe()
    if df.empty:
        st.info("No shifts recorded yet.")
        return

    multiplier = float(storage.get_settings()["overtime_multiplier"])
    df["week"] = df["checked_in_at"].dt.strftime("%Y-%U")
    df["regular_pay"] = df["regular_hours"] * df["hourly_rate"]
    df["overtime_pay"] = df["weekend_overtime_hours"] * df["hourly_rate"] * multiplier
    df["estimated_pay"] = df["regular_pay"] + df["overtime_pay"]

    summary = (
        df.groupby(["worker", "week"], as_index=False)
        .agg(
            total_hours=("hours", "sum"),
            average_shift_hours=("hours", "mean"),
            weekend_overtime_hours=("weekend_overtime_hours", "sum"),
            estimated_salary=("estimated_pay", "sum"),
        )
        .sort_values(["week", "worker"], ascending=[False, True])
    )
    summary["estimated_salary"] = summary["estimated_salary"].map(money)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("#### Shift log")
    display_df = df.copy()
    display_df["estimated_pay"] = display_df["estimated_pay"].map(money)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("MilkingTime")
    if storage_error:
        st.error(
            "Supabase storage is required, but the app could not connect. "
            "Add the Streamlit secrets and run supabase_schema.sql in Supabase, then restart Streamlit."
        )
        st.code(storage_error)
    elif supabase_configured(supabase_secrets):
        st.caption("Connected to Supabase using Streamlit secrets.")

    tab_clock, tab_enroll, tab_manager = st.tabs(["Clock", "Enroll", "Manager"])
    with tab_clock:
        worker_clock_screen()
    with tab_enroll:
        enrollment_screen()
    with tab_manager:
        manager_screen()


if __name__ == "__main__":
    main()
