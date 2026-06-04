from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pandas as pd
import streamlit as st

try:
    from streamlit_geolocation import streamlit_geolocation
except ImportError:
    streamlit_geolocation = None

from face_engine import average_embeddings, best_match, image_to_embedding, ranked_matches
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
def init_storage(supabase_url: str | None, supabase_key: str | None, storage_version: int):
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
        2,
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


def render_header() -> None:
    st.markdown(
        """
        <style>
            .milkingtime-header {
                margin: 0.25rem 0 1.4rem;
                padding: 1rem 2.2rem;
                border-radius: 1.35rem;
                background:
                    linear-gradient(105deg, #0d1824 0%, #122231 63%, #435160 100%);
                box-shadow:
                    0 18px 32px rgba(15, 23, 42, 0.18),
                    inset 0 1px 0 rgba(255, 255, 255, 0.08);
                color: #f8fafc;
                transition:
                    transform 180ms ease,
                    box-shadow 180ms ease,
                    filter 180ms ease;
                cursor: default;
            }

            .milkingtime-header:hover {
                transform: translateY(8px);
                box-shadow:
                    0 7px 14px rgba(15, 23, 42, 0.2),
                    inset 0 2px 10px rgba(0, 0, 0, 0.2);
                filter: brightness(0.96);
            }

            .milkingtime-header__eyebrow {
                margin: 0 0 0.25rem;
                font-size: 0.9rem;
                font-weight: 800;
                letter-spacing: 0.08rem;
                text-transform: uppercase;
                color: rgba(248, 250, 252, 0.88);
            }

            .milkingtime-header__title {
                margin: 0;
                font-size: clamp(2.2rem, 5vw, 4.2rem);
                line-height: 0.95;
                font-weight: 900;
                letter-spacing: 0;
                color: #ffffff;
            }

            .milkingtime-header__subtitle {
                margin: 0.45rem 0 0;
                max-width: 70rem;
                font-size: 1.05rem;
                line-height: 1.45;
                font-weight: 400;
                color: rgba(248, 250, 252, 0.9);
            }
        </style>
        <section class="milkingtime-header">
            <p class="milkingtime-header__eyebrow">Eenheid Farm</p>
            <h1 class="milkingtime-header__title">MilkingTime</h1>
            <p class="milkingtime-header__subtitle">
                Face-verified worker sign-in, geo-fenced shift tracking, and manager payroll visibility powered by Python and AI.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def reset_camera_inputs() -> None:
    st.session_state["camera_reset"] = st.session_state.get("camera_reset", 0) + 1


def replace_worker_embedding(worker_id: str, embedding: list[float]) -> None:
    if hasattr(storage, "replace_worker_embedding"):
        storage.replace_worker_embedding(worker_id, embedding)
        return

    storage.client.table("face_embeddings").delete().eq("worker_id", worker_id).execute()
    storage.client.table("face_embeddings").insert(
        {
            "id": str(uuid4()),
            "worker_id": worker_id,
            "embedding": embedding,
            "created_at": utc_now_iso(),
        }
    ).execute()


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


def recognize_from_camera(label: str) -> tuple[dict | None, float, list[float] | None, list[dict]]:
    reset_id = st.session_state.get("camera_reset", 0)
    photo = st.camera_input(label, key=f"worker_face_scan_{reset_id}")
    if not photo:
        return None, 0.0, None, []

    embedding = image_to_embedding(photo.getvalue())
    if storage is None:
        return None, 0.0, embedding, []
    known_workers = storage.list_worker_embeddings()
    worker, score = best_match(embedding, known_workers)
    matches = ranked_matches(embedding, known_workers)
    return worker, score, embedding, matches


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


def render_worker_shift_action(worker: dict, lat: float, lon: float) -> None:
    open_shift = storage.get_open_shift(worker["id"])

    if open_shift:
        st.info(f"{worker['full_name']} is currently signed in.")
        fence = current_fence()
        inside, distance = is_inside_geofence(lat, lon, fence)
        status = apply_geofence_status(worker["id"], lat, lon)
        st.write(status)
        if not inside:
            st.error(
                f"You are outside the farm geo-fence ({distance:.0f} m from center). "
                "Sign out is only available inside the farm geo-fence."
            )
            return
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button(f"Sign out {worker['full_name']}", type="primary"):
                storage.check_out(worker["id"], lat, lon)
                reset_camera_inputs()
                st.success(f"{worker['full_name']} signed out.")
                st.rerun()
        with col_b:
            if st.button("Refresh geo-fence status"):
                st.rerun()
    else:
        st.info(f"{worker['full_name']} is not currently signed in.")
        fence = current_fence()
        inside, distance = is_inside_geofence(lat, lon, fence)
        if not inside:
            st.error(f"You are outside the farm geo-fence ({distance:.0f} m from center).")
            return
        if st.button(f"Sign in {worker['full_name']}", type="primary"):
            storage.check_in(worker["id"], lat, lon)
            reset_camera_inputs()
            st.success(f"{worker['full_name']} signed in.")
            st.rerun()


def enrollment_screen() -> None:
    if storage is None:
        st.error("Supabase storage is not available. Check the setup message at the top of the app.")
        return

    st.subheader("Worker enrollment")
    st.caption("Capture three face images so the app can build one recognition vector.")

    mode = st.radio(
        "Enrollment type",
        ["New worker", "Re-enroll existing worker"],
        horizontal=True,
    )

    if mode == "New worker":
        with st.form("worker_details"):
            full_name = st.text_input("Full name")
            phone = st.text_input("Phone number")
            submitted = st.form_submit_button("Save details")

        if submitted:
            if not full_name.strip():
                st.error("Please enter a worker name.")
                return
            st.session_state["enrolling_worker"] = {
                "mode": "new",
                "full_name": full_name.strip(),
                "phone": phone.strip(),
                "embeddings": [],
            }
    else:
        workers = storage.list_workers(include_inactive=False)
        if not workers:
            st.info("No active workers are available to re-enroll.")
            return

        worker_options = {
            f"{worker['full_name']} ({worker.get('phone') or worker['id'][:8]})": worker
            for worker in workers
        }
        selected_name = st.selectbox("Worker to re-enroll", list(worker_options.keys()))
        selected_worker = worker_options[selected_name]

        if st.button(f"Start re-enrollment for {selected_worker['full_name']}"):
            st.session_state["enrolling_worker"] = {
                "mode": "reenroll",
                "worker_id": selected_worker["id"],
                "full_name": selected_worker["full_name"],
                "phone": selected_worker.get("phone") or "",
                "embeddings": [],
            }
            reset_camera_inputs()
            st.rerun()

    details = st.session_state.get("enrolling_worker")
    if not details:
        return

    action = "Re-enrolling" if details.get("mode") == "reenroll" else "Enrolling"
    st.info(f"{action} {details['full_name']}. Capture {3 - len(details['embeddings'])} more image(s).")
    reset_id = st.session_state.get("camera_reset", 0)
    photo = st.camera_input("Enrollment face capture", key=f"enrollment_face_capture_{reset_id}")
    if photo and st.button("Use this capture"):
        details["embeddings"].append(image_to_embedding(photo.getvalue()))
        st.session_state["enrolling_worker"] = details
        reset_camera_inputs()
        st.rerun()

    if len(details["embeddings"]) >= 3:
        embedding = average_embeddings(details["embeddings"])
        if details.get("mode") == "reenroll":
            worker_id = details["worker_id"]
            replace_worker_embedding(worker_id, embedding)
            success_message = f"Re-enrollment complete for {details['full_name']}."
        else:
            worker_id = storage.create_worker(
                details["full_name"],
                details["phone"],
                0.0,
                embedding,
            )
            success_message = f"Enrollment complete. Worker ID: {worker_id}"
        del st.session_state["enrolling_worker"]
        reset_camera_inputs()
        st.success(success_message)


def worker_clock_screen() -> None:
    if storage is None:
        st.error("Supabase storage is not available. Check the setup message at the top of the app.")
        return

    st.subheader("Worker clock in / out")
    st.caption("Use face scan and current location to manage a shift.")

    col_a, col_b = st.columns(2)
    with col_a:
        lat = st.number_input(
            "Current latitude",
            value=float(storage.get_settings()["farm_lat"]),
            format="%.6f",
            disabled=True,
        )
    with col_b:
        lon = st.number_input(
            "Current longitude",
            value=float(storage.get_settings()["farm_lon"]),
            format="%.6f",
            disabled=True,
        )

    worker, score, _, matches = recognize_from_camera("Worker face scan")
    if worker:
        st.success(f"Recognized {worker['full_name']} with score {score:.2f}.")
        render_worker_shift_action(worker, lat, lon)
        return

    if not matches:
        if score:
            st.warning(f"No match found. Best score: {score:.2f}")
        return

    st.warning(f"Low confidence match. Best score: {score:.2f}. Select the correct worker below.")
    options = {
        f"{match['worker']['full_name']} - score {match['score']:.2f}": match["worker"]
        for match in matches
    }
    selected_label = st.selectbox("Closest matching workers", list(options.keys()))
    selected_worker = options[selected_label]
    render_worker_shift_action(selected_worker, lat, lon)


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
    with st.expander("Show geo-fence map"):
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

        with st.expander("Update worker hourly rates"):
            for worker_row in workers:
                col_name, col_rate, col_action = st.columns([3, 2, 1])
                with col_name:
                    st.write(worker_row["full_name"])
                with col_rate:
                    new_rate = st.number_input(
                        "Hourly rate",
                        min_value=0.0,
                        value=float(worker_row.get("hourly_rate") or 0),
                        step=5.0,
                        key=f"hourly_rate_{worker_row['id']}",
                        label_visibility="collapsed",
                    )
                with col_action:
                    if st.button("Save", key=f"save_rate_{worker_row['id']}"):
                        storage.update_worker_hourly_rate(worker_row["id"], new_rate)
                        st.success(f"Updated {worker_row['full_name']}.")
                        st.rerun()
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
    summary_display = summary.copy()
    summary_display["estimated_salary"] = summary_display["estimated_salary"].map(money)
    st.dataframe(summary_display, use_container_width=True, hide_index=True)

    st.markdown("#### Worker hour plots")
    plot_df = summary.copy()
    plot_df["worker_week"] = plot_df["worker"] + " | " + plot_df["week"]
    total_hours_chart = plot_df.set_index("worker_week")[["total_hours"]]
    overtime_chart = plot_df.set_index("worker_week")[["weekend_overtime_hours"]]

    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Total weekly hours")
        st.bar_chart(total_hours_chart)
    with col_b:
        st.caption("Weekend overtime hours")
        st.bar_chart(overtime_chart)

    worker_totals = (
        df.groupby("worker", as_index=False)
        .agg(
            total_hours=("hours", "sum"),
            weekend_overtime_hours=("weekend_overtime_hours", "sum"),
        )
        .sort_values("total_hours", ascending=False)
    )
    st.caption("Total hours by worker")
    st.bar_chart(worker_totals.set_index("worker"))

    st.markdown("#### Shift log")
    display_df = df.copy()
    display_df["estimated_pay"] = display_df["estimated_pay"].map(money)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def main() -> None:
    render_header()
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
