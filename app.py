import datetime
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # to load keys from .env

from scheduler_module import build_schedule  # SCHEDULER TEAM
from rag_module import ingest_documents, answer_question  # RAG TEAM
from location_module import (
    suggest_places_for_task,
    suggest_places_for_task_with_coords,
)  # LOCATION TEAM
from streamlit_geolocation import streamlit_geolocation


# =========================
# Session State Utilities
# =========================

def init_session_state():
    """Initialize all keys used in st.session_state."""
    if "tasks" not in st.session_state:
        # List[Dict[str, Any]]
        st.session_state.tasks = []
    if "schedule" not in st.session_state:
        # Dict[str, Any]
        st.session_state.schedule = {}
    if "ingested_docs" not in st.session_state:
        st.session_state.ingested_docs = False
    if "user_location" not in st.session_state:
        st.session_state.user_location = ""
    if "search_radius_km" not in st.session_state:
        st.session_state.search_radius_km = 2.0
    if "current_lat" not in st.session_state:
        st.session_state.current_lat = None
    if "current_lon" not in st.session_state:
        st.session_state.current_lon = None


# =========================
# Sidebar: Global Settings
# =========================

def render_sidebar():
    st.sidebar.title("TaskScapeAI 🧭")
    st.sidebar.markdown("Context-aware To-Do & Schedule Assistant")

    st.sidebar.subheader("Day Settings")
    st.session_state.day_start = st.sidebar.time_input(
        "Day starts at", datetime.time(8, 0)
    )
    st.session_state.day_end = st.sidebar.time_input(
        "Day ends at", datetime.time(22, 0)
    )

    st.sidebar.subheader("Preferences")
    st.session_state.default_task_duration = st.sidebar.number_input(
        "Default task duration (minutes)",
        min_value=15,
        max_value=240,
        value=60,
        step=15,
    )
    st.session_state.prefers_outside_global = st.sidebar.checkbox(
        "I generally like doing deep work outside (café/library)",
        value=False,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "RAG, Scheduler, and Location logic plug into clearly marked sections in this UI."
    )


# =========================
# Planner / Scheduler Tab
# =========================

def find_task_by_title(title: str) -> Dict[str, Any] | None:
    for t in st.session_state.tasks:
        if t["title"] == title:
            return t
    return None


def render_planner_tab():
    st.header("📅 Planner & Tasks")

    col_form, col_tasks, col_schedule = st.columns([1.1, 1.2, 1.5])

    # ---- Add Task Form ----
    with col_form:
        st.subheader("Add a Task")

        with st.form(key="task_form", clear_on_submit=True):
            title = st.text_input("Task title", placeholder="e.g., Study ML exam")
            description = st.text_area("Notes (optional)", height=80)
            duration_min = st.number_input(
                "Estimated duration (minutes)",
                min_value=15,
                max_value=300,
                value=st.session_state.default_task_duration,
                step=15,
            )
            deadline = st.date_input(
                "Deadline (optional)",
                value=None,
                format="YYYY-MM-DD",
            )
            priority = st.selectbox("Priority", ["Low", "Medium", "High"], index=1)
            task_type = st.selectbox(
                "Task type", ["Study", "Project", "Admin", "Meeting", "Other"]
            )
            prefer_outside = st.checkbox(
                "Better done outside (café/library)?",
                value=st.session_state.prefers_outside_global,
            )

            submitted = st.form_submit_button("Add task")

        if submitted:
            if not title.strip():
                st.warning("Task must have a title.")
            else:
                st.session_state.tasks.append(
                    {
                        "id": len(st.session_state.tasks) + 1,
                        "title": title.strip(),
                        "description": description.strip(),
                        "duration_min": int(duration_min),
                        "deadline": str(deadline) if deadline else None,
                        "priority": priority,
                        "task_type": task_type,
                        "prefer_outside": prefer_outside,
                        "completed": False,
                    }
                )
                st.success(f"Task '{title}' added.")

    # ---- Task List with inline editing ----
    with col_tasks:
        st.subheader("Tasks")

        if not st.session_state.tasks:
            st.info("No tasks yet. Add one on the left.")
        else:
            for i, task in enumerate(list(st.session_state.tasks)):
                with st.expander(
                    f"{task['title']}  ({task['priority']})", expanded=False
                ):
                    st.markdown("**Edit task**")

                    new_title = st.text_input(
                        "Title",
                        value=task["title"],
                        key=f"title_{task['id']}",
                    )
                    new_desc = st.text_area(
                        "Notes",
                        value=task["description"],
                        key=f"desc_{task['id']}",
                        height=80,
                    )
                    new_duration = st.number_input(
                        "Duration (minutes)",
                        min_value=15,
                        max_value=300,
                        value=task["duration_min"],
                        step=15,
                        key=f"dur_{task['id']}",
                    )
                    new_deadline = st.text_input(
                        "Deadline (YYYY-MM-DD or empty)",
                        value=task["deadline"] or "",
                        key=f"dead_{task['id']}",
                    )
                    priority_options = ["Low", "Medium", "High"]
                    new_priority = st.selectbox(
                        "Priority",
                        priority_options,
                        index=priority_options.index(task["priority"]),
                        key=f"prio_{task['id']}",
                    )
                    type_options = ["Study", "Project", "Admin", "Meeting", "Other"]
                    new_type = st.selectbox(
                        "Task type",
                        type_options,
                        index=type_options.index(task["task_type"]),
                        key=f"type_{task['id']}",
                    )
                    new_outside = st.checkbox(
                        "Better done outside (café/library)?",
                        value=task["prefer_outside"],
                        key=f"outside_{task['id']}",
                    )

                    cols = st.columns(3)
                    with cols[0]:
                        new_completed = st.checkbox(
                            "Completed",
                            value=task["completed"],
                            key=f"completed_{task['id']}",
                        )
                        task["completed"] = new_completed

                    with cols[1]:
                        if st.button("Save changes", key=f"save_{task['id']}"):
                            task["title"] = new_title.strip() or task["title"]
                            task["description"] = new_desc.strip()
                            task["duration_min"] = int(new_duration)
                            task["deadline"] = new_deadline.strip() or None
                            task["priority"] = new_priority
                            task["task_type"] = new_type
                            task["prefer_outside"] = new_outside
                            st.success("Task updated.")

                    with cols[2]:
                        if st.button("Delete", key=f"delete_{task['id']}"):
                            st.session_state.tasks.pop(i)
                            st.experimental_rerun()

    # ---- Schedule Preview / Actions ----
    with col_schedule:
        st.subheader("Schedule")

        st.markdown(
            "This will show your generated schedule using the **Scheduler module**."
        )

        if st.button("Generate / Update Schedule"):
            if not st.session_state.tasks:
                st.warning("You have no tasks yet. Add tasks before scheduling.")
            else:
                st.session_state.schedule = build_schedule(
                    tasks=st.session_state.tasks,
                    day_start=st.session_state.day_start,
                    day_end=st.session_state.day_end,
                )

                st.success("Schedule generated!")

        if st.session_state.schedule:
            render_schedule_view(st.session_state.schedule)
        else:
            st.info("No schedule yet. Click **Generate / Update Schedule** to create one.")


def render_schedule_view(schedule_data: Dict[str, Any]):
    """
    Simple visual placeholder for the schedule.

    Expected format:
    {
        "days": {
            "YYYY-MM-DD": [
                {"time": "09:00-10:30", "task": "Study ML", "type": "Study"},
                ...
            ],
            ...
        }
    }
    """
    if "days" not in schedule_data or not schedule_data["days"]:
        st.write(schedule_data)
        return

    for day, blocks in schedule_data["days"].items():
        st.markdown(f"### {day}")
        for block in blocks:
            st.markdown(
                f"- `{block.get('time', '')}` — **{block.get('task', '')}** "
                f"({block.get('type', 'task')})"
            )


# =========================
# Docs & RAG Tab
# =========================

def render_docs_rag_tab():
    st.header("📄 Docs & RAG (Schedules, Syllabi, etc.)")

    st.markdown(
        "Upload PDFs / docs that describe your timetable, syllabi, or project deadlines. "
        "The RAG module will ingest these so the agent can understand your real constraints."
    )

    uploaded_files = st.file_uploader(
        "Upload one or more documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    cols = st.columns(2)
    with cols[0]:
        if st.button("Ingest documents into RAG"):
            if not uploaded_files:
                st.warning("Please upload at least one document.")
            else:
                num_chunks = ingest_documents(uploaded_files)
                st.session_state.ingested_docs = num_chunks > 0
                if num_chunks > 0:
                    st.success(f"Ingested {num_chunks} chunks into RAG.")
                else:
                    st.warning("No chunks were ingested. Check your documents.")

    with cols[1]:
        st.metric(
            "Documents ingested?",
            "Yes" if st.session_state.ingested_docs else "No",
        )

    st.markdown("---")
    st.subheader("Ask about your schedule / deadlines")

    query = st.text_input(
        "Ask a question about your documents",
        placeholder="e.g., When is my ML midterm? What deadlines are in the next two weeks?",
    )

    if st.button("Ask RAG"):
        if not st.session_state.ingested_docs:
            st.warning("You haven't ingested any documents yet.")
        elif not query.strip():
            st.warning("Please enter a question.")
        else:
            answer = answer_question(query)
            st.markdown("**Answer:**")
            st.write(answer)


# =========================
# Locations / Study Places Tab
# =========================

def render_locations_tab():
    st.header("📍 Study Places & Locations")

    st.markdown(
        "This tab helps you find suitable places (e.g., cafés, libraries) "
        "to complete certain tasks, based on your location and task type."
    )

    # --------- User Location & Radius ---------
    st.subheader("Your Location")

    col_geo, col_loc, col_radius = st.columns([1, 2, 1])

    with col_geo:
        st.markdown("**Use my current location**")
        geo = streamlit_geolocation()
        if geo and geo.get("latitude") is not None:
            st.session_state.current_lat = geo.get("latitude")
            st.session_state.current_lon = geo.get("longitude")
            st.caption(
                f"📍 Current location detected: {geo['latitude']:.4f}, {geo['longitude']:.4f}"
            )

    with col_loc:
        st.session_state.user_location = st.text_input(
            "Or type a location manually",
            value=st.session_state.user_location,
            placeholder="e.g., Byblos, Lebanon",
        )

    with col_radius:
        st.session_state.search_radius_km = st.slider(
            "Search radius (km)",
            min_value=0.5,
            max_value=10.0,
            value=float(st.session_state.search_radius_km),
            step=0.5,
        )

    st.markdown("---")

    if not st.session_state.tasks:
        st.info("You have no tasks yet. Add tasks in the Planner tab first.")
        return

    mode = st.radio(
        "Choose mode",
        ["For a specific task", "For today's scheduled tasks"],
        index=0,
        horizontal=True,
    )

    if mode == "For a specific task":
        render_location_mode_specific_task()
    else:
        render_location_mode_today_schedule()


def render_location_mode_specific_task():
    st.subheader("Mode: For a specific task")

    task_labels = [f"{t['id']}: {t['title']}" for t in st.session_state.tasks]
    selected_label = st.selectbox("Select a task", task_labels)
    selected_task_id = int(selected_label.split(":")[0])
    selected_task = next(t for t in st.session_state.tasks if t["id"] == selected_task_id)

    st.markdown(
        f"**Selected task:** {selected_task['title']}  "
        f"({selected_task['task_type']}, {selected_task['duration_min']} min)"
    )

    if st.button("Find places for this task"):
        has_text_loc = bool(st.session_state.user_location.strip())
        has_coords = (
            st.session_state.get("current_lat") is not None
            and st.session_state.get("current_lon") is not None
        )

        if not has_text_loc and not has_coords:
            st.warning("Please share your location (button) or type a city/area above.")
            return

        # Typed location has priority over GPS
        if has_text_loc:
            places = suggest_places_for_task(
                task=selected_task,
                user_location=st.session_state.user_location,
                radius_km=st.session_state.search_radius_km,
            )
        else:
            places = suggest_places_for_task_with_coords(
                task=selected_task,
                lat=st.session_state.current_lat,
                lon=st.session_state.current_lon,
                radius_km=st.session_state.search_radius_km,
            )

        if not places:
            st.warning("No places found near that location for this task.")
        else:
            first_name = (places[0].get("name") or "").lower()
            if "location not found" in first_name or "place search error" in first_name:
                st.warning(
                    places[0].get("notes", "There was an error searching for places.")
                )
            else:
                st.success("Places generated based on your location.")
            show_places_list(places)


def render_location_mode_today_schedule():
    st.subheader("Mode: For today's scheduled tasks")

    if not st.session_state.schedule or "days" not in st.session_state.schedule:
        st.info("No schedule found. Generate a schedule in the Planner tab first.")
        return

    today_str = str(datetime.date.today())
    day_blocks = st.session_state.schedule["days"].get(today_str, [])

    if not day_blocks:
        st.info("No tasks scheduled for today (or scheduler not yet integrated).")
        return

    st.markdown(f"### Tasks scheduled for today ({today_str})")

    for idx, block in enumerate(day_blocks):
        col_info, col_btn = st.columns([3, 1])
        with col_info:
            st.markdown(
                f"- `{block.get('time', '')}` — **{block.get('task', '')}** "
                f"({block.get('type', 'task')})"
            )
        with col_btn:
            if st.button("Suggest places", key=f"suggest_today_{idx}"):

                has_text_loc = bool(st.session_state.user_location.strip())
                has_coords = (
                    st.session_state.get("current_lat") is not None
                    and st.session_state.get("current_lon") is not None
                )

                if not has_text_loc and not has_coords:
                    st.warning(
                        "Please share your location (button) or type a city/area above."
                    )
                    continue

                task = find_task_by_title(block.get("task", ""))
                if task is None:
                    st.warning("Could not match this scheduled block to a task.")
                    continue

                if has_text_loc:
                    places = suggest_places_for_task(
                        task=task,
                        user_location=st.session_state.user_location,
                        radius_km=st.session_state.search_radius_km,
                    )
                else:
                    places = suggest_places_for_task_with_coords(
                        task=task,
                        lat=st.session_state.current_lat,
                        lon=st.session_state.current_lon,
                        radius_km=st.session_state.search_radius_km,
                    )

                if not places:
                    st.warning("No places found near that location for this task.")
                else:
                    st.success("Places generated for this scheduled task.")
                    show_places_list(places)


def show_places_list(places: List[Dict[str, Any]]):
    st.markdown("### Suggested Places")
    for p in places:
        name = p.get("name", "Unknown Place")
        dist = p.get("distance_km", "?")
        best_for = p.get("best_for", "N/A")
        notes = p.get("notes", "")

        lat = p.get("lat")
        lon = p.get("lon")
        if lat is not None and lon is not None:
            maps_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
            maps_link = f"[Open in Google Maps]({maps_url})"
        else:
            maps_link = ""

        st.markdown(
            f"**{name}** — {dist} km away  \n"
            f"*Best for:* {best_for}  \n"
            f"*Notes:* {notes}  \n"
            f"{maps_link}"
        )


# =========================
# Main App
# =========================

def main():
    st.set_page_config(
        page_title="TaskScapeAI 🧭",
        page_icon="🧭",
        layout="wide",
    )

    init_session_state()
    render_sidebar()

    tab_planner, tab_docs, tab_locations = st.tabs(
        ["📅 Planner", "📄 Docs & RAG", "📍 Study Places"]
    )

    with tab_planner:
        render_planner_tab()

    with tab_docs:
        render_docs_rag_tab()

    with tab_locations:
        render_locations_tab()


if __name__ == "__main__":
    main()
