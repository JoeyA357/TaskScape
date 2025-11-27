import datetime
from typing import List, Dict, Any

import streamlit as st

# Optional: once teammates create these files, you can uncomment the imports.
# For now, we keep them commented so the UI runs with placeholders.

# from scheduler_module import build_schedule          # SCHEDULER TEAM
# from rag_module import ingest_documents, answer_question  # RAG TEAM
# from location_module import suggest_places_for_task  # LOCATION TEAM


# =========================
# Session State Utilities
# =========================

def init_session_state():
    """Initialize all keys used in st.session_state."""
    if "tasks" not in st.session_state:
        st.session_state.tasks = []
    if "schedule" not in st.session_state:
        st.session_state.schedule = {}
    if "ingested_docs" not in st.session_state:
        st.session_state.ingested_docs = False
    if "user_location" not in st.session_state:
        st.session_state.user_location = ""
    if "search_radius_km" not in st.session_state:
        st.session_state.search_radius_km = 2.0


# =========================
# Sidebar: Global Settings
# =========================

def render_sidebar():
    st.sidebar.title("TaskScapeAI 🧭")
    st.sidebar.markdown("Context-aware To-Do & Schedule Assistant")

    st.sidebar.subheader("Day Settings")
    st.session_state.day_start = st.sidebar.time_input("Day starts at", datetime.time(8, 0))
    st.session_state.day_end = st.sidebar.time_input("Day ends at", datetime.time(22, 0))

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
        "RAG, Scheduler, and Location logic will plug into clearly marked sections in this UI."
    )


# =========================
# Planner / Scheduler Tab
# =========================

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
            task_type = st.selectbox("Task type", ["Study", "Project", "Admin", "Meeting", "Other"])
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

    # ---- Task List ----
    with col_tasks:
        st.subheader("Tasks")

        if not st.session_state.tasks:
            st.info("No tasks yet. Add one on the left.")
        else:
            for i, task in enumerate(st.session_state.tasks):
                with st.expander(f"{task['title']}  ({task['priority']})", expanded=False):
                    st.markdown(f"**Type:** {task['task_type']}")
                    st.markdown(f"**Duration:** {task['duration_min']} min")
                    if task["deadline"]:
                        st.markdown(f"**Deadline:** {task['deadline']}")
                    if task["description"]:
                        st.markdown(f"**Notes:** {task['description']}")
                    st.markdown(
                        f"**Prefer outside:** {'Yes' if task['prefer_outside'] else 'No'}"
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
                # === SCHEDULER TEAM: integrate your build_schedule() here ===
                #
                # Expected signature (you can adjust slightly, but keep the idea):
                # from scheduler_module import build_schedule
                #
                # schedule = build_schedule(
                #     tasks=st.session_state.tasks,
                #     day_start=st.session_state.day_start,
                #     day_end=st.session_state.day_end,
                # )
                #
                # st.session_state.schedule = schedule
                #
                # For now, we use a placeholder:
                st.session_state.schedule = {
                    "info": "Scheduler not yet connected. This is a placeholder.",
                    "days": {},
                }
                st.success("Schedule generation triggered (placeholder).")

        if st.session_state.schedule:
            render_schedule_view(st.session_state.schedule)
        else:
            st.info("No schedule yet. Click **Generate / Update Schedule** to create one.")


def render_schedule_view(schedule_data: Dict[str, Any]):
    """
    Simple visual placeholder for the schedule.

    SCHEDULER TEAM:
    ---------------
    If you return a dict of the form:
    {
        "days": {
            "2025-11-27": [
                {"time": "09:00-10:30", "task": "Study ML", "type": "Study"},
                ...
            ],
            ...
        }
    }

    this renderer will show it nicely.
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
                # === RAG TEAM: call your ingest_documents() function here ===
                #
                # from rag_module import ingest_documents
                # ingest_documents(uploaded_files)
                #
                # For now, we just set a flag:
                st.session_state.ingested_docs = True
                st.success("Documents marked as ingested (placeholder).")

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
            # === RAG TEAM: call your answer_question() function here ===
            #
            # from rag_module import answer_question
            # answer = answer_question(query)
            #
            # For now, we show a placeholder:
            answer = (
                "RAG answer placeholder. "
                "Once connected, this will show answers derived from your uploaded docs."
            )
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

    col_loc, col_radius = st.columns([2, 1])
    with col_loc:
        st.session_state.user_location = st.text_input(
            "Enter your location",
            value=st.session_state.user_location,
            placeholder="e.g., LAU Beirut, Hamra, Brummana...",
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
        if not st.session_state.user_location.strip():
            st.warning("Please enter your location above.")
        else:
            # === LOCATION TEAM: call your suggest_places_for_task() here ===
            #
            # from location_module import suggest_places_for_task
            # places = suggest_places_for_task(
            #     task=selected_task,
            #     user_location=st.session_state.user_location,
            #     radius_km=st.session_state.search_radius_km,
            # )
            #
            # For now, we mock a couple of places:
            places = [
                {
                    "name": "Quiet Beans Café",
                    "distance_km": 0.8,
                    "best_for": "deep focus / study",
                    "notes": "Usually quiet before 5pm, good Wi-Fi.",
                },
                {
                    "name": "Central Library",
                    "distance_km": 1.3,
                    "best_for": "long study sessions",
                    "notes": "Silent floor, many outlets.",
                },
            ]

            st.success("Places generated (placeholder).")
            show_places_list(places)


def render_location_mode_today_schedule():
    st.subheader("Mode: For today's scheduled tasks")

    if not st.session_state.schedule or "days" not in st.session_state.schedule:
        st.info("No schedule found. Generate a schedule in the Planner tab first.")
        return

    today_str = str(datetime.date.today())
    day_blocks = st.session_state.schedule["days"].get(today_str, [])

    # Filter for tasks that make sense to do outside (based on 'prefer_outside' flag if you include it in schedule)
    if not day_blocks:
        st.info("No tasks scheduled for today (or scheduler not yet integrated).")
        return

    st.markdown(f"### Tasks scheduled for today ({today_str})")

    # Simple list of today's tasks with a button per task
    for idx, block in enumerate(day_blocks):
        col_info, col_btn = st.columns([3, 1])
        with col_info:
            st.markdown(
                f"- `{block.get('time', '')}` — **{block.get('task', '')}** "
                f"({block.get('type', 'task')})"
            )
        with col_btn:
            if st.button("Suggest places", key=f"suggest_today_{idx}"):
                if not st.session_state.user_location.strip():
                    st.warning("Please enter your location above.")
                else:
                    # === LOCATION TEAM: you may want a slightly different call here ===
                    #
                    # For example, if your scheduler includes the original task_id,
                    # you can map back to the full task dict and call suggest_places_for_task().
                    #
                    # from location_module import suggest_places_for_task
                    # places = suggest_places_for_task(
                    #     task=...,  # matched from block
                    #     user_location=st.session_state.user_location,
                    #     radius_km=st.session_state.search_radius_km,
                    # )
                    #
                    # Placeholder mock:
                    places = [
                        {
                            "name": "Today Spot Café",
                            "distance_km": 0.9,
                            "best_for": "this type of task",
                            "notes": "Good for the scheduled time slot.",
                        }
                    ]
                    st.success("Places generated for this scheduled task (placeholder).")
                    show_places_list(places)


def show_places_list(places: List[Dict[str, Any]]):
    st.markdown("### Suggested Places")
    for p in places:
        st.markdown(
            f"**{p.get('name', 'Unknown Place')}** — {p.get('distance_km', '?')} km away  \n"
            f"*Best for:* {p.get('best_for', 'N/A')}  \n"
            f"*Notes:* {p.get('notes', '')}"
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
