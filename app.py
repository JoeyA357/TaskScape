import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv
load_dotenv() 


from scheduler_module import build_schedule, check_time_conflict          
from rag_module import ingest_documents, answer_question 
from location_module import (
    suggest_places_for_task,
    suggest_places_for_task_with_coords,
)  
from streamlit_geolocation import streamlit_geolocation


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
    if "current_lat" not in st.session_state:
        st.session_state.current_lat = None
    if "current_lon" not in st.session_state:
        st.session_state.current_lon = None


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


def render_planner_tab():
    st.header("📅 Planner & Tasks")

    col_form, col_tasks = st.columns([1, 1])

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

    with col_tasks:
        st.subheader("Tasks")

        if not st.session_state.tasks:
            st.info("No tasks yet. Add one on the left.")
        else:
            tasks_to_remove = []
            
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
                        
                        if new_completed and not task["completed"]:
                            tasks_to_remove.append(i)
                        
                        task["completed"] = new_completed

                    with cols[1]:
                        if st.button("Delete", key=f"delete_{task['id']}"):
                            st.session_state.tasks.pop(i)
                            st.rerun()
            
            if tasks_to_remove:
                for idx in sorted(tasks_to_remove, reverse=True):
                    task_title = st.session_state.tasks[idx]["title"]
                    st.session_state.tasks.pop(idx)
                    
                    if st.session_state.schedule and "days" in st.session_state.schedule:
                        for day_key, blocks in st.session_state.schedule["days"].items():
                            st.session_state.schedule["days"][day_key] = [
                                b for b in blocks if b.get("task") != task_title
                            ]
                
                st.success("✅ Completed task(s) removed!")
                st.rerun()

    st.markdown("---")

    st.subheader("Schedule")
    st.markdown("This will show your generated schedule using the **Scheduler module**.")

    import os
    schedule_blocks_file = os.path.join("rag_data", "schedule_blocks.json")
    has_blocked_times = os.path.exists(schedule_blocks_file)
    
    col_btn, col_check = st.columns([2, 2])
    
    with col_check:
        use_blocked_times = st.checkbox(
            "Avoid class/schedule times",
            value=False,
            disabled=not has_blocked_times,
            help="Schedule tasks around your class times (requires ingested schedule in RAG tab)"
        )
        
        if use_blocked_times and not has_blocked_times:
            st.warning("⚠️ No schedule found. Please ingest a schedule in the RAG tab first.")
        elif has_blocked_times and not use_blocked_times:
            st.info("💡 Tip: Check the box to avoid scheduling during class times")
    
    with col_btn:
        if st.button("Generate / Update Schedule", use_container_width=True):
            if not st.session_state.tasks:
                st.warning("You have no tasks yet. Add tasks before scheduling.")
            elif use_blocked_times and not has_blocked_times:
                st.error("Cannot use blocked times without an ingested schedule. Please upload a schedule in the RAG tab.")
            else:
                try:
                    st.session_state.schedule = build_schedule(
                        tasks=st.session_state.tasks,
                        day_start=st.session_state.day_start,
                        day_end=st.session_state.day_end,
                        use_blocked_times=True
                    )
                    
                    if use_blocked_times:
                        st.success("✅ Schedule generated! Tasks scheduled around your classes.")
                    else:
                        st.success("✅ Schedule generated!")
                except Exception as e:
                    st.error(f"Error generating schedule: {str(e)}")

    if st.session_state.schedule:
        if st.session_state.schedule.get("using_blocked_times"):
            st.info("📚 This schedule avoids your class times. Tasks are placed in free time slots.")
        
        render_schedule_view(st.session_state.schedule)
        
        st.markdown("---")
        render_editable_task_timings()
    else:
        st.info("No schedule yet. Click **Generate / Update Schedule** to create one.")


def render_schedule_view(schedule_data: Dict[str, Any]):
    """
    Display schedule in a continuous vertical layout:
    - One tall column per day.
    - Each task/class block is a single div whose top & height are proportional
      to its start/end times, so it visually stretches over the time range.
    - Class blocks get a consistent color per course across the week.
    """
    import datetime
    from datetime import time
    import streamlit as st

    if "days" not in schedule_data or not schedule_data["days"]:
        st.info("No schedule data available.")
        return

    week_start = datetime.datetime.strptime(
        schedule_data.get("week_start", str(datetime.date.today())),
        "%Y-%m-%d",
    ).date()

    DAY_START_HOUR = 8
    DAY_END_HOUR = 20
    slot_labels = [f"{h:02d}:00" for h in range(DAY_START_HOUR, DAY_END_HOUR + 1)]
    SLOT_HEIGHT_PX = 60  

    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

  
    course_palette = [
        "#dcefff",  
        "#ffcfef",  
        "#dbc7ff",  
        "#b2f2d5",  
        "#fbccc5",  
        "#d5dfb6",  
        "#c19cc8",  
        "#f3e5f5",  
        "#dcf5ff",  
        "#c4f8ff",  
    ]

    class_courses = []
    for day_blocks in schedule_data["days"].values():
        for block in day_blocks:
            if block.get("type") == "Class":
                title = block.get("task", "")
                if title and title not in class_courses:
                    class_courses.append(title)

    course_colors = {
        course: course_palette[i % len(course_palette)]
        for i, course in enumerate(class_courses)
    }

    # Helpers
    def _to_time(t):
        if isinstance(t, time):
            return t
        if isinstance(t, str):
            try:
                return datetime.datetime.strptime(t, "%H:%M").time()
            except Exception:
                return None
        return None

    def _minutes(t: time) -> int:
        return t.hour * 60 + t.minute

    day_start_min = DAY_START_HOUR * 60
    day_end_min = DAY_END_HOUR * 60

    st.markdown(
        f"""
    <style>
    .schedule-wrapper {{
        background-color: #151515;
        border-radius: 8px;
        border: 2px solid #444;
        padding: 10px;
        margin-top: 10px;
        color: #e0e0e0;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    .schedule-header-row {{
        display: grid;
        grid-template-columns: 80px repeat(6, 1fr);
        margin-bottom: 6px;
    }}

    .schedule-header-cell {{
        background-color: #2b2b2b;
        padding: 10px 6px;
        border-right: 1px solid #444;
        text-align: center;
        font-weight: 600;
        font-size: 13px;
    }}

    .schedule-header-cell small {{
        display: block;
        color: #999;
        font-size: 11px;
        font-weight: 400;
        margin-top: 2px;
    }}

    .schedule-body {{
        display: grid;
        grid-template-columns: 80px 1fr;  /* time + all days */
        gap: 0;
    }}

    .time-column {{
        display: flex;
        flex-direction: column;
        align-items: stretch;
    }}

    .time-slot-label {{
        height: {SLOT_HEIGHT_PX}px;
        border-top: 1px solid #333;
        border-right: 1px solid #444;
        font-size: 11px;
        color: #aaa;
        display: flex;
        justify-content: center;
        align-items: flex-start;
        padding-top: 6px;
    }}

    .days-container {{
        display: grid;
        grid-template-columns: repeat(6, 1fr);  /* 6 equal day columns */
        border-left: 1px solid #444;
    }}

    .day-column {{
        position: relative;
        height: {SLOT_HEIGHT_PX * (len(slot_labels) - 1)}px;
        border-left: 1px solid #444;
        border-right: 1px solid #444;
        background:
            repeating-linear-gradient(
                to bottom,
                #202020 0,
                #202020 1px,
                #161616 1px,
                #161616 {SLOT_HEIGHT_PX}px
            );
    }}

    .task-block {{
        position: absolute;
        left: 6px;
        right: 6px;
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        border-radius: 4px;
        padding: 6px 8px;
        font-size: 11px;
        line-height: 1.4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.4);
        overflow: hidden;
    }}

    /* Priority colors – mainly for tasks (Study, etc.) */
    .task-block.priority-High {{
        background-color: #ffcdd2;
        border-left-color: #f44336;
    }}
    .task-block.priority-Medium {{
        background-color: #ffe0b2;
        border-left-color: #ff9800;
    }}
    .task-block.priority-Low {{
        background-color: #c8e6c9;
        border-left-color: #4caf50;
    }}

    .task-title {{
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 2px;
        font-size: 12px;
    }}
    .task-time {{
        font-size: 10px;
        color: #424242;
        font-weight: 500;
    }}
    .task-type {{
        font-size: 10px;
        color: #666;
        font-style: italic;
        margin-top: 2px;
    }}

    .priority-legend {{
        display: flex;
        gap: 20px;
        margin-top: 15px;
        padding: 10px;
        background-color: #1e1e1e;
        border-radius: 6px;
        align-items: center;
    }}
    .legend-item {{
        display: flex;
        align-items: center;
        gap: 8px;
        color: #e0e0e0;
        font-size: 13px;
    }}
    .legend-color {{
        width: 16px;
        height: 16px;
        border-radius: 3px;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    html = ['<div class="schedule-wrapper">']

    html.append('<div class="schedule-header-row">')
    html.append('<div class="schedule-header-cell">Time</div>')
    for i, day in enumerate(days_of_week):
        day_date = week_start + datetime.timedelta(days=i)
        html.append(
            f'<div class="schedule-header-cell">{day}'
            f'<small>{day_date.strftime("%m/%d")}</small></div>'
        )
    html.append("</div>")  

    html.append('<div class="schedule-body">')

    html.append('<div class="time-column">')
    for label in slot_labels[:-1]:  
        html.append(f'<div class="time-slot-label">{label}</div>')
    html.append("</div>")  
    html.append('<div class="days-container">')

    for i in range(6):
        day_date = week_start + datetime.timedelta(days=i)
        day_key = str(day_date)
        blocks = schedule_data["days"].get(day_key, [])

        html.append('<div class="day-column">')

        for block in blocks:
            s = _to_time(block.get("start_time"))
            e = _to_time(block.get("end_time"))
            if not s or not e:
                try:
                    start_str, end_str = block["time"].split("-")
                    s = datetime.datetime.strptime(start_str.strip(), "%H:%M").time()
                    e = datetime.datetime.strptime(end_str.strip(), "%H:%M").time()
                except Exception:
                    continue

            start_min = max(_minutes(s), day_start_min)
            end_min = min(_minutes(e), day_end_min)
            if end_min <= start_min:
                continue

            top_px = (start_min - day_start_min) / 60 * SLOT_HEIGHT_PX
            height_px = max(
                SLOT_HEIGHT_PX, 
                (end_min - start_min) / 60 * SLOT_HEIGHT_PX,
            )

            priority = block.get("priority", "Medium")
            title = block.get("task", "")
            time_range = block.get("time", "")
            ttype = block.get("type", "")

            style = f"top:{top_px:.1f}px;height:{height_px:.1f}px;"

            if ttype == "Class":
                course_color = course_colors.get(title, "#e3f2fd")
                style += f"background-color:{course_color};border-left-color:{course_color};"

            html.append(
                f'<div class="task-block priority-{priority}" '
                f'style="{style}">'
                f'<div class="task-title">{title}</div>'
                f'<div class="task-time">{time_range}</div>'
                f'<div class="task-type">{ttype}</div>'
                f'</div>'
            )

        html.append("</div>")  

    html.append("</div>")  
    html.append("</div>")  
    html.append("</div>")  

    st.markdown("\n".join(html), unsafe_allow_html=True)

    st.markdown(
        """
    <div class="priority-legend">
        <span style="color: #aaa; font-weight: 600; margin-right: 10px;">Priority:</span>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #f44336;"></div>
            <span>High (tasks)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #ff9800;"></div>
            <span>Medium (tasks)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #4caf50;"></div>
            <span>Low (tasks)</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

def render_editable_task_timings():
    """Allow users to edit task timings and days in the generated schedule."""
    st.subheader("✏️ Edit Task Timings & Days")
    st.markdown("Adjust the timing and day of scheduled tasks below. Changes are validated against class times.")
    
    if "days" not in st.session_state.schedule:
        return
    
    task_blocks = []
    for day_key, blocks in st.session_state.schedule["days"].items():
        for idx, block in enumerate(blocks):
            if block.get("type") != "Class":
                task_blocks.append({
                    "day_key": day_key,
                    "block_idx": idx,
                    "block": block
                })
    
    if not task_blocks:
        st.info("No tasks in schedule to edit.")
        return
    
    week_start = datetime.datetime.strptime(
        st.session_state.schedule.get("week_start", str(datetime.date.today())),
        "%Y-%m-%d"
    ).date()
    
    week_days = []
    for i in range(7):
        day = week_start + datetime.timedelta(days=i)
        week_days.append((str(day), day.strftime("%A, %m/%d")))
    
    for i, task_info in enumerate(task_blocks):
        day_key = task_info["day_key"]
        block_idx = task_info["block_idx"]
        block = task_info["block"]
        
        current_time_str = block.get("time", "")
        task_name = block.get("task", "Task")
        task_type = block.get("type", "")
        priority = block.get("priority", "Medium")
        
        day_date = datetime.datetime.strptime(day_key, "%Y-%m-%d").date()
        day_name = day_date.strftime("%A, %m/%d")
        
        with st.expander(f"📝 {task_name} - {day_name} ({current_time_str})", expanded=False):
            st.markdown(f"**Task:** {task_name}")
            st.markdown(f"**Type:** {task_type} | **Priority:** {priority}")
            st.markdown(f"**Current schedule:** {day_name} at {current_time_str}")
            
            st.markdown("---")
            
            try:
                start_str, end_str = current_time_str.split("-")
                current_start = datetime.datetime.strptime(start_str.strip(), "%H:%M").time()
                current_end = datetime.datetime.strptime(end_str.strip(), "%H:%M").time()
            except:
                current_start = datetime.time(9, 0)
                current_end = datetime.time(10, 0)
            
            col1, col2 = st.columns(2)
            
            with col1:
                current_day_idx = [d[0] for d in week_days].index(day_key)
                new_day_key = st.selectbox(
                    "Select day",
                    options=[d[0] for d in week_days],
                    format_func=lambda x: dict(week_days)[x],
                    index=current_day_idx,
                    key=f"edit_day_{day_key}_{block_idx}_{i}"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)  
                day_changed = new_day_key != day_key
                if day_changed:
                    st.info(f"→ Moving to {dict(week_days)[new_day_key]}")
            
            col3, col4 = st.columns(2)
            
            with col3:
                new_start = st.time_input(
                    "Start time",
                    value=current_start,
                    key=f"edit_start_{day_key}_{block_idx}_{i}"
                )
            
            with col4:
                new_end = st.time_input(
                    "End time",
                    value=current_end,
                    key=f"edit_end_{day_key}_{block_idx}_{i}"
                )
            
            col5, col6 = st.columns([1, 3])
            
            with col5:
                if st.button("✅ Apply Changes", key=f"apply_{day_key}_{block_idx}_{i}", use_container_width=True):
                    if new_start >= new_end:
                        st.error("❌ Start time must be before end time!")
                    else:
                        exclude_idx = None if day_changed else block_idx
                        
                        conflict = check_time_conflict(
                            day_key=new_day_key,
                            new_start=new_start,
                            new_end=new_end,
                            schedule_data=st.session_state.schedule,
                            exclude_block_idx=exclude_idx
                        )
                        
                        if conflict:
                            conflict_type = "blocked time" if conflict["type"] == "Class" else "task"
                            st.error(f"🚫 **Time conflict with {conflict_type}: '{conflict['task']}'** ({conflict['time']})  \nPlease adjust the time again.")
                        else:
                            duration_min = int((datetime.datetime.combine(datetime.date.today(), new_end) - 
                                              datetime.datetime.combine(datetime.date.today(), new_start)).total_seconds() / 60)
                            
                            updated_block = block.copy()
                            updated_block.update({
                                "time": f"{new_start.strftime('%H:%M')}-{new_end.strftime('%H:%M')}",
                                "start_time": new_start,
                                "end_time": new_end,
                                "duration_min": duration_min
                            })
                            
                            if day_changed:
                                st.session_state.schedule["days"][day_key].pop(block_idx)
                                
                                if new_day_key not in st.session_state.schedule["days"]:
                                    st.session_state.schedule["days"][new_day_key] = []
                                st.session_state.schedule["days"][new_day_key].append(updated_block)
                                
                                st.session_state.schedule["days"][new_day_key].sort(
                                    key=lambda b: b["start_time"] if isinstance(b["start_time"], datetime.time) else datetime.time(0, 0)
                                )
                                
                                st.success(f"✅ Moved {task_name} to {dict(week_days)[new_day_key]} at {new_start.strftime('%H:%M')}-{new_end.strftime('%H:%M')}")
                            else:
                                st.session_state.schedule["days"][day_key][block_idx] = updated_block
                                st.success(f"✅ Updated {task_name} to {new_start.strftime('%H:%M')}-{new_end.strftime('%H:%M')}")
                            
                            st.rerun()
            
            with col6:
                if st.button("🗑️ Remove from Schedule", key=f"remove_{day_key}_{block_idx}_{i}", use_container_width=True):
                    st.session_state.schedule["days"][day_key].pop(block_idx)
                    st.success(f"✅ Removed {task_name} from schedule")
                    st.rerun()


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



def find_task_by_title(title: str) -> Optional[Dict[str, Any]]:
    """Find a task in the task list by its title."""
    for t in st.session_state.tasks:
        if t["title"] == title:
            return t
    return None

def render_locations_tab():
    st.header("📍 Study Places & Locations")

    st.markdown(
        "This tab helps you find suitable places (e.g., cafés, libraries) "
        "to complete certain tasks, based on your location and task type."
    )

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
        has_text_loc = bool(st.session_state.user_location.strip())
        has_coords = (
            st.session_state.get("current_lat") is not None
            and st.session_state.get("current_lon") is not None
        )

        if not has_text_loc and not has_coords:
            st.warning("Please share your location (button) or type a city/area above.")
            return

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

