# scheduler_module.py

from datetime import time, date
from typing import List, Dict, Any

def build_schedule(
    tasks: List[Dict[str, Any]],
    day_start: time,
    day_end: time,
) -> Dict[str, Any]:
    """
    Inputs:
        tasks: list of task dicts from st.session_state.tasks
        day_start, day_end: working hours

    Output format expected by UI:
    {
        "days": {
            "YYYY-MM-DD": [
                {"time": "HH:MM-HH:MM", "task": "Title", "type": "Study" },
                ...
            ],
            ...
        }
    }
    """
    # TODO: implement scheduling logic here
    from datetime import datetime, timedelta, date

    # 1) Convert priorities → numeric for sorting
    def priority_value(p):
        return {"High": 0, "Medium": 1, "Low": 2}.get(p, 2)

    # 2) Convert deadlines to dates
    def parse_deadline(d):
        return datetime.strptime(d, "%Y-%m-%d").date() if d else date.today()

    enriched_tasks = []
    for t in tasks:
        enriched_tasks.append({
            **t,
            "deadline_date": parse_deadline(t.get("deadline")),
            "priority_rank": priority_value(t.get("priority")),
        })

    # 3) Sort tasks by: deadline → priority → shorter duration first
    enriched_tasks.sort(
        key=lambda t: (t["deadline_date"], t["priority_rank"], t["duration_min"])
    )

    # Working hours range
    day_start_dt = datetime.combine(date.today(), day_start)
    day_end_dt = datetime.combine(date.today(), day_end)
    daily_minutes = int((day_end_dt - day_start_dt).total_seconds() // 60)

    # Prepare schedule output
    schedule = {"days": {}}

    # Start scheduling from today
    current_day = date.today()
    current_start_dt = datetime.combine(current_day, day_start)
    remaining_minutes_today = daily_minutes

    # 4) Fill tasks in order
    for t in enriched_tasks:
        duration = t["duration_min"]

        # If it doesn't fit today → move to next day
        while duration > remaining_minutes_today:
            current_day += timedelta(days=1)
            current_start_dt = datetime.combine(current_day, day_start)
            remaining_minutes_today = daily_minutes

        # Block start / end
        block_start = current_start_dt
        block_end = block_start + timedelta(minutes=duration)

        # Save to schedule
        day_key = str(current_day)
        if day_key not in schedule["days"]:
            schedule["days"][day_key] = []

        schedule["days"][day_key].append({
            "time": f"{block_start.strftime('%H:%M')}-{block_end.strftime('%H:%M')}",
            "task": t["title"],
            "type": t["task_type"],
        })

        # Move pointer forward
        current_start_dt = block_end
        remaining_minutes_today -= duration

    return schedule
