# scheduler_module.py

from datetime import time, date, datetime, timedelta
from typing import List, Dict, Any, Tuple

def build_schedule(
    tasks: List[Dict[str, Any]],
    day_start: time,
    day_end: time,
) -> Dict[str, Any]:
    """
    Inputs:
        tasks: list of task dicts from st.session_state.tasks
        day_start, day_end: working hours

    Output format:
    {
        "days": {
            "YYYY-MM-DD": [
                {
                    "time": "HH:MM-HH:MM", 
                    "task": "Title", 
                    "type": "Study",
                    "start_time": time_obj,
                    "end_time": time_obj,
                    "duration_min": int
                },
                ...
            ],
            ...
        },
        "week_start": "YYYY-MM-DD",
        "week_end": "YYYY-MM-DD"
    }
    """
    
    # 1) Convert priorities → numeric for sorting
    def priority_value(p):
        return {"High": 0, "Medium": 1, "Low": 2}.get(p, 2)

    # 2) Convert deadlines to dates
    def parse_deadline(d):
        return datetime.strptime(d, "%Y-%m-%d").date() if d else date.today() + timedelta(days=7)

    enriched_tasks = []
    for t in tasks:
        if t.get("completed"):
            continue  # Skip completed tasks
            
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

    # Prepare schedule output - start from next Monday for a full week view
    today = date.today()
    days_until_monday = (7 - today.weekday()) % 7  # 0 = Monday
    if days_until_monday == 0:
        days_until_monday = 0  # If today is Monday, start today
    
    week_start = today + timedelta(days=days_until_monday)
    week_end = week_start + timedelta(days=6)  # Full week

    schedule = {
        "days": {},
        "week_start": str(week_start),
        "week_end": str(week_end)
    }

    # Initialize all days of the week
    for i in range(7):
        day = week_start + timedelta(days=i)
        schedule["days"][str(day)] = []

    # Start scheduling from the week start
    current_day = week_start
    current_start_dt = datetime.combine(current_day, day_start)
    remaining_minutes_today = daily_minutes

    # 4) Fill tasks in order
    for t in enriched_tasks:
        duration = t["duration_min"]

        # If it doesn't fit today → move to next day
        while duration > remaining_minutes_today:
            current_day += timedelta(days=1)
            
            # Skip to next week if we exceed current week
            if current_day > week_end:
                current_day = week_start + timedelta(days=7)
                week_end = current_day + timedelta(days=6)
                
                # Add new week days to schedule
                for i in range(7):
                    day = current_day + timedelta(days=i)
                    if str(day) not in schedule["days"]:
                        schedule["days"][str(day)] = []
            
            current_start_dt = datetime.combine(current_day, day_start)
            remaining_minutes_today = daily_minutes

        # Block start / end
        block_start = current_start_dt
        block_end = block_start + timedelta(minutes=duration)

        # Save to schedule
        day_key = str(current_day)
        
        schedule["days"][day_key].append({
            "time": f"{block_start.strftime('%H:%M')}-{block_end.strftime('%H:%M')}",
            "task": t["title"],
            "type": t["task_type"],
            "start_time": block_start.time(),
            "end_time": block_end.time(),
            "duration_min": duration,
            "priority": t.get("priority", "Medium"),
        })

        # Move pointer forward
        current_start_dt = block_end
        remaining_minutes_today -= duration

    return schedule


def time_to_minutes(t: time) -> int:
    """Convert time object to minutes since midnight."""
    return t.hour * 60 + t.minute


def get_time_slot_position(start_time: time, grid_start: time, grid_end: time) -> float:
    """Calculate the vertical position (0-1) of a time within the grid."""
    start_min = time_to_minutes(start_time)
    grid_start_min = time_to_minutes(grid_start)
    grid_end_min = time_to_minutes(grid_end)
    
    total_minutes = grid_end_min - grid_start_min
    offset_minutes = start_min - grid_start_min
    
    return max(0, min(1, offset_minutes / total_minutes))


def get_task_height(duration_min: int, grid_start: time, grid_end: time) -> float:
    """Calculate the height (0-1) of a task block."""
    grid_start_min = time_to_minutes(grid_start)
    grid_end_min = time_to_minutes(grid_end)
    total_minutes = grid_end_min - grid_start_min
    
    return min(1, duration_min / total_minutes)