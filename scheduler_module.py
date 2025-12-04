# scheduler_module.py - COMPLETE REPLACEMENT

from datetime import time, date, datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import json
import os

# Path to schedule blocks file (created by RAG)
DATA_DIR = "rag_data"
SCHEDULE_BLOCKS_FILE = os.path.join(DATA_DIR, "schedule_blocks.json")


def load_blocked_times() -> Dict[str, List[Dict[str, Any]]]:
    """Load blocked time slots from RAG-extracted schedule."""
    try:
        if os.path.exists(SCHEDULE_BLOCKS_FILE):
            with open(SCHEDULE_BLOCKS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading blocked times: {e}")
    return {}


def time_to_minutes(t: time) -> int:
    """Convert time object to minutes since midnight."""
    return t.hour * 60 + t.minute


def minutes_to_time(minutes: int) -> time:
    """Convert minutes since midnight to time object."""
    hours = minutes // 60
    mins = minutes % 60
    return time(hour=hours, minute=mins)


def parse_time_str(time_str: str) -> Optional[time]:
    """Parse time string HH:MM to time object."""
    try:
        parts = time_str.split(":")
        return time(hour=int(parts[0]), minute=int(parts[1]))
    except:
        return None


def get_day_name(date_obj: date) -> str:
    """Get lowercase day name from date object."""
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    return days[date_obj.weekday()]


def find_free_windows(
    day_date: date,
    day_start: time,
    day_end: time,
    blocked_times: Dict[str, List[Dict[str, Any]]]
) -> List[Tuple[time, time]]:
    """
    Find free time windows on a given day, avoiding blocked times.
    Returns list of (start_time, end_time) tuples.
    """
    day_name = get_day_name(day_date)
    
    # Get blocked slots for this day
    blocks = blocked_times.get(day_name, [])
    
    if not blocks:
        # Entire day is free
        return [(day_start, day_end)]
    
    # Convert everything to minutes
    day_start_min = time_to_minutes(day_start)
    day_end_min = time_to_minutes(day_end)
    
    # Parse and sort blocked times
    blocked_intervals = []
    for block in blocks:
        start_time = parse_time_str(block["start_time"])
        end_time = parse_time_str(block["end_time"])
        
        if start_time and end_time:
            blocked_intervals.append((
                time_to_minutes(start_time),
                time_to_minutes(end_time)
            ))
    
    # Sort by start time
    blocked_intervals.sort()
    
    # Find gaps between blocked times
    free_windows = []
    current_time = day_start_min
    
    for block_start, block_end in blocked_intervals:
        # If there's a gap before this block
        if current_time < block_start:
            free_windows.append((
                minutes_to_time(current_time),
                minutes_to_time(block_start)
            ))
        # Move past this block
        current_time = max(current_time, block_end)
    
    # Check if there's time after the last block
    if current_time < day_end_min:
        free_windows.append((
            minutes_to_time(current_time),
            minutes_to_time(day_end_min)
        ))
    
    return free_windows

def find_free_windows_for_day_blocks(
    day_blocks: List[Dict[str, Any]],
    day_start: time,
    day_end: time,
) -> List[Tuple[time, time]]:
    """
    Find free time windows on a given day from the *current* schedule blocks
    (classes + already scheduled tasks).

    day_blocks: list of blocks, each having at least 'start_time' and 'end_time' as
                datetime.time objects or "HH:MM" strings.
    Returns list of (start_time, end_time) tuples.
    """
    day_start_min = time_to_minutes(day_start)
    day_end_min = time_to_minutes(day_end)

    if not day_blocks:
        return [(day_start, day_end)]

    # Build occupied intervals in minutes
    intervals: List[Tuple[int, int]] = []
    for block in day_blocks:
        s = block.get("start_time")
        e = block.get("end_time")

        # If for some reason these are strings, parse them
        if isinstance(s, str):
            s = parse_time_str(s)
        if isinstance(e, str):
            e = parse_time_str(e)

        if not s or not e:
            continue

        s_min = max(time_to_minutes(s), day_start_min)
        e_min = min(time_to_minutes(e), day_end_min)
        if e_min <= s_min:
            continue

        intervals.append((s_min, e_min))

    if not intervals:
        return [(day_start, day_end)]

    # Merge intervals and find gaps
    intervals.sort()
    free_windows: List[Tuple[time, time]] = []
    current = day_start_min

    for s_min, e_min in intervals:
        if e_min <= current:
            current = max(current, e_min)
            continue

        if s_min > current:
            free_windows.append(
                (minutes_to_time(current), minutes_to_time(s_min))
            )

        current = max(current, e_min)

    if current < day_end_min:
        free_windows.append(
            (minutes_to_time(current), minutes_to_time(day_end_min))
        )

    return free_windows



def can_fit_task(
    window_start: time,
    window_end: time,
    task_duration_min: int
) -> Optional[Tuple[time, time]]:
    """
    Check if a task can fit in a time window.
    Returns (task_start, task_end) if it fits, None otherwise.
    """
    window_start_min = time_to_minutes(window_start)
    window_end_min = time_to_minutes(window_end)
    window_duration = window_end_min - window_start_min
    
    if window_duration >= task_duration_min:
        task_end_min = window_start_min + task_duration_min
        return (window_start, minutes_to_time(task_end_min))
    
    return None


def build_schedule(
    tasks: List[Dict[str, Any]],
    day_start: time,
    day_end: time,
    use_blocked_times: bool = False,
) -> Dict[str, Any]:
    """
    Build a weekly schedule.

    Behaviour:
      - Avoids class times (from schedule_blocks.json) when use_blocked_times=True.
      - Never overlaps tasks with each other or with classes.
      - Respects deadlines as much as possible (tasks are placed on or before
        their deadline date within the current week).
      - High-priority tasks are treated before lower-priority ones.
      - Distributes workload across days instead of packing everything on Monday.
    """

    # -------------------------
    # 0) Load blocked (class) times
    # -------------------------
    blocked_times: Dict[str, List[Dict[str, Any]]] = {}
    if use_blocked_times:
        blocked_times = load_blocked_times()
        if not blocked_times:
            print("Warning: use_blocked_times=True but no blocked times found")

    # -------------------------
    # 1) Enrich tasks with deadline & priority
    # -------------------------
    def priority_value(p: Optional[str]) -> int:
        return {"High": 0, "Medium": 1, "Low": 2}.get(p, 2)

    def parse_deadline(d: Optional[str]) -> date:
        # If no deadline, assume 7 days from today
        if not d:
            return date.today() + timedelta(days=7)
        try:
            return datetime.strptime(d, "%Y-%m-%d").date()
        except Exception:
            # Fallback: treat as 7 days from today if parsing fails
            return date.today() + timedelta(days=7)

    enriched_tasks: List[Dict[str, Any]] = []
    for t in tasks:
        if t.get("completed"):
            continue
        enriched_tasks.append(
            {
                **t,
                "deadline_date": parse_deadline(t.get("deadline")),
                "priority_rank": priority_value(t.get("priority")),
            }
        )

    # Sort by: earlier deadline → higher priority → shorter duration
    enriched_tasks.sort(
        key=lambda t: (t["deadline_date"], t["priority_rank"], t["duration_min"])
    )

    # -------------------------
    # 2) Week range (Monday–Sunday of current week)
    # -------------------------
    today = date.today()
    week_start = today - timedelta(days=today.weekday())  # Monday
    week_end = week_start + timedelta(days=6)

    schedule: Dict[str, Any] = {
        "days": {},
        "week_start": str(week_start),
        "week_end": str(week_end),
        "using_blocked_times": use_blocked_times,
    }

    # Minutes per day in working window
    workday_total_min = time_to_minutes(day_end) - time_to_minutes(day_start)

    # Track load per day
    class_minutes_by_date: Dict[str, int] = {}
    task_minutes_by_date: Dict[str, int] = {}

    # -------------------------
    # 3) Initialise days + add CLASS blocks
    # -------------------------
    for i in range(7):
        day = week_start + timedelta(days=i)
        day_key = str(day)
        schedule["days"][day_key] = []
        class_minutes_by_date[day_key] = 0
        task_minutes_by_date[day_key] = 0

        if use_blocked_times and blocked_times:
            day_name = get_day_name(day)  # "monday", ...
            for block in blocked_times.get(day_name, []):
                s = parse_time_str(block["start_time"])
                e = parse_time_str(block["end_time"])
                if not s or not e:
                    continue

                # Clip to working hours
                if s < day_start:
                    s = day_start
                if e > day_end:
                    e = day_end

                dur = time_to_minutes(e) - time_to_minutes(s)
                if dur <= 0:
                    continue

                class_minutes_by_date[day_key] += dur

                schedule["days"][day_key].append(
                    {
                        "time": f"{s.strftime('%H:%M')}-{e.strftime('%H:%M')}",
                        "task": block.get("course", "Class"),
                        "type": "Class",
                        "start_time": s,
                        "end_time": e,
                        "duration_min": dur,
                        "priority": "High",  # show as 'High' in legend (red)
                    }
                )

        # keep class blocks sorted by start_time
        schedule["days"][day_key].sort(
            key=lambda b: time_to_minutes(b["start_time"])
        )

    # -------------------------
    # 4) Place tasks (respect deadlines & spread load)
    # -------------------------
    for t in enriched_tasks:
        duration = int(t["duration_min"])
        if duration <= 0:
            continue

        deadline = t["deadline_date"]

        # Clamp deadline into the current week for scheduling
        effective_deadline = min(max(deadline, week_start), week_end)

        # Candidate days: from week_start to effective_deadline (inclusive)
                # Candidate days: from week_start to effective_deadline (inclusive)
                # Candidate days: from week_start to effective_deadline (inclusive)
        days_range: List[date] = []
        d = week_start
        while d <= effective_deadline:
            days_range.append(d)
            d += timedelta(days=1)

        # If deadline is before week_start, fall back to whole week
        if not days_range:
            d = week_start
            while d <= week_end:
                days_range.append(d)
                d += timedelta(days=1)

        # ---- Prefer earlier days before the deadline, but still consider load ----
        # Margin: how many days *before* the deadline we try to finish if possible.
        # If deadline is far from week_start -> use 2 days margin, otherwise 1 or 0.
        diff_days = (effective_deadline - week_start).days
        if diff_days >= 3:
            margin = 2
        elif diff_days >= 1:
            margin = 1
        else:
            margin = 0

        early_deadline = effective_deadline - timedelta(days=margin)

        # Early days: "safe zone" before the margin
        early_days = [d for d in days_range if d <= early_deadline]
        # Late days: very close to the deadline
        late_days = [d for d in days_range if d > early_deadline]

        # Load now counts classes fully (they do make you tired)
        def day_load(day_date: date) -> float:
            dk = str(day_date)
            return task_minutes_by_date.get(dk, 0) + class_minutes_by_date.get(dk, 0)

        early_days.sort(key=day_load)
        late_days.sort(key=day_load)

        # We will try early_days first, then late_days
        candidate_days = early_days + late_days



        scheduled = False

        for day_date in candidate_days:
            day_key = str(day_date)
            day_blocks = schedule["days"].get(day_key, [])

            # If the day is already very packed (>70% of work window), skip it
            total_busy_min = (
                class_minutes_by_date.get(day_key, 0)
                + task_minutes_by_date.get(day_key, 0)
            )
            if total_busy_min >= 0.7 * workday_total_min:
                continue

            free_windows = find_free_windows_for_day_blocks(
                day_blocks=day_blocks,
                day_start=day_start,
                day_end=day_end,
            )

            for window_start, window_end in free_windows:
                fit = can_fit_task(window_start, window_end, duration)
                if not fit:
                    continue

                task_start, task_end = fit

                schedule["days"][day_key].append(
                    {
                        "time": f"{task_start.strftime('%H:%M')}-{task_end.strftime('%H:%M')}",
                        "task": t["title"],
                        "type": t["task_type"],
                        "start_time": task_start,
                        "end_time": task_end,
                        "duration_min": duration,
                        "priority": t.get("priority", "Medium"),
                    }
                )

                # Keep blocks sorted
                schedule["days"][day_key].sort(
                    key=lambda b: time_to_minutes(b["start_time"])
                )

                task_minutes_by_date[day_key] += duration
                scheduled = True
                break

            if scheduled:
                break

        # If not scheduled before deadline, we currently leave it unscheduled.
        # You could add a second pass here to place it after the deadline if you prefer.

    return schedule




def time_to_minutes_helper(t: time) -> int:
    """Convert time object to minutes since midnight."""
    return t.hour * 60 + t.minute


def get_time_slot_position(start_time: time, grid_start: time, grid_end: time) -> float:
    """Calculate the vertical position (0-1) of a time within the grid."""
    start_min = time_to_minutes_helper(start_time)
    grid_start_min = time_to_minutes_helper(grid_start)
    grid_end_min = time_to_minutes_helper(grid_end)
    
    total_minutes = grid_end_min - grid_start_min
    offset_minutes = start_min - grid_start_min
    
    return max(0, min(1, offset_minutes / total_minutes))


def get_task_height(duration_min: int, grid_start: time, grid_end: time) -> float:
    """Calculate the height (0-1) of a task block."""
    grid_start_min = time_to_minutes_helper(grid_start)
    grid_end_min = time_to_minutes_helper(grid_end)
    total_minutes = grid_end_min - grid_start_min
    
    return min(1, duration_min / total_minutes)