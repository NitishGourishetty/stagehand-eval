import json
from pathlib import Path

def parse_webarena_task_ids():
    """Parse task IDs from WebArena results file"""
    # Read the JSON file
    results_path = Path("results/webarena_results.json")
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Extract task IDs - WebArena uses numeric IDs starting from 0
    task_ids = []
    for item in data:
        if "task_id" in item:
            task_ids.append(item["task_id"])
    
    # Print total count and task IDs
    print(f"Found {len(task_ids)} WebArena task IDs")
    print("WebArena Task IDs:", task_ids)
    
    return task_ids

if __name__ == "__main__":
    parse_webarena_task_ids() 