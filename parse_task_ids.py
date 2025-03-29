import json
from pathlib import Path

def parse_task_ids():
    # Read the JSON file
    results_path = Path("results/mind2web_results.json")
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Extract task IDs
    task_ids = [item["task_id"] for item in data]
    
    # Print total count and task IDs
    print(f"Found {len(task_ids)} task IDs")
    print("Task IDs:", task_ids)
    
    return task_ids

if __name__ == "__main__":
    parse_task_ids() 


def parse_webarena_task_ids():
    """
    Parse WebArena task IDs from the results JSON file.
    The JSON structure contains an array of task objects with task_id field.
    """
    results_path = Path("results/webarena_results.json")
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Extract task IDs
    task_ids = [item["task_id"] for item in data]
    
    # Print total count and task IDs
    print(f"Found {len(task_ids)} WebArena task IDs")
    print("WebArena Task IDs:", task_ids)
    
    return task_ids


def parse_webvoyager_task_ids():
    """
    """
