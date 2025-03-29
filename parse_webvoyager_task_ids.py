import json
from pathlib import Path

def parse_webvoyager_task_ids():
    """Parse task IDs from WebVoyager results file"""
    # Read the JSON file
    results_path = Path("results/webvoyager_results.json")
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Extract task IDs
    task_ids = [item["task_id"] for item in data]
    
    # Print total count and task IDs
    print(f"Found {len(task_ids)} WebVoyager task IDs")
    print("WebVoyager Task IDs:", task_ids)
    
    return task_ids

if __name__ == "__main__":
    parse_webvoyager_task_ids() 