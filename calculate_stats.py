import json
import os
from typing import Dict, List
import numpy as np

def load_results(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_stats(results: List[Dict]) -> Dict:
    accuracy_scores = [result.get('accuracy_score', 0) for result in results]
    total_tasks = len(results)
    successful_tasks = sum(1 for score in accuracy_scores if score > 0)
    
    stats = {
        'total_tasks': total_tasks,
        'successful_tasks': successful_tasks,
        'success_rate': (successful_tasks / total_tasks) * 100 if total_tasks > 0 else 0,
        'average_accuracy': np.mean(accuracy_scores) if accuracy_scores else 0,
        'median_accuracy': np.median(accuracy_scores) if accuracy_scores else 0,
        'std_accuracy': np.std(accuracy_scores) if accuracy_scores else 0,
        'min_accuracy': min(accuracy_scores) if accuracy_scores else 0,
        'max_accuracy': max(accuracy_scores) if accuracy_scores else 0,
    }
    
    # Calculate action statistics
    all_actions = []
    for result in results:
        if 'stagehand_actions' in result:
            all_actions.extend(result['stagehand_actions'])
    
    action_types = {}
    for action in all_actions:
        action_type = action.split(']')[0].strip('[') if ']' in action else 'unknown'
        action_types[action_type] = action_types.get(action_type, 0) + 1
    
    stats['action_types'] = action_types
    stats['total_actions'] = len(all_actions)
    stats['average_actions_per_task'] = len(all_actions) / total_tasks if total_tasks > 0 else 0
    
    return stats

def process_dataset(dataset: str, results_dir: str):
    versions = ['', '_filtered', '_reprocessed']
    
    for version in versions:
        file_path = os.path.join(results_dir, f'{dataset}_results{version}.json')
        if os.path.exists(file_path):
            results = load_results(file_path)
            stats = calculate_stats(results)
            
            # Save stats to file
            stats_file = os.path.join(results_dir, f'{dataset}_stats{version}.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Print summary
            version_name = "Raw" if not version else version[1:].capitalize()
            print(f"\n{dataset.upper()} {version_name} Statistics:")
            print(f"Total Tasks: {stats['total_tasks']}")
            print(f"Success Rate: {stats['success_rate']:.2f}%")
            print(f"Average Accuracy: {stats['average_accuracy']:.2f}")
            print(f"Median Accuracy: {stats['median_accuracy']:.2f}")
            print(f"Standard Deviation: {stats['std_accuracy']:.2f}")
            print(f"Average Actions per Task: {stats['average_actions_per_task']:.2f}")
            print("\nAction Types:")
            for action_type, count in stats['action_types'].items():
                print(f"  {action_type}: {count}")

def main():
    results_dir = 'results'
    datasets = ['mind2web', 'webarena', 'webvoyager']
    
    for dataset in datasets:
        process_dataset(dataset, results_dir)

if __name__ == '__main__':
    main() 