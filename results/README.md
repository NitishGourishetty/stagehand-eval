# Results Directory

This directory contains evaluation results from running Stagehand on different web automation benchmarks. Each file contains the results of running Stagehand on a specific dataset, with both raw and processed versions available.

## File Descriptions

### Mind2Web Results
- `mind2web_results.json`: Raw results from running Stagehand on the Mind2Web benchmark. Contains task IDs, websites, tasks, ground truth actions, Stagehand's actions, and accuracy scores.
- `mind2web_results_filtered.json`: Filtered version of the raw results, likely containing only valid or relevant entries.
- `mind2web_results_reprocessed.json`: Reprocessed version of the results with additional analysis or formatting.
- `mind2web_stats.json`: Statistics for raw results (1009 tasks, 72.05% success rate)
- `mind2web_stats_filtered.json`: Statistics for filtered results (732 tasks, 99.32% success rate)
- `mind2web_stats_reprocessed.json`: Statistics for reprocessed results (732 tasks, 100% success rate)

### WebArena Results
- `webarena_results.json`: Raw results from running Stagehand on the WebArena benchmark. Contains task IDs, websites, tasks, ground truth actions, Stagehand's actions, and accuracy scores.
- `webarena_results_filtered.json`: Filtered version of the raw results, likely containing only valid or relevant entries.
- `webarena_results_reprocessed.json`: Reprocessed version of the results with additional analysis or formatting.
- `webarena_stats.json`: Statistics for raw results (597 tasks, 57.96% success rate)
- `webarena_stats_filtered.json`: Statistics for filtered results (495 tasks, 69.90% success rate)
- `webarena_stats_reprocessed.json`: Statistics for reprocessed results (495 tasks, 98.59% success rate)

### WebVoyager Results
- `webvoyager_results.json`: Raw results from running Stagehand on the WebVoyager benchmark. Contains task IDs, websites, tasks, ground truth actions, Stagehand's actions, and accuracy scores.
- `webvoyager_results_filtered.json`: Filtered version of the raw results, likely containing only valid or relevant entries.
- `webvoyager_results_reprocessed.json`: Reprocessed version of the results with additional analysis or formatting.
- `webvoyager_stats.json`: Statistics for raw results (90 tasks, 88.89% success rate)
- `webvoyager_stats_filtered.json`: Statistics for filtered results (86 tasks, 93.02% success rate)
- `webvoyager_stats_reprocessed.json`: Statistics for reprocessed results (86 tasks, 100% success rate)

## File Creation Process

1. Raw Results Files (`*_results.json`):
   - Created by running Stagehand on each respective benchmark
   - Contains the complete, unprocessed output from the evaluation runs

2. Filtered Results Files (`*_results_filtered.json`):
   - Created by filtering the raw results to remove invalid entries or focus on specific aspects
   - Maintains the same structure as raw results but with a subset of entries

3. Reprocessed Results Files (`*_results_reprocessed.json`):
   - Created by applying additional processing or analysis to the filtered results
   - May include additional metrics, reformatted data, or aggregated statistics

4. Statistics Files (`*_stats*.json`):
   - Generated using the `calculate_stats.py` script
   - Available for raw, filtered, and reprocessed results
   - Contains comprehensive statistics including:
     - Total number of tasks
     - Success rate (percentage of tasks with accuracy > 0)
     - Average, median, and standard deviation of accuracy scores
     - Distribution of action types (NAVIGATE, ACT, OBSERVE, EXTRACT)
     - Average number of actions per task

## Data Structure

Each result file contains an array of objects with the following structure:
```json
{
  "task_id": "unique identifier",
  "website": "target website URL",
  "task": "description of the task to perform",
  "ground_truth_actions": ["list of expected actions"],
  "stagehand_actions": ["list of actions performed by Stagehand"],
  "extract_content": ["extracted information"],
  "accuracy_score": "numerical score"
}
```

## Logs Directory

The `logs/` directory contains detailed execution logs from the evaluation runs, which can be used for debugging or analyzing the performance of Stagehand in more detail. 