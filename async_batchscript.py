import asyncio
import subprocess
import datetime
import os
import signal
import sys
import time
import re
import pandas as pd
import json
import requests
import ast
import random
import argparse
from dotenv import load_dotenv
from parse_task_ids import parse_task_ids, parse_webarena_task_ids

# Load environment variables
load_dotenv()

# Configuration
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
LOG_DIR = os.path.join(RESULTS_DIR, 'logs')
TIMEOUT_SECONDS = 90   # Timeout for each task
LAUNCH_INTERVAL = 35  # Launch a new task every 22.5 seconds

RATE_LIMIT_ERROR = "rate_limit_error"
RETRY_ERROR = "RetryError"

# Dataset paths
MIND2WEB_PATH = "mind2web_train_filtered.xlsx"
WEBARENA_PATH = "webarena.json"
WEBVOYAGER_PATH = "webvoyager.jsonl"

# Use OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

def load_mind2web_tasks():
    print("Loading tasks from Mind2Web spreadsheet")
    try:
        df = pd.read_excel(MIND2WEB_PATH)
        tasks = []

        completed_task_ids = parse_task_ids()
        for _, row in df.iterrows():
            task_id = int(row['id'])  # Ensure integer
            if task_id in completed_task_ids:
                continue
            website = row['website_id'] if 'website_id' in row else row['website']
            task = row['task'] if 'task' in row else row['confirmed_task']
            tasks.append({
                'id': task_id,
                'website': website,
                'task': task
            })
        return tasks
    except Exception as e:
        print(f"Error loading Mind2Web spreadsheet: {str(e)}")
        sys.exit(1)

def load_webarena_tasks():
    print("Loading tasks from WebArena JSON")
    try:
        with open(WEBARENA_PATH, 'r') as f:
            completed_task_ids = parse_webarena_task_ids()
            data = json.load(f)
            tasks = []
            for item in data:
                task_id = int(item['task_id'])
                if task_id in completed_task_ids:
                    continue

                tasks.append({
                    'id': task_id,
                    'website': item['sites'][0] if item['sites'] else 'unknown',
                    'task': item['intent']
                })
            return tasks
    except Exception as e:
        print(f"Error loading WebArena JSON: {str(e)}")
        sys.exit(1)

def load_webvoyager_tasks():
    print("Loading tasks from WebVoyager JSONL")
    try:
        tasks = []
        with open(WEBVOYAGER_PATH, 'r') as f:
            for line in f:
                item = json.loads(line)
                tasks.append({
                    'id': item['task_id'],  # Keep as string/UUID
                    'website': item['web'],
                    'task': item['ques']
                })
        random.shuffle(tasks)
        return tasks
    except Exception as e:
        print(f"Error loading WebVoyager JSONL: {str(e)}")
        sys.exit(1)

def load_tasks(dataset):
    """Load tasks based on selected dataset"""
    # Get results file path first
    results_file, _ = get_dataset_paths(dataset)
    completed_task_ids = set()

    # Load completed task IDs from results file
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
                completed_task_ids = {r.get('task_id') for r in results if r.get('task_id') is not None}
        except Exception as e:
            print(f"Warning: Error reading results file: {str(e)}")

    # Load and filter tasks based on dataset
    if dataset == 'mind2web':
        tasks = load_mind2web_tasks()
    elif dataset == 'webarena':
        tasks = load_webarena_tasks()
    elif dataset == 'webvoyager':
        tasks = load_webvoyager_tasks()
    else:
        print(f"Unknown dataset: {dataset}")
        sys.exit(1)

    # Filter out completed tasks
    filtered_tasks = [task for task in tasks if task['id'] not in completed_task_ids]
    
    print(f"Loaded {len(tasks)} total tasks")
    print(f"Found {len(completed_task_ids)} completed tasks")
    print(f"Remaining tasks to process: {len(filtered_tasks)}")
    
    return filtered_tasks

def get_dataset_paths(dataset):
    """Get dataset-specific file paths"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    results_file = os.path.join(RESULTS_DIR, f"{dataset}_results.json")
    log_file = os.path.join(LOG_DIR, f"{dataset}_log.txt")
    return results_file, log_file

def extract_stagehand_actions(log_content):
    actions = []
    pattern = r'<stagehand>\[(ACT|NAVIGATE|OBSERVE|EXTRACT)\](.*?)</stagehand>'
    matches = re.findall(pattern, log_content, re.DOTALL)
    
    for action_type, content in matches:
        content = content.strip()
        if action_type == "NAVIGATE":
            url_match = re.search(r'(https?://\S+)', content)
            if url_match:
                actions.append(f"[NAVIGATE] {url_match.group(1)}")
        elif action_type == "ACT":
            if "Using page.act" in content or "Execution complete" in content:
                continue
            # Attempt JSON parse
            if "{" in content and "}" in content:
                try:
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(0))
                        if "action" in data:
                            actions.append(f"[ACT] {data['action']}")
                            continue
                except:
                    pass
            actions.append(f"[ACT] {content}")
        elif action_type == "OBSERVE":
            actions.append(f"[OBSERVE] {content}")
        elif action_type == "EXTRACT":
            actions.append(f"[EXTRACT] {content}")
    
    return actions

def get_extract_content(log_content):
    extracts = []
    extract_pattern = r'<stagehand>\[EXTRACT\](.*?)</stagehand>'
    matches = re.findall(extract_pattern, log_content, re.DOTALL)
    return [match.strip() for match in matches] if matches else []

def calculate_steps_percentage(stagehand_actions, ground_truth_actions):
    if not ground_truth_actions:
        return 0
    steps_taken = min(len(stagehand_actions), len(ground_truth_actions))
    return (steps_taken / len(ground_truth_actions)) * 100

# def evaluate_with_llm(stagehand_actions, ground_truth_actions, website, task):
#     """Uses an LLM to generate an accuracy and similarity score. (Same logic as before.)"""
#     if not stagehand_actions:
#         return {
#             "accuracy_score": 0,
#             "similarity_score": 0,
#             "steps_percentage": 0,
#             "explanation": "No stagehand actions were found to evaluate."
#         }
    
#     steps_percentage = calculate_steps_percentage(stagehand_actions, ground_truth_actions)
#     prompt = f"""
#     Evaluate how well the Stagehand AI system completed this task:
    
#     Website: {website}
#     Task: {task}
    
#     Ground truth actions (expected steps):
#     {json.dumps(ground_truth_actions, indent=2)}
    
#     Stagehand actions (actual steps taken):
#     {json.dumps(stagehand_actions, indent=2)}
    
#     Rate from 0-100:
#     1. Accuracy score: Exact matches between ground truth and actual actions
#     2. Similarity score: Semantic similarity even if worded differently
#     3. Brief explanation of your evaluation

#     IMPORTANT SCORING GUIDANCE:
#     - Be very lenient due to time limits
#     - If the system is on the right track but ran out of time, give higher scores
#     - The system likely didn't have time to complete all actions (this is ok because we put an artifical time limit so factor it in)
#     - Reward partial progress and correct intent
#     - If the system was on the right track but ran out of time, give higher scores
#     - Consider semantic similarity over exact wording
#     - Even if only 1-2 actions were completed correctly, consider scores of 60-70 if they were the right initial steps

#     JSON response only:
#     {{
#       "accuracy_score": <score>,
#       "similarity_score": <score>,
#       "explanation": "<explanation>"
#     }}
#     """

#     if not OPENAI_API_KEY:
#         print("Warning: OPENAI_API_KEY not found. Using random placeholder values.")
#         raise Exception("OPENAI_API_KEY not found")
#     try:
#         print("Calling OpenAI API for evaluation...")
#         response = requests.post(
#             "https://api.openai.com/v1/chat/completions",
#             headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
#             json={
#                 "model": "gpt-4o-2024-08-06",
#                 "max_tokens": 1000,
#                 "messages": [{"role": "user", "content": prompt}]
#             },
#             timeout=30
#         )
        
#         if response.status_code == 200:
#             result = response.json()
#             if "choices" in result and len(result["choices"]) > 0:
#                 response_text = result["choices"][0]["message"]["content"]
#                 json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
#                 if json_match:
#                     try:
#                         evaluation = json.loads(json_match.group(0))
#                         evaluation["steps_percentage"] = steps_percentage
#                         return evaluation
#                     except json.JSONDecodeError:
#                         print(f"Error parsing JSON from LLM response")
        
#         return {
#             "accuracy_score": 50,
#             "similarity_score": 50,
#             "steps_percentage": steps_percentage,
#             "explanation": f"OpenAI API error or invalid response. Status code: {response.status_code}"
#         }
#     except Exception as e:
#         print(f"Error evaluating with OpenAI: {str(e)}")
#         return {
#             "accuracy_score": 0,
#             "similarity_score": 0,
#             "steps_percentage": steps_percentage,
#             "explanation": f"Exception: {str(e)}"
#         }

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

def evaluate_with_llm(stagehand_actions, ground_truth_actions, website, task):
    if not stagehand_actions:
        return {
            "accuracy_score": 0,
            "similarity_score": 0,
            "steps_percentage": 0,
            "explanation": "No stagehand actions were found to evaluate."
        }

    steps_percentage = calculate_steps_percentage(stagehand_actions, ground_truth_actions)

    prompt = f"""
Evaluate how well the Stagehand AI system completed this task:

Website: {website}
Task: {task}

Ground truth actions (expected steps):
{json.dumps(ground_truth_actions, indent=2)}

Stagehand actions (actual steps taken):
{json.dumps(stagehand_actions, indent=2)}

Rate from 0-100:
1. Accuracy score: Exact matches between ground truth and actual actions
2. Similarity score: Semantic similarity even if worded differently
3. Brief explanation of your evaluation

IMPORTANT SCORING GUIDANCE:
- Be very lenient due to time limits
- If the system is on the right track but ran out of time, give higher scores
- The system likely didn't have time to complete all actions (this is ok because we put an artificial time limit so factor it in)
- Reward partial progress and correct intent
- If the system was on the right track but ran out of time, give higher scores
- Consider semantic similarity over exact wording
- Even if only 1-2 actions were completed correctly, consider scores of 60-70 if they were the right initial steps

JSON response only:
{{
  "accuracy_score": <score>,
  "similarity_score": <score>,
  "explanation": "<explanation>"
}}
    """

    if not ANTHROPIC_API_KEY:
        print("Warning: ANTHROPIC_API_KEY not found. Using placeholder values.")
        raise Exception("ANTHROPIC_API_KEY not found")

    try:
        print("Calling Claude API for evaluation...")
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-3-opus-20240229",
                "max_tokens": 1000,
                "temperature": 0.3,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if "content" in result and isinstance(result["content"], list):
                response_text = "".join([block.get("text", "") for block in result["content"]])
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        evaluation = json.loads(json_match.group(0))
                        evaluation["steps_percentage"] = steps_percentage
                        return evaluation
                    except json.JSONDecodeError:
                        print("Error parsing JSON from Claude response")
        
        return {
            "accuracy_score": 50,
            "similarity_score": 50,
            "steps_percentage": steps_percentage,
            "explanation": f"Claude API error or invalid response. Status code: {response.status_code}"
        }

    except Exception as e:
        print(f"Error evaluating with Claude: {str(e)}")
        return {
            "accuracy_score": 0,
            "similarity_score": 0,
            "steps_percentage": steps_percentage,
            "explanation": f"Exception: {str(e)}"
        }
    
def get_ground_truth_actions(task_id, dataset):
    """Retrieve expected actions from each dataset's ground truth structure."""
    try:
        if dataset == 'mind2web':
            df = pd.read_excel(MIND2WEB_PATH)
            task_row = df[df['id'] == task_id]
            if task_row.empty:
                return []
            return ast.literal_eval(task_row.iloc[0]['action_reprs'])
        elif dataset == 'webarena':
            with open(WEBARENA_PATH, 'r') as f:
                data = json.load(f)
                for item in data:
                    if int(item['task_id']) == task_id:
                        if 'eval' in item and 'reference_answers' in item['eval']:
                            ref = item['eval']['reference_answers']
                            if ref:
                                if 'fuzzy_match' in ref:
                                    return [str(x) for x in ref['fuzzy_match']]
                                elif 'must_include' in ref:
                                    return [str(x) for x in ref['must_include']]
                        return []
            return []
        elif dataset == 'webvoyager':
            with open(WEBVOYAGER_PATH, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    if item['task_id'] == task_id:
                        return [str(item.get('Final answer', ''))]
            return []
        return []
    except Exception as e:
        print(f"Error getting ground truth actions: {str(e)}")
        return []

def evaluate_task_results(task_id, website, task_description, start_time, status, log_content, dataset, results_file):
    """Wrapper that extracts actions, calls the LLM evaluation, and saves results."""
    try:
        elapsed_time = time.time() - start_time
        ground_truth_actions = get_ground_truth_actions(task_id, dataset)
        
        print("\nExtracting stagehand actions from log file...")
        stagehand_actions = extract_stagehand_actions(log_content)
        print(f"Found {len(stagehand_actions)} stagehand actions:")
        for idx, action in enumerate(stagehand_actions):
            print(f"  {idx+1}. {action}")
        
        extract_content = get_extract_content(log_content)
        
        print("\nEvaluating stagehand actions against ground truth...")
        evaluation_result = evaluate_with_llm(stagehand_actions, ground_truth_actions, website, task_description)
        print("Evaluation result:")
        print(f"  Accuracy score: {evaluation_result['accuracy_score']}")
        print(f"  Similarity score: {evaluation_result['similarity_score']}")
        print(f"  Steps percentage: {evaluation_result['steps_percentage']:.1f}%")
        print(f"  Explanation: {evaluation_result['explanation']}")
        
        print("\nSaving evaluation result to JSON...")
        save_evaluation_result(
            task_id, website, task_description, ground_truth_actions, stagehand_actions,
            evaluation_result, elapsed_time, status, extract_content, results_file
        )
        return True
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return False

def save_evaluation_result(
    task_id, 
    website, 
    task, 
    ground_truth_actions, 
    stagehand_actions,
    evaluation_result, 
    total_time, 
    status, 
    extract_content, 
    results_file
):
    """Append final result to the dataset's results JSON."""
    try:
        result = {
            'task_id': task_id,
            'website': website,
            'task': task,
            'ground_truth_actions': ground_truth_actions,
            'stagehand_actions': stagehand_actions,
            'extract_content': extract_content,
            'accuracy_score': evaluation_result['accuracy_score'],
            'similarity_score': evaluation_result['similarity_score'],
            'steps_percentage': evaluation_result['steps_percentage'],
            'explanation': evaluation_result['explanation'],
            'total_time_seconds': total_time,
            'status': status,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        results = []
        if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
            with open(results_file, 'r') as f:
                try:
                    results = json.load(f)
                except json.JSONDecodeError:
                    print("Error reading results file; creating new.")
                    results = []
    
        results.append(result)

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved evaluation result for task ID {task_id} to {results_file}")
    except Exception as e:
        print(f"Error saving evaluation result: {str(e)}")

async def watch_subprocess(cmd_args, log_content_ref, on_complete_callback):
    """
    Launch the npm process, capturing output asynchronously.
    If we see certain triggers or end-of-output, we call on_complete_callback.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        buffer = []  # For detecting rate limit in multi-line output
        
        while True:
            line = await process.stdout.readline()
            if not line:
                exit_code = await process.wait()
                on_complete_callback(log_content_ref[0], exit_code)
                return
                
            decoded_line = line.decode('utf-8', errors='ignore')
            log_content_ref[0] += decoded_line
            
            # Keep last lines to detect rate-limit errors
            buffer.append(decoded_line)
            if len(buffer) > 10:
                buffer.pop(0)
            
            # Rate limit or other error triggers
            if RATE_LIMIT_ERROR in decoded_line or (
                RETRY_ERROR in decoded_line and any(RATE_LIMIT_ERROR in b for b in buffer)
            ):
                process.terminate()  # Ensure process is terminated
                await process.wait()  # Wait for termination
                on_complete_callback(log_content_ref[0], "rate_limit_error")
                return
            
            if "Task Completed" in decoded_line:
                process.terminate()  # Ensure process is terminated
                await process.wait()  # Wait for termination
                on_complete_callback(log_content_ref[0], "completed")
                return
            
            if "[AI_RETRYError]" in decoded_line:
                process.terminate()  # Ensure process is terminated
                await process.wait()  # Wait for termination
                on_complete_callback(log_content_ref[0], "ai_retry_error")
                return
    except Exception as e:
        print(f"Error in watch_subprocess: {str(e)}")
        try:
            process.terminate()  # Ensure process is terminated even on error
            await process.wait()
        except:
            pass
        on_complete_callback(log_content_ref[0], f"error: {str(e)}")
        return

async def run_task_async(
    task_index,
    tasks,
    dataset,
    results_file,
    log_file
):
    """
    Runs one task asynchronously, with a 90s timeout. We do NOT block further tasks.
    We'll keep the same flow for capturing logs, evaluating, and saving results.
    """
    if task_index >= len(tasks):
        return

    current_task = tasks[task_index]
    task_id = current_task['id']
    website = current_task['website']
    task_description = current_task['task']

    # Check if already processed
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
        with open(results_file, 'r') as f:
            try:
                results = json.load(f)
                if any(r.get('task_id') == task_id for r in results):
                    print(f"Task ID {task_id} already processed. Skipping...")
                    return
            except:
                pass

    current_query = f"USE {website} to do this task: {task_description}"
    print(f"\n===== ASYNC TASK {task_index+1}/{len(tasks)}: ID={task_id} =====")
    print(f"Query: '{current_query}'")

    with open(log_file, 'a') as f:
        f.write(f"\n\nTask {task_index+1} of {len(tasks)}: {current_query}\n")

    log_content_ref = [""]  # mutable reference for capturing logs
    start_time = time.time()

    # Callback after process finishes or error
    def on_process_complete(log_contents, exit_code_or_reason):
        evaluate_task_results(
            task_id,
            website,
            task_description,
            start_time,
            str(exit_code_or_reason),
            log_contents,
            dataset,
            results_file
        )
    
    # Start the subprocess watcher
    watch_task = asyncio.create_task(
        watch_subprocess(['npm', 'start', current_query], log_content_ref, on_process_complete)
    )

    # Wait for it to finish or time out in 90s
    done, pending = await asyncio.wait({watch_task}, timeout=TIMEOUT_SECONDS)
    if watch_task in pending:
        # Timeout
        watch_task.cancel()
        print(f"Timeout for task {task_id}. Terminating process.")
        on_process_complete(log_content_ref[0], "timeout")

    # Not blocking further tasks
    print(f"Finished async task {task_index+1} for ID {task_id}.\n")


async def main_async():
    parser = argparse.ArgumentParser(description='Run Stagehand evaluation concurrently, launching a new task every 22.5s.')
    parser.add_argument('dataset', choices=['mind2web', 'webarena', 'webvoyager'])
    args = parser.parse_args()

    results_file, log_file = get_dataset_paths(args.dataset)
    tasks = load_tasks(args.dataset)
    
    # Prevent   
    random.shuffle(tasks)

    # Initialize results file if needed
    if not os.path.exists(results_file):
        if not os.path.exists(results_file):
            with open(results_file, 'w') as f:
                json.dump([], f, indent=2)
            print(f"Created new results file: {results_file}")

    # Init log file
    if not os.path.exists(log_file):
        open(log_file, 'w').close()

    def handle_sigint(sig, frame):
        print('Received SIGINT. Exiting...')
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    # We will store references to all tasks we launch
    running_tasks = []
    for i in range(len(tasks)):
        # Start the task
        task_coro = run_task_async(i, tasks, args.dataset, results_file, log_file)
        running_tasks.append(asyncio.create_task(task_coro))
        
        # Wait 22.5s before launching the next one
        # so tasks run concurrently with a pipeline offset
        if i < len(tasks) - 1:  # avoid extra sleep after the last
            await asyncio.sleep(LAUNCH_INTERVAL)

    # Wait for all launched tasks to complete
    await asyncio.gather(*running_tasks)
    print("All async tasks completed.")

if __name__ == "__main__":
    asyncio.run(main_async())
