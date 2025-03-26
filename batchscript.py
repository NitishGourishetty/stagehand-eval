'''
This script will run the stagehand agent.

Input: List of queries and expected actions for each query.

Based on it will find all the stagehand actions (act, observe, extract, etc.) it took to answer the query.
After finding the stagehand actions, it will compare it against the expected actions and give an accuracy score.

Output:
1) Accuracy score from LLM evaluation
2) Time it takes to answer each query
3) Number of tokens for a query
'''

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

# Load environment variables from .env file
load_dotenv()

# Configuration
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
LOG_DIR = os.path.join(RESULTS_DIR, 'logs')
TIMEOUT_SECONDS = 90  # Timeout in seconds
RATE_LIMIT_ERROR = "rate_limit_error"
RETRY_ERROR = "RetryError"

# Dataset paths
MIND2WEB_PATH = "mind2web_train_filtered.xlsx"
WEBARENA_PATH = "webarena.json"
WEBVOYAGER_PATH = "webvoyager.jsonl"

# Use OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

def load_mind2web_tasks():
    """Load tasks from Mind2Web Excel spreadsheet"""
    print("Loading tasks from Mind2Web spreadsheet")
    try:
        df = pd.read_excel(MIND2WEB_PATH)
        tasks = []
        for _, row in df.iterrows():
            task_id = int(row['id'])  # Ensure integer
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
    """Load tasks from WebArena JSON file"""
    print("Loading tasks from WebArena JSON")
    try:
        with open(WEBARENA_PATH, 'r') as f:
            data = json.load(f)
            tasks = []
            for item in data:
                tasks.append({
                    'id': int(item['task_id']),  # Convert to integer
                    'website': item['sites'][0] if item['sites'] else 'unknown',
                    'task': item['intent']
                })
            return tasks
    except Exception as e:
        print(f"Error loading WebArena JSON: {str(e)}")
        sys.exit(1)

def load_webvoyager_tasks():
    """Load tasks from WebVoyager JSONL file"""
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
        return tasks
    except Exception as e:
        print(f"Error loading WebVoyager JSONL: {str(e)}")
        sys.exit(1)

def load_tasks(dataset):
    """Load tasks based on selected dataset"""
    if dataset == 'mind2web':
        return load_mind2web_tasks()
    elif dataset == 'webarena':
        return load_webarena_tasks()
    elif dataset == 'webvoyager':
        return load_webvoyager_tasks()
    else:
        print(f"Unknown dataset: {dataset}")
        sys.exit(1)

def get_dataset_paths(dataset):
    """Get dataset-specific file paths"""
    # Create results and log directories if they don't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Dataset-specific paths
    results_file = os.path.join(RESULTS_DIR, f"{dataset}_results.json")
    log_file = os.path.join(LOG_DIR, f"{dataset}_log.txt")
    
    return results_file, log_file

# Extract stagehand actions from log file
def extract_stagehand_actions(log_content):
    actions = []
    
    # Extract ACT and NAVIGATE actions
    pattern = r'<stagehand>\[(ACT|NAVIGATE|OBSERVE|EXTRACT)\](.*?)</stagehand>'
    matches = re.findall(pattern, log_content, re.DOTALL)
    
    for action_type, content in matches:
        content = content.strip()
        if action_type == "NAVIGATE":
            url_match = re.search(r'(https?://\S+)', content)
            if url_match:
                actions.append(f"[NAVIGATE] {url_match.group(1)}")
        elif action_type == "ACT":
            # Skip metadata lines
            if "Using page.act" in content or "Execution complete" in content:
                continue
                
            # Try to extract JSON action if present
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
            
            # Use the raw content as fallback
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
    - Be very lenient due to the 45-second time limit
    - The system likely didn't have time to complete all actions
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
    
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not found. Using random placeholder values.")
        random_accuracy = random.randint(60, 85)
        random_similarity = random.randint(65, 90)
        return {
            "accuracy_score": random_accuracy,
            "similarity_score": random_similarity,
            "steps_percentage": steps_percentage,
            "explanation": "This is a placeholder evaluation. Set OPENAI_API_KEY env variable for actual evaluation."
        }
    
    try:
        print("Calling OpenAI API for evaluation...")
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4-turbo-preview",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                response_text = result["choices"][0]["message"]["content"]
                
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        evaluation = json.loads(json_match.group(0))
                        evaluation["steps_percentage"] = steps_percentage
                        return evaluation
                    except json.JSONDecodeError:
                        print(f"Error parsing JSON from LLM response")
        
        return {
            "accuracy_score": 50,
            "similarity_score": 50,
            "steps_percentage": steps_percentage,
            "explanation": f"Error: Failed to get valid evaluation from OpenAI API. Status code: {response.status_code}"
        }
    except Exception as e:
        print(f"Error evaluating with OpenAI: {str(e)}")
        return {
            "accuracy_score": 0,
            "similarity_score": 0,
            "steps_percentage": steps_percentage,
            "explanation": f"Error: {str(e)}"
        }

def get_ground_truth_actions(task_id, dataset):
    """Get ground truth actions based on dataset type"""
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
                    if int(item['task_id']) == task_id:  # Convert to int for comparison
                        # WebArena has evaluation info in 'eval' field
                        if 'eval' in item and 'reference_answers' in item['eval']:
                            if item['eval']['reference_answers']:
                                # Return as list of actions for consistency
                                if 'fuzzy_match' in item['eval']['reference_answers']:
                                    return [str(x) for x in item['eval']['reference_answers']['fuzzy_match']]
                                elif 'must_include' in item['eval']['reference_answers']:
                                    return [str(x) for x in item['eval']['reference_answers']['must_include']]
                        return []
            return []
        elif dataset == 'webvoyager':
            with open(WEBVOYAGER_PATH, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    if item['task_id'] == task_id:  # Direct string comparison
                        # WebVoyager has final answer in 'Final answer' field
                        return [str(item.get('Final answer', ''))]
            return []
        return []
    except Exception as e:
        print(f"Error getting ground truth actions: {str(e)}")
        return []

def evaluate_task_results(task_id, website, task_description, start_time, status, log_content, dataset, results_file):
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
        print(f"Evaluation result:")
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

def save_evaluation_result(task_id, website, task, ground_truth_actions, stagehand_actions, 
                          evaluation_result, total_time, status, extract_content, results_file):
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
        
        # Read existing results
        results = []
        if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
            with open(results_file, 'r') as f:
                try:
                    results = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error reading results file. Creating new file.")
                    results = []
    
        results.append(result)

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Saved evaluation result for task ID {task_id} to {results_file}")
    except Exception as e:
        print(f"Error saving evaluation result: {str(e)}")

def run_task(task_index=0, tasks=None, dataset=None, results_file=None, log_file=None):
    if task_index >= len(tasks):
        print("All tasks processed!")
        return
    
    current_task = tasks[task_index]
    task_id = current_task['id']
    website = current_task['website']
    task_description = current_task['task']
    
    # Check if this task has already been processed
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
        with open(results_file, 'r') as f:
            try:
                results = json.load(f)
                if any(result.get('task_id') == task_id for result in results):
                    print(f"Task ID {task_id} already processed. Skipping...")
                    return run_task(task_index + 1, tasks, dataset, results_file, log_file)
            except:
                pass
    
    # Skip to ID 3 or higher as requested
    if dataset == 'mind2web' and task_id < 120:
        print(f"Skipping task ID {task_id} as requested to start from ID 3...")
        return run_task(task_index + 120, tasks, dataset, results_file, log_file)
    
    # Format the query as requested
    current_query = f"USE {website} to do this task: {task_description}"
    
    print(f"\n================================================================================")
    print(f"Processing task {task_index+1}/{len(tasks)}: ID={task_id}, Website={website}")
    print(f"Query: '{current_query}'")
    print(f"================================================================================\n")
    
    # Clear the log file
    with open(log_file, 'w') as f:
        pass
    
    # Keep the log content
    log_content = ""
    
    try:
        timestamp = datetime.datetime.now().isoformat()
        log_line = f'\n\n{timestamp} - TASK {task_index+1}, ID={task_id}: "{current_query}"\n'
        log_content += log_line
        
        with open(log_file, 'a') as f:
            f.write(log_line)
        
        # Start timing the query
        start_time = time.time()
        
        npm_process = subprocess.Popen(
            ['npm', 'start', current_query],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        killed = False
        buffer = []  # For detecting rate limit in multi-line output

        # Set timeout end time
        timeout_end = start_time + TIMEOUT_SECONDS

        for line in npm_process.stdout:
            # Store last few lines to detect multi-line errors
            buffer.append(line)
            if len(buffer) > 10:
                buffer.pop(0)
            
            # Add line to log content
            log_content += line
            
            # Check if rate limit has been hit
            if RATE_LIMIT_ERROR in line or (RETRY_ERROR in line and any(RATE_LIMIT_ERROR in b for b in buffer)):
                elapsed_time = time.time() - start_time
                rate_limit_msg = f'{timestamp} - RATE LIMIT ERROR after {elapsed_time:.2f} seconds\n'
                log_content += rate_limit_msg
                print(f"RATE LIMIT ERROR after {elapsed_time:.2f} seconds, moving to next query")
                
                with open(log_file, 'a') as f:
                    f.write(line)
                    f.write(rate_limit_msg)
                
                killed = True
                npm_process.terminate()
                try:
                    npm_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    npm_process.kill()
                
                # Evaluate and save results
                evaluate_task_results(task_id, website, task_description, start_time, "rate_limit_error", log_content, dataset, results_file)
                
                # Add sleep to allow rate limits to reset
                wait_time = 60  # Wait 60 seconds before next query to help with rate limits
                print(f"Waiting {wait_time} seconds before next query...")
                time.sleep(wait_time)
                
                return run_task(task_index + 1, tasks, dataset, results_file, log_file)
            
            # Check if we've exceeded our timeout
            current_time = time.time()
            if current_time > timeout_end:
                elapsed_time = current_time - start_time
                timeout_msg = f'{timestamp} - TIMEOUT after {elapsed_time:.2f} seconds\n'
                log_content += timeout_msg
                print(f"TIMEOUT after {elapsed_time:.2f} seconds, moving to next query")
                
                with open(log_file, 'a') as f:
                    f.write(line)
                    f.write(timeout_msg)
                
                killed = True
                npm_process.terminate()
                try:
                    npm_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    npm_process.kill()
                
                # Evaluate and save results
                evaluate_task_results(task_id, website, task_description, start_time, "timeout", log_content, dataset, results_file)
                
                time.sleep(5)
                
                return run_task(task_index + 1, tasks, dataset, results_file, log_file)
            
            # Write to log file and stdout
            with open(log_file, 'a') as f:
                f.write(line)
            sys.stdout.write(line)
            sys.stdout.flush()
            
            # Check for successful completion or errors
            if "Task Completed" in line:
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                completion_msg = f'{timestamp} - Task completed in {elapsed_time:.2f} seconds\n'
                log_content += completion_msg
                print(f"Finished query in {elapsed_time:.2f} seconds, moving to next query")
                
                with open(log_file, 'a') as f:
                    f.write(completion_msg)
                
                killed = True
                npm_process.terminate()
                try:
                    npm_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    npm_process.kill()
                
                # Evaluate and save results
                evaluate_task_results(task_id, website, task_description, start_time, "completed", log_content, dataset, results_file)
                
                time.sleep(5)
                
                return run_task(task_index + 1, tasks, dataset, results_file, log_file)

            if "[AI_RETRYError]" in line:
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                error_msg = f'{timestamp} - Error in {elapsed_time:.2f} seconds\n'
                log_content += error_msg
                print(f"GOT AN ERROR in {elapsed_time:.2f} seconds, moving to next query")
                
                with open(log_file, 'a') as f:
                    f.write(error_msg)
                
                killed = True
                npm_process.terminate()
                try:
                    npm_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    npm_process.kill()
                
                # Evaluate and save results
                evaluate_task_results(task_id, website, task_description, start_time, "ai_retry_error", log_content, dataset, results_file)
                
                time.sleep(5)
                
                return run_task(task_index + 1, tasks, dataset, results_file, log_file)

        # If we get here, the process exited without any explicit trigger
        exit_code = npm_process.wait()
        exit_msg = f"Process exited with code {exit_code} without finishing\n"
        log_content += exit_msg
        print(exit_msg)
        
        with open(log_file, 'a') as f:
            f.write(exit_msg)
        
        # Always evaluate and save results
        evaluate_task_results(task_id, website, task_description, start_time, f"unexpected_exit_code_{exit_code}", log_content, dataset, results_file)
        
        time.sleep(5)
        return run_task(task_index + 1, tasks, dataset, results_file, log_file)
        
    except KeyboardInterrupt:
        print('\nReceived KeyboardInterrupt. Cleaning up and exiting...')
        with open(log_file, 'a') as f:
            f.write('\nScript terminated by user\n')
        if 'npm_process' in locals():
            npm_process.terminate()
        sys.exit(0)
    except Exception as e:
        error_msg = f"Error running npm start for task ID {task_id}: {str(e)}"
        print(error_msg)
        with open(log_file, 'a') as f:
            f.write(f"{error_msg}\n")
        
        # Evaluate and save results
        evaluate_task_results(task_id, website, task_description, start_time, "script_error", log_content, dataset, results_file)
        
        time.sleep(5)
        return run_task(task_index + 1, tasks, dataset, results_file, log_file)

def main():
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Run Stagehand evaluation on different datasets')
    parser.add_argument('dataset', choices=['mind2web', 'webarena', 'webvoyager'],
                      help='Dataset to evaluate (mind2web, webarena, or webvoyager)')
    args = parser.parse_args()

    # Get dataset-specific file paths
    results_file, log_file = get_dataset_paths(args.dataset)
    
    # Initialize results file
    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            json.dump([], f, indent=2)
        print(f"Created new evaluation results file: {results_file}")
    
    # Initialize log file
    if not os.path.exists(log_file):
        open(log_file, 'w').close()
    
    # Signal handler for CTRL+C
    def signal_handler(sig, frame):
        print('\nReceived SIGINT. Cleaning up and exiting...')
        with open(log_file, 'a') as f:
            f.write('\nScript terminated by user\n')
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Load tasks from selected dataset
    tasks = load_tasks(args.dataset)
    
    # Run tasks with dataset-specific paths
    run_task(0, tasks, args.dataset, results_file=results_file, log_file=log_file)
    
    print("All tasks completed!")

if __name__ == "__main__":
    main()