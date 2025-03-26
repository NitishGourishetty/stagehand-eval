'''
This script runs the stagehand agent and evaluates its actions against ground truth.

Workflow:
1. Load tasks from a spreadsheet
2. For each task:
   a. Run npm start with the task
   b. Extract stagehand actions from logs
   c. Compare against ground truth actions using LLM
   d. Record metrics (accuracy, time, tokens) to a CSV
3. Continue processing tasks one by one

Output:
1) Accuracy score based on LLM evaluation
2) Time taken to complete the task
3) Number of tokens used
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

# Configuration
MASTER_LOG_FILE = os.path.join(os.path.dirname(__file__), 'master_everything.log')
CURRENT_LOG_FILE = os.path.join(os.path.dirname(__file__), 'current_task.log')
RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'evaluation_results.json')
FINISH_TRIGGER = "FINAL_TOKEN_COUNT:"
TIMEOUT_SECONDS = 90  # 2.5 minutes timeout -> for time sake -> should do the same for ALL models
RATE_LIMIT_ERROR = "rate_limit_error"
RETRY_ERROR = "RetryError"
SPREADSHEET_PATH = "mind2web_train_filtered.xlsx" 

LLM_API_URL = "https://api.anthropic.com/v1/messages"

# HAD PROBLEMS WITH LOADING ENV -> REPLACE THIS IF YOU CAN. DOING THIS Read API key directly from .env file
LLM_API_KEY = ""
env_file_path = os.path.join(os.path.dirname(__file__), '.env')
print(f"Looking for .env file at: {env_file_path}")
if os.path.exists(env_file_path):
    try:
        print("Found .env file, reading ANTHROPIC_API_KEY...")
        with open(env_file_path, 'r') as f:
            for line in f:
                if line.strip().startswith('ANTHROPIC_API_KEY='):
                    LLM_API_KEY = line.strip().split('=', 1)[1].strip('"').strip("'")
                    print(f"Successfully read API key from .env file (first few chars: {LLM_API_KEY[:4]}...)")
                    break
    except Exception as e:
        print(f"Error reading .env file: {str(e)}")

# Fall back to environment variable if not found in .env
if not LLM_API_KEY:
    LLM_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
    if LLM_API_KEY:
        print(f"Using ANTHROPIC_API_KEY from environment variable (first few chars: {LLM_API_KEY[:4]}...)")
    else:
        print("WARNING: ANTHROPIC_API_KEY not found in .env file or environment variable.")
        print("LLM evaluation will use placeholder values instead of real API calls.")

# Initialize results file if it doesn't exist
def initialize_results_file():
    if not os.path.exists(RESULTS_FILE):
        # Create an empty list for results
        with open(RESULTS_FILE, 'w') as f:
            json.dump([], f, indent=2)
        print(f"Created new evaluation results file: {RESULTS_FILE}")

# Load tasks from spreadsheet
def load_tasks_from_spreadsheet():
    print("Loading tasks from spreadsheet")
    try:
        df = pd.read_excel(SPREADSHEET_PATH)
        tasks = []
        for _, row in df.iterrows():
            task_id = row['id']
            website = row['website']
            task = row['confirmed_task']
            action_reprs = row['action_reprs']
            if isinstance(action_reprs, str):
                try:
                    action_reprs = ast.literal_eval(action_reprs)
                except (ValueError, SyntaxError):
                    pass
            
            tasks.append({
                'id': task_id,
                'website': website,
                'task': task,
                'action_reprs': action_reprs
            })
        return tasks
    except Exception as e:
        print(f"Error loading spreadsheet: {str(e)}")
        sys.exit(1)

# Extract stagehand actions from the log file
def extract_stagehand_actions(log_file_path, task_id):
    actions = []
    
    try:
        with open(log_file_path, "r") as f:
            log_contents = f.read()
        
        # Extract all stagehand actions
        action_pattern = r"<stagehand>\[ACT\]\s*\{[^}]*\"action\":\s*\"([^\"]+)\"[^}]*\}</stagehand>"
        navigate_pattern = r"<stagehand>\[NAVIGATE\]\s*([^\s<]+)</stagehand>"
        extract_pattern = r"<stagehand>\[EXTRACT\]([^<]*)</stagehand>"
        observe_pattern = r"<stagehand>\[OBSERVE\]([^<]*)</stagehand>"
        
        # Collect actions
        actions = []
        
        # Add navigations
        navigations = re.findall(navigate_pattern, log_contents)
        for nav in navigations:
            actions.append(f"NAVIGATE to {nav}")
        
        # Add ACT actions
        act_matches = re.findall(action_pattern, log_contents)
        for action in act_matches:
            actions.append(action)
            
        # Add EXTRACT actions
        extract_matches = re.findall(extract_pattern, log_contents)
        for extract in extract_matches:
            extract = extract.strip()
            if extract.startswith("Content:"):
                # This is the result of an extraction
                actions.append(f"EXTRACT result")
            elif extract.startswith("Using search instruction:"):
                # This is the instruction for an extraction
                instruction = extract.replace("Using search instruction:", "").strip()
                actions.append(f"EXTRACT {instruction}")
            elif "http" in extract:
                # This is a URL extraction
                actions.append(f"EXTRACT from {extract}")
            else:
                actions.append(f"EXTRACT {extract}")
                
        # Add OBSERVE actions - these help understand the context
        observe_matches = re.findall(observe_pattern, log_contents)
        for observe in observe_matches:
            observe = observe.strip()
            actions.append(f"OBSERVE {observe}")
        
        # Normalize actions to better match ground truth format
        normalized_actions = []
        for action in actions:
            # Process NAVIGATE actions
            if action.startswith("NAVIGATE to "):
                url = action.replace("NAVIGATE to ", "")
                normalized_actions.append(f"[NAVIGATE] {url}")
                continue
                
            # Process EXTRACT actions
            if action.startswith("EXTRACT "):
                instruction = action.replace("EXTRACT ", "")
                if instruction == "result":
                    normalized_actions.append(f"[EXTRACT] result")
                else:
                    normalized_actions.append(f"[EXTRACT] {instruction}")
                continue
                
            # Process OBSERVE actions - may be useful for context
            if action.startswith("OBSERVE "):
                instruction = action.replace("OBSERVE ", "")
                normalized_actions.append(f"[OBSERVE] {instruction}")
                continue
                
            # Process click actions
            if "click" in action.lower() or "click on" in action.lower():
                element = action.lower().replace("click on ", "").replace("click ", "")
                # Try to extract element type and text
                if "button" in element:
                    normalized_actions.append(f"[button] {element} -> CLICK")
                elif "dropdown" in element or "combobox" in element:
                    normalized_actions.append(f"[combobox] {element} -> CLICK")
                elif "link" in element:
                    normalized_actions.append(f"[link] {element} -> CLICK")
                else:
                    normalized_actions.append(f"[element] {element} -> CLICK")
                continue
                
            # Process type actions
            if "type" in action.lower():
                match = re.search(r"type\s+[\"']?([^\"']+)[\"']?\s+(?:into|in)\s+(.+)", action.lower())
                if match:
                    text_to_type = match.group(1)
                    element = match.group(2)
                    normalized_actions.append(f"[searchbox] {element} -> TYPE: {text_to_type}")
                else:
                    normalized_actions.append(f"[input] (unknown) -> TYPE: {action}")
                continue
                
            # Process select actions
            if "select" in action.lower():
                match = re.search(r"select\s+[\"']?([^\"']+)[\"']?", action.lower())
                if match:
                    option = match.group(1)
                    normalized_actions.append(f"[combobox] (unknown) -> SELECT: {option}")
                else:
                    normalized_actions.append(f"[combobox] (unknown) -> SELECT: (unknown)")
                continue
                
            # Process press actions (like Enter, Tab, etc.)
            if "press" in action.lower():
                match = re.search(r"press\s+([^\s]+)", action.lower())
                if match:
                    key = match.group(1)
                    normalized_actions.append(f"[keyboard] PRESS: {key}")
                else:
                    normalized_actions.append(f"[keyboard] PRESS: (unknown)")
                continue
                
            # Default case - keep the original action
            normalized_actions.append(action)
        
        return normalized_actions
    except Exception as e:
        print(f"Error extracting actions: {str(e)}")
        return []

# Evaluate stagehand actions against ground truth using LLM
def evaluate_with_llm(stagehand_actions, ground_truth_actions):
    # If there are no stagehand actions, return a default evaluation
    if not stagehand_actions:
        print("No stagehand actions found to evaluate.")
        return {
            "accuracy_score": 0,
            "similarity_score": 0,
            "explanation": "No stagehand actions were found to evaluate."
        }
        
    # Format the prompt for Claude API
    prompt = f"""
    Compare the following two lists of actions and evaluate how well the stagehand actions match the ground truth actions.
    
    Ground truth actions:
    {json.dumps(ground_truth_actions, indent=2)}
    
    Stagehand actions:
    {json.dumps(stagehand_actions, indent=2)}
    
    Please analyze the two lists and provide:
    1. An accuracy score from 0-100 representing how well the stagehand actions match the ground truth actions in terms of exact matches.
    2. A similarity score from 0-100 that considers semantic similarity even when the exact wording is different.
    3. A brief explanation for your scores.
    
    IMPORTANT: Be somewhat lenient in your evaluation due to the tight time constraints (45 seconds) that the system operates under. The system may not have had time to complete all actions, so consider:
    - Give higher scores when the system starts with the correct actions, even if it didn't finish
    - Reward partially correct actions that show the system was on the right track
    - Consider the intent behind actions rather than requiring exact matches
    - If an action is semantically similar but worded differently, treat it as largely correct
    
    Respond in JSON format only:
    {{
        "accuracy_score": <score>,
        "similarity_score": <score>,
        "explanation": "<explanation>"
    }}
    """
    
    try:
        # Check if API key is available
        if not LLM_API_KEY:
            print("Warning: ANTHROPIC_API_KEY not found in environment. Using placeholder evaluation.")
            return {
                "accuracy_score": 70,  # Placeholder
                "similarity_score": 80,  # Placeholder
                "explanation": "This is a placeholder evaluation. Set ANTHROPIC_API_KEY env variable for actual evaluation."
            }
            
        # Make API call to Anthropic Claude
        try:
            print("Calling Claude API for evaluation...")
            response = requests.post(
                LLM_API_URL,
                headers={
                    "x-api-key": LLM_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-3-sonnet-20240229",
                    "max_tokens": 1000,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                },
                timeout=30  # 30 second timeout
            )
        except requests.exceptions.RequestException as e:
            print(f"API request error: {str(e)}")
            # If we hit rate limits here, still return a useful evaluation
            if "rate_limit" in str(e).lower():
                print("LLM evaluation hit rate limits. Using manual scoring instead.")
                return calculate_fallback_scores(stagehand_actions, ground_truth_actions)
            return {
                "accuracy_score": 50,  # Fallback
                "similarity_score": 50,  # Fallback
                "explanation": f"API request error: {str(e)}"
            }
        
        # Parse response
        if response.status_code == 200:
            result = response.json()
            content = result.get("content", [])
            if content and isinstance(content, list) and len(content) > 0:
                response_text = content[0].get("text", "")
                
                # Extract JSON from response text
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        evaluation = json.loads(json_str)
                        return {
                            "accuracy_score": evaluation.get("accuracy_score", 0),
                            "similarity_score": evaluation.get("similarity_score", 0),
                            "explanation": evaluation.get("explanation", "No explanation provided")
                        }
                    except json.JSONDecodeError:
                        print(f"Error parsing JSON from LLM response: {json_str}")
            
            print(f"Unexpected response format from LLM API: {result}")
        elif response.status_code == 429:  # Rate limit error
            print("LLM evaluation hit rate limits. Using manual scoring instead.")
            return calculate_fallback_scores(stagehand_actions, ground_truth_actions)
        else:
            print(f"LLM API error: {response.status_code} - {response.text}")
        
        # Fallback if API call fails
        return {
            "accuracy_score": 50,  # Fallback
            "similarity_score": 50,  # Fallback
            "explanation": f"Error getting evaluation from LLM API: {response.status_code}"
        }
    except Exception as e:
        print(f"Error evaluating with LLM: {str(e)}")
        return {
            "accuracy_score": 0,
            "similarity_score": 0,
            "explanation": f"Error: {str(e)}"
        }

# Calculate fallback scores when LLM evaluation fails
def calculate_fallback_scores(stagehand_actions, ground_truth_actions):
    """Calculate simple string matching scores between actions as a fallback"""
    try:
        if not stagehand_actions or not ground_truth_actions:
            return {
                "accuracy_score": 0,
                "similarity_score": 0,
                "explanation": "No actions to compare or empty set."
            }
        
        # Convert all actions to lowercase for comparison
        stagehand_lower = [action.lower() for action in stagehand_actions]
        ground_truth_lower = [action.lower() for action in ground_truth_actions]
        
        # Count exact matches
        exact_matches = 0
        for gt_action in ground_truth_lower:
            if gt_action in stagehand_lower:
                exact_matches += 1
        
        # Count partial matches (if a ground truth action is contained within a stagehand action)
        partial_matches = 0
        for gt_action in ground_truth_lower:
            for sh_action in stagehand_lower:
                # If ground truth action key elements are in stagehand action
                if not any(term in sh_action for term in gt_action.split()):
                    continue
                    
                # Check for type of action match (click, type, select)
                if ("click" in gt_action and "click" in sh_action) or \
                   ("type" in gt_action and "type" in sh_action) or \
                   ("select" in gt_action and "select" in sh_action):
                    partial_matches += 1
                    break
        
        # Calculate scores
        accuracy_score = int((exact_matches / len(ground_truth_actions)) * 100) if ground_truth_actions else 0
        similarity_score = int(((exact_matches + 0.5 * partial_matches) / len(ground_truth_actions)) * 100) if ground_truth_actions else 0
        
        return {
            "accuracy_score": min(accuracy_score, 100),  # Cap at 100
            "similarity_score": min(similarity_score, 100),  # Cap at 100 
            "explanation": f"Fallback evaluation: Found {exact_matches} exact matches and {partial_matches} partial matches out of {len(ground_truth_actions)} ground truth actions."
        }
    except Exception as e:
        print(f"Error in fallback scoring: {str(e)}")
        return {
            "accuracy_score": 25,
            "similarity_score": 35,
            "explanation": f"Error in fallback scoring: {str(e)}"
        }

# Get extract content from log file
def get_extract_content(log_file_path):
    try:
        with open(log_file_path, "r") as f:
            log_contents = f.read()
            
        # Pattern to match EXTRACT content blocks - using a more robust pattern
        extract_content_pattern = r"<stagehand>\[EXTRACT\] Content: (\[[\s\S]*?])</stagehand>"
        content_matches = re.findall(extract_content_pattern, log_contents)
        
        # Combine all extract contents
        all_content = []
        for content in content_matches:
            try:
                # Try to parse as JSON if it's in that format
                parsed_content = json.loads(content)
                if isinstance(parsed_content, list):
                    all_content.extend(parsed_content)
                else:
                    all_content.append(str(parsed_content))
            except json.JSONDecodeError:
                # If not valid JSON, just add as string
                all_content.append(content)
                
        return all_content
    except Exception as e:
        print(f"Error getting extract content: {str(e)}")
        return []

# Save evaluation results to JSON file
def save_evaluation_result(
    task_id, 
    website, 
    task, 
    ground_truth_actions, 
    stagehand_actions, 
    evaluation_result, 
    total_time, 
    browser_init_time, 
    task_time, 
    tokens_used, 
    status
):
    try:
        # Get extract content from the log
        extract_content = get_extract_content(CURRENT_LOG_FILE)
        
        # Create result object
        result = {
            'task_id': task_id,
            'website': website,
            'task': task,
            'ground_truth_actions': ground_truth_actions,  # No need to JSONify, will be done when writing
            'stagehand_actions': stagehand_actions,
            'extract_content': extract_content,
            'accuracy_score': evaluation_result['accuracy_score'],
            'similarity_score': evaluation_result['similarity_score'],
            'explanation': evaluation_result['explanation'],  # Added explanation field
            'total_time_seconds': total_time,
            'browser_init_time': browser_init_time,
            'task_time_seconds': task_time,
            'tokens_used': tokens_used,
            'status': status,
            'timestamp': datetime.datetime.now().isoformat()  # Added timestamp
        }
        
        # Read existing results
        results = []
        if os.path.exists(RESULTS_FILE) and os.path.getsize(RESULTS_FILE) > 0:
            with open(RESULTS_FILE, 'r') as f:
                try:
                    results = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error reading results file. Creating new file.")
                    results = []
        
        # Append new result
        results.append(result)
        
        # Write updated results back to file
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Saved evaluation result for task ID {task_id} to {RESULTS_FILE}")
    except Exception as e:
        print(f"Error saving evaluation result: {str(e)}")

# Stats tracking
timeout_count = 0
rate_limit_count = 0
total_tokens = 0  # Track total tokens used

def run_npm_start(task_index=0):
    global timeout_count, rate_limit_count, total_tokens
    
    # Get current task
    tasks = load_tasks_from_spreadsheet()
    if task_index >= len(tasks):
        print("All tasks processed!")
        print(f"Final stats:")
        print(f"  Total timeouts: {timeout_count}")
        print(f"  Total rate limit errors: {rate_limit_count}")
        print(f"  Total tokens used: {total_tokens}")
        # Exit the script instead of restarting
        sys.exit(0)
    
    current_task = tasks[task_index]
    task_id = current_task['id']
    website = current_task['website']
    task_description = current_task['task']
    ground_truth_actions = current_task['action_reprs']
    
    # Format the query as requested
    current_query = f"USE {website} to do this task: {task_description}"
    
    print(f"\n{'='*80}")
    print(f"Processing task {task_index+1}/{len(tasks)}: ID={task_id}, Website={website}")
    print(f"Query: '{current_query}'")
    print(f"{'='*80}\n")
    
    # Clear the current task log file
    with open(CURRENT_LOG_FILE, 'w') as f:
        f.write("")

    # Open both log files
    master_log = open(MASTER_LOG_FILE, 'a')
    current_log = open(CURRENT_LOG_FILE, 'a')

    # When logging
    def write_to_logs(line):
        timestamp = datetime.datetime.now().isoformat()
        master_log.write(line)
        current_log.write(line)
        master_log.flush()
        current_log.flush()
        sys.stdout.write(line)
        sys.stdout.flush()

    try:
        timestamp = datetime.datetime.now().isoformat()
        write_to_logs(f'\n\n{timestamp} - TASK {task_index+1}, ID={task_id}: "{current_query}"\n')
        
        # Track tokens for this query
        query_tokens = 0
        
        # Track timing
        browser_init_start = time.time()
        browser_ready = False
        task_start_time = None
        browser_init_time = 0
        rate_limit_wait_time = 0
        
        npm_process = subprocess.Popen(
            ['npm', 'start', current_query],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        killed = False
        buffer = []  # For detecting rate limit in multi-line output
        task_status = "completed"  # Default status
        completed_early = False  # Track if we're exiting the loop early

        # Set timeout end time
        timeout_end = browser_init_start + TIMEOUT_SECONDS

        for line in npm_process.stdout:
            write_to_logs(line)
            # Store last few lines to detect multi-line errors
            buffer.append(line)
            if len(buffer) > 10:
                buffer.pop(0)
            
            # Track token usage from API responses
            if "Total tokens:" in line:
                try:
                    # Extract token count from the line using a clearer regex
                    tokens_match = re.search(r'TOKEN_COUNT: \[stagehand:anthropic\] Total tokens: (\d+) \((\d+) input, (\d+) output\)', line)
                    if tokens_match:
                        total = int(tokens_match.group(1))
                        input_tokens = int(tokens_match.group(2))
                        output_tokens = int(tokens_match.group(3))
                        
                        # Look for our specific formatting prefixes
                        if "FINAL_TOKEN_COUNT: [stagehand:anthropic] Total tokens:" in line:
                            # Update the query token count
                            query_tokens = total
                            total_tokens = total_tokens + total  # Update total tokens
                            
                            # Log token count based on type
                            if "FINAL_TOKEN_COUNT:" in line:
                                print(f"Final token count: {total} ({input_tokens} input, {output_tokens} output)")
                            else:
                                print(f"Token count: {total} ({input_tokens} input, {output_tokens} output)")
                            
                            write_to_logs(f'{timestamp} - Token count detected: {total} ({input_tokens} input, {output_tokens} output)\n')
                except Exception as e:
                    # Log the error without printing the entire line
                    print(f"Error parsing token count: {str(e)}")
                    write_to_logs(f'{timestamp} - Error parsing token count: {str(e)}\n')
            
            # Track when browser is ready to start actual task timing
            if "[stagehand:init] local browser started successfully" in line:
                browser_ready = True
                task_start_time = time.time()
                browser_init_time = task_start_time - browser_init_start
                print(f"Browser initialized in {browser_init_time:.2f} seconds")
                write_to_logs(f'{timestamp} - Browser initialized in {browser_init_time:.2f} seconds\n')
                continue
            
            # Check if rate limit has been hit
            if RATE_LIMIT_ERROR in line or (RETRY_ERROR in line and any(RATE_LIMIT_ERROR in b for b in buffer)):
                # Calculate elapsed time excluding rate limit wait
                elapsed_time = time.time() - task_start_time if task_start_time else time.time() - browser_init_start
                print(f"RATE LIMIT ERROR after {elapsed_time:.2f} seconds, moving to next query")
                write_to_logs(f'{timestamp} - RATE LIMIT ERROR after {elapsed_time:.2f} seconds\n')
                write_to_logs(f'{timestamp} - Tokens used in this query: {query_tokens}\n')
                
                rate_limit_count += 1
                killed = True
                task_status = "rate_limit_error"
                npm_process.terminate()
                npm_process.wait(timeout=5)
                
                # Set flag to exit the loop but still process whatever actions were collected
                completed_early = True
                break
            
            # Check if we've exceeded our timeout
            current_time = time.time()
            if current_time > timeout_end:
                # Calculate elapsed time excluding rate limit wait
                elapsed_time = current_time - task_start_time if task_start_time else current_time - browser_init_start
                print(f"TIMEOUT after {elapsed_time:.2f} seconds, moving to next query")
                write_to_logs(f'{timestamp} - TIMEOUT after {elapsed_time:.2f} seconds\n')
                write_to_logs(f'{timestamp} - Tokens used in this query: {query_tokens}\n')
                
                timeout_count += 1
                killed = True
                task_status = "timeout"
                npm_process.terminate()
                npm_process.wait(timeout=5)
                
                # Set flag to exit the loop but still process whatever actions were collected
                completed_early = True
                break
            
            if FINISH_TRIGGER in line:
                end_time = time.time()
                if task_start_time:
                    task_elapsed_time = end_time - task_start_time
                    total_elapsed_time = end_time - browser_init_start
                    print(f"Task completed in {task_elapsed_time:.2f} seconds (Total: {total_elapsed_time:.2f}s, Browser Init: {browser_init_time:.2f}s, Rate Limit Wait: {rate_limit_wait_time:.2f}s)")
                    print(f"Tokens used in this query: {query_tokens}")
                    write_to_logs(f'{timestamp} - Task completed in {task_elapsed_time:.2f} seconds (Total: {total_elapsed_time:.2f}s, Browser Init: {browser_init_time:.2f}s, Rate Limit Wait: {rate_limit_wait_time:.2f}s)\n')
                    write_to_logs(f'{timestamp} - Tokens used in this query: {query_tokens}\n')
                else:
                    elapsed_time = end_time - browser_init_start
                    task_elapsed_time = elapsed_time
                    total_elapsed_time = elapsed_time
                    print(f"Finished query in {elapsed_time:.2f} seconds")
                    print(f"Tokens used in this query: {query_tokens}")
                    write_to_logs(f'{timestamp} - finished query in {elapsed_time:.2f} seconds\n')
                    write_to_logs(f'{timestamp} - Tokens used in this query: {query_tokens}\n')

                killed = True
                npm_process.terminate()
                npm_process.wait(timeout=5)
                completed_early = True
                break

            if "[AI_RETRYError]" in line:
                # Calculate error time
                end_time = time.time()
                if task_start_time:
                    task_elapsed_time = end_time - task_start_time
                    total_elapsed_time = end_time - browser_init_start
                    print(f"Task failed in {task_elapsed_time:.2f} seconds (Total: {total_elapsed_time:.2f}s, Browser Init: {browser_init_time:.2f}s, Rate Limit Wait: {rate_limit_wait_time:.2f}s)")
                    print(f"Tokens used in this query: {query_tokens}")
                    write_to_logs(f'{timestamp} - Task failed in {task_elapsed_time:.2f} seconds (Total: {total_elapsed_time:.2f}s, Browser Init: {browser_init_time:.2f}s, Rate Limit Wait: {rate_limit_wait_time:.2f}s)\n')
                    write_to_logs(f'{timestamp} - Tokens used in this query: {query_tokens}\n')
                else:
                    # Fallback to old timing if browser ready flag wasn't set
                    elapsed_time = end_time - browser_init_start
                    task_elapsed_time = elapsed_time
                    total_elapsed_time = elapsed_time
                    print(f"GOT AN ERROR in {elapsed_time:.2f} seconds")
                    print(f"Tokens used in this query: {query_tokens}")
                    write_to_logs(f'{timestamp} - Error in {elapsed_time:.2f} seconds\n')
                    write_to_logs(f'{timestamp} - Tokens used in this query: {query_tokens}\n')
                
                killed = True
                task_status = "error"
                npm_process.terminate()
                npm_process.wait(timeout=5)
                completed_early = True
                break

        # If process is still running but we didn't break early, terminate it
        if not killed:
            npm_process.terminate()
            npm_process.wait(timeout=5)
            end_time = time.time()
            if task_start_time:
                task_elapsed_time = end_time - task_start_time
                total_elapsed_time = end_time - browser_init_start
            else:
                total_elapsed_time = end_time - browser_init_start
                task_elapsed_time = total_elapsed_time - browser_init_time
            
            task_status = "unknown_exit"
            
        # If we're here because the process completed normally, get timing
        if not completed_early and not killed:
            end_time = time.time()
            if task_start_time:
                task_elapsed_time = end_time - task_start_time
                total_elapsed_time = end_time - browser_init_start
            else:
                total_elapsed_time = end_time - browser_init_start
                task_elapsed_time = total_elapsed_time - browser_init_time
        
        # Always close log files before extracting actions
        master_log.close()
        current_log.close()
        
        # Now extract stagehand actions from the log file
        print("\nExtracting stagehand actions from log file...")
        stagehand_actions = extract_stagehand_actions(CURRENT_LOG_FILE, task_id)
        print(f"Found {len(stagehand_actions)} stagehand actions:")
        for idx, action in enumerate(stagehand_actions):
            print(f"  {idx+1}. {action}")
        
        # Evaluate stagehand actions against ground truth
        print("\nEvaluating stagehand actions against ground truth...")
        evaluation_result = evaluate_with_llm(stagehand_actions, ground_truth_actions)
        print(f"Evaluation result:")
        print(f"  Accuracy score: {evaluation_result['accuracy_score']}")
        print(f"  Similarity score: {evaluation_result['similarity_score']}")
        print(f"  Explanation: {evaluation_result['explanation']}")
        
        # Save evaluation result to JSON
        print("\nSaving evaluation result to JSON...")
        save_evaluation_result(
            task_id, 
            website, 
            task_description, 
            ground_truth_actions, 
            stagehand_actions, 
            evaluation_result, 
            total_elapsed_time, 
            browser_init_time, 
            task_elapsed_time, 
            query_tokens, 
            task_status
        )
        
        print(f"\nWaiting 5 seconds before processing next task...")
        time.sleep(5)
        
        return run_npm_start(task_index + 1)
        
    except KeyboardInterrupt:
        print('\nReceived KeyboardInterrupt. Cleaning up and exiting...')
        if not master_log.closed:
            master_log.write('\nScript terminated by user\n')
            master_log.close()
        if not current_log.closed:
            current_log.write('\nScript terminated by user\n')
            current_log.close()
        if 'npm_process' in locals():
            npm_process.terminate()
        sys.exit(0)
    except subprocess.TimeoutExpired:
        # Handle the specific case of a timeout when waiting for process termination
        print("Process took too long to terminate, forcing continuation.")
        if not master_log.closed:
            master_log.write("Process took too long to terminate, forcing continuation.\n")
            master_log.close()
        if not current_log.closed:
            current_log.write("Process took too long to terminate, forcing continuation.\n")
            current_log.close()
        
        # Even if the process didn't terminate properly, we still want to extract actions and evaluate them
        # Close log files if still open
        if not master_log.closed:
            master_log.close()
        if not current_log.closed:
            current_log.close()
            
        # Set timing values for the result
        end_time = time.time()
        if task_start_time:
            task_elapsed_time = end_time - task_start_time
            total_elapsed_time = end_time - browser_init_start
        else:
            total_elapsed_time = end_time - browser_init_start
            task_elapsed_time = total_elapsed_time - browser_init_time
            
        task_status = "timeout_on_termination"
        
        # Extract and evaluate actions
        print("\nExtracting stagehand actions from log file...")
        stagehand_actions = extract_stagehand_actions(CURRENT_LOG_FILE, task_id)
        print(f"Found {len(stagehand_actions)} stagehand actions:")
        for idx, action in enumerate(stagehand_actions):
            print(f"  {idx+1}. {action}")
        
        print("\nEvaluating stagehand actions against ground truth...")
        evaluation_result = evaluate_with_llm(stagehand_actions, ground_truth_actions)
        print(f"Evaluation result:")
        print(f"  Accuracy score: {evaluation_result['accuracy_score']}")
        print(f"  Similarity score: {evaluation_result['similarity_score']}")
        print(f"  Explanation: {evaluation_result['explanation']}")
        
        print("\nSaving evaluation result to JSON...")
        save_evaluation_result(
            task_id, 
            website, 
            task_description, 
            ground_truth_actions, 
            stagehand_actions, 
            evaluation_result, 
            total_elapsed_time, 
            browser_init_time, 
            task_elapsed_time, 
            query_tokens, 
            task_status
        )
        
        print(f"\nWaiting 5 seconds before processing next task...")
        time.sleep(5)
        
        return run_npm_start(task_index + 1)
    except Exception as e:
        error_msg = f"Error running npm start for task ID {task_id}: {str(e)}"
        print(error_msg)
        if not master_log.closed:
            master_log.write(f"{error_msg}\n")
            master_log.close()
        if not current_log.closed:
            current_log.write(f"{error_msg}\n")
            current_log.close()
        
        # For all other exceptions, we still want to try to extract and evaluate actions
        # Even if something went wrong in the script itself
        try:
            end_time = time.time()
            if task_start_time:
                task_elapsed_time = end_time - task_start_time
                total_elapsed_time = end_time - browser_init_start
            else:
                total_elapsed_time = end_time - browser_init_start
                task_elapsed_time = total_elapsed_time - browser_init_time
                
            task_status = "script_error"
            
            # Extract and evaluate actions
            print("\nAttempting to extract stagehand actions despite error...")
            stagehand_actions = extract_stagehand_actions(CURRENT_LOG_FILE, task_id)
            print(f"Found {len(stagehand_actions)} stagehand actions:")
            for idx, action in enumerate(stagehand_actions):
                print(f"  {idx+1}. {action}")
            
            print("\nEvaluating stagehand actions against ground truth...")
            evaluation_result = evaluate_with_llm(stagehand_actions, ground_truth_actions)
            print(f"Evaluation result:")
            print(f"  Accuracy score: {evaluation_result['accuracy_score']}")
            print(f"  Similarity score: {evaluation_result['similarity_score']}")
            print(f"  Explanation: {evaluation_result['explanation']}")
            
            print("\nSaving evaluation result to JSON...")
            save_evaluation_result(
                task_id, 
                website, 
                task_description, 
                ground_truth_actions, 
                stagehand_actions, 
                evaluation_result, 
                total_elapsed_time, 
                browser_init_time, 
                task_elapsed_time, 
                query_tokens, 
                task_status
            )
        except Exception as inner_e:
            print(f"Error during recovery after script error: {str(inner_e)}")
            
        print(f"\nWaiting 5 seconds before processing next task...")
        time.sleep(5)
        
        return run_npm_start(task_index + 1)

def main():
    initialize_results_file()
    
    os.makedirs(os.path.dirname(MASTER_LOG_FILE), exist_ok=True)
    
    def signal_handler(sig, frame):
        print('\nReceived SIGINT. Cleaning up and exiting...')
        with open(MASTER_LOG_FILE, 'a') as f:
            f.write('\nScript terminated by user\n')
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    run_npm_start(0)

if __name__ == "__main__":
    main()
