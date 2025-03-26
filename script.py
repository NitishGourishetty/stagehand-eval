'''
This script will run the stagehand agent.

Input: List of queries and expected actions for each query.

Based on it will find all the stagehand actions (act, observe, extract, etc.) it took to answer the query.
After finding the stagehand actions, it will compare it against the expected actions and give an accuracy score.

Output:
1) Accuracy score (placeholder for now)
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


LOG_FILE = os.path.join(os.path.dirname(__file__), 'everything.log')
FINISH_TRIGGER = "---FINISHED---"
TIMEOUT_SECONDS = 300  # 4 minutes timeout
RATE_LIMIT_ERROR = "rate_limit_error"
RETRY_ERROR = "RetryError"
SPREADSHEET_PATH = "mind2web_train_filtered.xlsx"  # Path to your spreadsheet

# Load tasks from spreadsheet instead of hardcoded queries
def load_tasks_from_spreadsheet():
    print("Loading tasks from spreadsheet")
    try:
        df = pd.read_excel(SPREADSHEET_PATH)
        tasks = []
        for _, row in df.iterrows():
            task_id = row['id']
            website = row['website']
            task = row['confirmed_task']
            tasks.append({
                'id': task_id,
                'website': website,
                'task': task
            })
        return tasks
    except Exception as e:
        print(f"Error loading spreadsheet: {str(e)}")
        sys.exit(1)

# Replace the hardcoded QUERIES with tasks from spreadsheet
TASKS = load_tasks_from_spreadsheet()

# Stats tracking
timeout_count = 0
rate_limit_count = 0

def run_npm_start(task_index=0):
    global timeout_count, rate_limit_count
    
    # Get current task
    if task_index >= len(TASKS):
        print("All tasks processed!")
        print(f"Final stats:")
        print(f"  Total timeouts: {timeout_count}")
        print(f"  Total rate limit errors: {rate_limit_count}")
        # Exit the script instead of restarting
        sys.exit(0)
    
    current_task = TASKS[task_index]
    task_id = current_task['id']
    website = current_task['website']
    task_description = current_task['task']
    
    # Format the query as requested
    current_query = f"USE {website} to do this task: {task_description}"
    
    print(f"Processing task {task_index+1}/{len(TASKS)}: ID={task_id}, Website={website}")
    print(f"Query: '{current_query}'")
    
    # Open the log files
    log_file = open(LOG_FILE, 'a')
    
    try:
        timestamp = datetime.datetime.now().isoformat()
        log_file.write(f'\n\n{timestamp} - TASK {task_index+1}, ID={task_id}: "{current_query}"\n')
        log_file.flush()
        
        # Start timing the query
        start_time = time.time()
        
        npm_process = subprocess.Popen(
            ['npm', 'start', current_query],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        killed = False
        in_stagehand_tag = False
        stagehand_content = []
        buffer = []  # For detecting rate limit in multi-line output

        # Set timeout end time
        timeout_end = start_time + TIMEOUT_SECONDS

        for line in npm_process.stdout:
            # Store last few lines to detect multi-line errors
            buffer.append(line)
            if len(buffer) > 10:
                buffer.pop(0)
            
            # Check if rate limit has been hit
            if RATE_LIMIT_ERROR in line or (RETRY_ERROR in line and any(RATE_LIMIT_ERROR in b for b in buffer)):
                elapsed_time = time.time() - start_time
                print(f"RATE LIMIT ERROR after {elapsed_time:.2f} seconds, moving to next query")
                log_file.write(f'{timestamp} - RATE LIMIT ERROR after {elapsed_time:.2f} seconds\n')
                
                rate_limit_count += 1
                killed = True
                npm_process.terminate()
                npm_process.wait(timeout=5)
                
                # Add sleep to allow rate limits to reset
                wait_time = 60  # Wait 60 seconds before next query to help with rate limits
                print(f"Waiting {wait_time} seconds before next query...")
                time.sleep(wait_time)
                
                log_file.close()
                return run_npm_start(task_index + 1)
            
            # Check if we've exceeded our timeout
            current_time = time.time()
            if current_time > timeout_end:
                elapsed_time = current_time - start_time
                print(f"TIMEOUT after {elapsed_time:.2f} seconds, moving to next query")
                log_file.write(f'{timestamp} - TIMEOUT after {elapsed_time:.2f} seconds\n')
                
                timeout_count += 1
                killed = True
                npm_process.terminate()
                npm_process.wait(timeout=5)
                
                time.sleep(5)
                
                log_file.close()
                return run_npm_start(task_index + 1)
            
            log_file.write(line)
            log_file.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
            
            if "<stagehand>" in line:
                in_stagehand_tag = True
                start_pos = line.find("<stagehand>") + len("<stagehand>")
                stagehand_content.append(line[start_pos:])
                continue
            
            if "</stagehand>" in line:
                in_stagehand_tag = False
                end_pos = line.find("</stagehand>")
                if end_pos > 0:
                    stagehand_content.append(line[:end_pos])
                
                stagehand_log = ''.join(stagehand_content)
                
                stagehand_content = []
                continue
            
            if in_stagehand_tag:
                stagehand_content.append(line)

            if FINISH_TRIGGER in line:
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                print(f"Finished query in {elapsed_time:.2f} seconds, moving to next query")
                log_file.write(f'{timestamp} - finished query in {elapsed_time:.2f} seconds\n')                

                # can calculate accuracy here or later
                
                killed = True
                npm_process.terminate()
                npm_process.wait(timeout=5)

                time.sleep(5)
                
                log_file.close()
                return run_npm_start(task_index + 1)

            if "[AI_RETRYError]" in line:
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                print(f"GOT AN ERROR in {elapsed_time:.2f} seconds, moving to next query")
                log_file.write(f'{timestamp} - finished query in {elapsed_time:.2f} seconds\n')                
                
                killed = True
                npm_process.terminate()
                npm_process.wait(timeout=5)

                time.sleep(5)
                
                log_file.close()
                return run_npm_start(task_index + 1)


        # If we get here, the process exited without seeing FINISH_TRIGGER
        # OR API CALL ERROR -> FIX IT!!!!!!!
        exit_code = npm_process.wait()
        print(f"Process exited with code {exit_code} without finishing")
        log_file.write(f"Process exited with code {exit_code} without finishing\n")

        if not killed:
            print("Process terminated unexpectedly, restarting same query...")
            log_file.write("Restarting same query\n")
            log_file.flush()
            time.sleep(1)
            
            log_file.close()
            return run_npm_start(task_index)
        
    except KeyboardInterrupt:
        print('\nReceived KeyboardInterrupt. Cleaning up and exiting...')
        log_file.write('\nScript terminated by user\n')
        log_file.close()
        npm_process.terminate()
        sys.exit(0)
    except Exception as e:
        error_msg = f"Error running npm start for task ID {task_id}: {str(e)}"
        print(error_msg)
        log_file.write(f"{error_msg}\n")
        log_file.flush()
        log_file.close()
        time.sleep(1)
        return run_npm_start(task_index)
    finally:
        if not log_file.closed:
            log_file.close()

def main():
    if not os.path.exists(LOG_FILE):
        open(LOG_FILE, 'w').close()
    
    # ctrl c
    def signal_handler(sig, frame):
        print('\nReceived SIGINT. Cleaning up and exiting...')
        with open(LOG_FILE, 'a') as f:
            f.write('\nScript terminated by user\n')
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    run_npm_start(0)

if __name__ == "__main__":
    main()