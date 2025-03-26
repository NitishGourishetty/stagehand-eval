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


LOG_FILE = os.path.join(os.path.dirname(__file__), 'everything.log')
# FINISH_TRIGGER = "---FINISHED---"
FINISH_TRIGGER = "FINAL_TOKEN_COUNT:"
TIMEOUT_SECONDS = 300  # 3 minutes timeout
RATE_LIMIT_ERROR = "rate_limit_error"
RETRY_ERROR = "RetryError"

# List of queries to process
QUERIES = [
    "What is the weather in San Francisco",
]

# Stats tracking
timeout_count = 0
rate_limit_count = 0
total_tokens = 0  # Track total tokens used

def run_npm_start(query_index=0):
    global timeout_count, rate_limit_count, total_tokens
    
    # Get current query
    if query_index >= len(QUERIES):
        print("All queries processed!")
        print(f"Final stats:")
        print(f"  Total timeouts: {timeout_count}")
        print(f"  Total rate limit errors: {rate_limit_count}")
        print(f"  Total tokens used: {total_tokens}")
        # Exit the script instead of restarting
        sys.exit(0)
    
    current_query = QUERIES[query_index]
    print(f"Processing query {query_index+1}/{len(QUERIES)}: '{current_query}'")
    
    # Open the log files
    log_file = open(LOG_FILE, 'a')
    
    try:
        timestamp = datetime.datetime.now().isoformat()
        log_file.write(f'\n\n{timestamp} - QUERY {query_index+1}: "{current_query}"\n')
        log_file.flush()
        
        # Track tokens for this query
        query_tokens = 0
        
        # NEW: Track browser initialization and actual task timing separately
        browser_init_start = time.time()
        browser_ready = False
        task_start_time = None
        rate_limit_wait_time = 0
        
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
        timeout_end = browser_init_start + TIMEOUT_SECONDS

        for line in npm_process.stdout:
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
                            print("at least we detected it")
                            total_tokens = total  # Update total tokens
                            # Update the query token count
                            query_tokens = total
                            total_tokens = total_tokens - query_tokens + total  # Update total tokens
                            
                            # Log token count based on type
                            if "FINAL_TOKEN_COUNT:" in line:
                                print(f"Final token count: {total} ({input_tokens} input, {output_tokens} output)")
                            else:
                                print(f"Token count: {total} ({input_tokens} input, {output_tokens} output)")
                            
                            log_file.write(f'{timestamp} - Token count detected: {total} ({input_tokens} input, {output_tokens} output)\n')
                except Exception as e:
                    # Log the error without printing the entire line
                    print(f"Error parsing token count: {str(e)}")
                    log_file.write(f'{timestamp} - Error parsing token count: {str(e)}\n')
            
            # NEW: Track when browser is ready to start actual task timing
            if "[stagehand:init] local browser started successfully" in line:
                browser_ready = True
                task_start_time = time.time()
                browser_init_time = task_start_time - browser_init_start
                print(f"Browser initialized in {browser_init_time:.2f} seconds")
                log_file.write(f'{timestamp} - Browser initialized in {browser_init_time:.2f} seconds\n')
                continue
            
            # Check if rate limit has been hit
            if RATE_LIMIT_ERROR in line or (RETRY_ERROR in line and any(RATE_LIMIT_ERROR in b for b in buffer)):
                # NEW: Calculate elapsed time excluding rate limit wait
                elapsed_time = time.time() - task_start_time if task_start_time else time.time() - browser_init_start
                print(f"RATE LIMIT ERROR after {elapsed_time:.2f} seconds, moving to next query")
                log_file.write(f'{timestamp} - RATE LIMIT ERROR after {elapsed_time:.2f} seconds\n')
                log_file.write(f'{timestamp} - Tokens used in this query: {query_tokens}\n')
                
                rate_limit_count += 1
                killed = True
                npm_process.terminate()
                npm_process.wait(timeout=5)
                
                # Add sleep to allow rate limits to reset
                wait_time = 60  # Wait 60 seconds before next query to help with rate limits
                print(f"Waiting {wait_time} seconds before next query...")
                rate_limit_wait_time += wait_time
                time.sleep(wait_time)
                
                log_file.close()
                return run_npm_start(query_index + 1)
            
            # Check if we've exceeded our timeout
            current_time = time.time()
            if current_time > timeout_end:
                # NEW: Calculate elapsed time excluding rate limit wait
                elapsed_time = current_time - task_start_time if task_start_time else current_time - browser_init_start
                print(f"TIMEOUT after {elapsed_time:.2f} seconds, moving to next query")
                log_file.write(f'{timestamp} - TIMEOUT after {elapsed_time:.2f} seconds\n')
                log_file.write(f'{timestamp} - Tokens used in this query: {query_tokens}\n')
                
                timeout_count += 1
                killed = True
                npm_process.terminate()
                npm_process.wait(timeout=5)
                
                time.sleep(5)
                
                log_file.close()
                return run_npm_start(query_index + 1)
            
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
                # NEW: Calculate task completion time excluding browser init and rate limit waits
                end_time = time.time()
                if task_start_time:
                    task_elapsed_time = end_time - task_start_time
                    total_elapsed_time = end_time - browser_init_start
                    print(f"Task completed in {task_elapsed_time:.2f} seconds (Total: {total_elapsed_time:.2f}s, Browser Init: {browser_init_time:.2f}s, Rate Limit Wait: {rate_limit_wait_time:.2f}s)")
                    print(f"Tokens used in this query: {query_tokens}")
                    log_file.write(f'{timestamp} - Task completed in {task_elapsed_time:.2f} seconds (Total: {total_elapsed_time:.2f}s, Browser Init: {browser_init_time:.2f}s, Rate Limit Wait: {rate_limit_wait_time:.2f}s)\n')
                    log_file.write(f'{timestamp} - Tokens used in this query: {query_tokens}\n')
                else:
                    # Fallback to old timing if browser ready flag wasn't set
                    elapsed_time = end_time - browser_init_start
                    print(f"Finished query in {elapsed_time:.2f} seconds")
                    print(f"Tokens used in this query: {query_tokens}")
                    log_file.write(f'{timestamp} - finished query in {elapsed_time:.2f} seconds\n')
                    log_file.write(f'{timestamp} - Tokens used in this query: {query_tokens}\n')

                killed = True
                npm_process.terminate()
                npm_process.wait(timeout=5)

                time.sleep(5)
                
                log_file.close()
                return run_npm_start(query_index + 1)

            if "[AI_RETRYError]" in line:
                # NEW: Calculate error time excluding browser init and rate limit waits
                end_time = time.time()
                if task_start_time:
                    task_elapsed_time = end_time - task_start_time
                    total_elapsed_time = end_time - browser_init_start
                    print(f"Task failed in {task_elapsed_time:.2f} seconds (Total: {total_elapsed_time:.2f}s, Browser Init: {browser_init_time:.2f}s, Rate Limit Wait: {rate_limit_wait_time:.2f}s)")
                    print(f"Tokens used in this query: {query_tokens}")
                    log_file.write(f'{timestamp} - Task failed in {task_elapsed_time:.2f} seconds (Total: {total_elapsed_time:.2f}s, Browser Init: {browser_init_time:.2f}s, Rate Limit Wait: {rate_limit_wait_time:.2f}s)\n')
                    log_file.write(f'{timestamp} - Tokens used in this query: {query_tokens}\n')
                else:
                    # Fallback to old timing if browser ready flag wasn't set
                    elapsed_time = end_time - browser_init_start
                    print(f"GOT AN ERROR in {elapsed_time:.2f} seconds")
                    print(f"Tokens used in this query: {query_tokens}")
                    log_file.write(f'{timestamp} - finished query in {elapsed_time:.2f} seconds\n')
                    log_file.write(f'{timestamp} - Tokens used in this query: {query_tokens}\n')
                
                killed = True
                npm_process.terminate()
                npm_process.wait(timeout=5)

                time.sleep(5)
                
                log_file.close()
                return run_npm_start(query_index + 1)


        # If we get here, the process exited without seeing FINISH_TRIGGER
        # Handle process exit with error code
        end_time = time.time()
        exit_code = npm_process.wait()
        
        # NEW: Calculate timing for unexpected exit
        if task_start_time:
            task_elapsed_time = end_time - task_start_time
            total_elapsed_time = end_time - browser_init_start
            error_msg = f"Process exited with code {exit_code} after {task_elapsed_time:.2f} seconds (Total: {total_elapsed_time:.2f}s, Browser Init: {browser_init_time:.2f}s, Rate Limit Wait: {rate_limit_wait_time:.2f}s)"
        else:
            elapsed_time = end_time - browser_init_start
            error_msg = f"Process exited with code {exit_code} after {elapsed_time:.2f} seconds"
            
        print(error_msg)
        log_file.write(f"{error_msg}\n")

        # NEW: Handle npm exit codes
        if exit_code != 0:
            print(f"Process failed with exit code: {exit_code}")
            log_file.write(f"Process failed with exit code: {exit_code}\n")

        if not killed:
            print("Process terminated unexpectedly, restarting same query...")
            log_file.write("Restarting same query\n")
            log_file.flush()
            time.sleep(1)
            
            log_file.close()
            return run_npm_start(query_index)
        
    except KeyboardInterrupt:
        print('\nReceived KeyboardInterrupt. Cleaning up and exiting...')
        log_file.write('\nScript terminated by user\n')
        log_file.close()
        npm_process.terminate()
        sys.exit(0)
    except Exception as e:
        error_msg = f"Error running npm start: {str(e)}"
        print(error_msg)
        log_file.write(f"{error_msg}\n")
        log_file.flush()
        log_file.close()
        time.sleep(1)
        return run_npm_start(query_index)
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
