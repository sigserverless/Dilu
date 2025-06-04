
import requests
import time
import random
import threading
from datetime import datetime
import numpy as np
import argparse
np.random.seed(42)  


latencies = []
latency_lock = threading.Lock()
words_list = ["example", "word", "list", "random", "content", "text", "request", "response", "test", "flask", "localhost", "server", "async", "http", "post", "get", "python", "programming", "code", "function"]


def send_request(request_id, request_content, url):
    data = {"text": request_content}
    start_time = time.time()      
    current_time = datetime.now().strftime("%d/%b/%Y %H:%M:%S")           
    response = requests.post(url, json=data)
    end_time = time.time()         
    latency = (end_time - start_time) * 1000    
    if response.status_code == 200:
        with latency_lock:              
            latencies.append(latency)
        print(f"Request {request_id} sent at {current_time} completed successfully. Latency: {latency:.2f} ms")
    else:
        with latency_lock:            
            latencies.append(2000)
        print(f"Request {request_id} sent at {current_time} failed with status {response.status_code}. Latency: 2000 ms")

def model_prewarming(urls):
    threads = []
    for url in urls:
        request_content = ' '.join(random.choices(words_list, k=128))
        for i in range(5):
            thread = threading.Thread(target=send_request, args=(i, request_content, url))
            thread.start()
            threads.append(thread)
            time.sleep(1/5)
    
    for thread in threads:
        thread.join()
        

def thread_manager(file_path, urls):
    threads = []
    try:
        with open(file_path, 'r') as file:
            rates = [float(line.strip()) * 10 for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return

    model_prewarming(urls)
    
    request_id = 0
    for rate in rates:
        for _ in range(int(rate)):  # Assumed that rates can be fractional and need to be rounded
            request_content = ' '.join(random.choices(words_list, k=128))
            thread = threading.Thread(target=send_request, args=(request_id, request_content, urls[request_id % len(urls)]))
            thread.start()
            threads.append(thread)
            time.sleep(1 / max(rate, 1))  # Avoid division by zero
            request_id += 1

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    if latencies:
        latencies_array = np.array(latencies)
        print("Latency Mean:", np.mean(latencies_array))
        print("Latency P1:", np.percentile(latencies_array, 1))   
        print("Latency P50:", np.percentile(latencies_array, 50))
        print("Latency P90:", np.percentile(latencies_array, 90))
        print("Latency P95:", np.percentile(latencies_array, 95))
        print("Latency P99:", np.percentile(latencies_array, 99))
    else:
        print("No latencies recorded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', '--file', required=True, type=str)
    parser.add_argument('-func_name', '--func_name', default='', type=str)
    args = parser.parse_args()
    
    urls = ["http://172.16.120.44:14999/{}".format(args.func_name)]
    thread_manager(args.file, urls)
