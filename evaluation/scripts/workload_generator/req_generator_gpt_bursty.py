
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
    """
    POSTURL，。
    """
    data = {"text": request_content}
    
    start_time = time.time() 
    current_time = datetime.now().strftime("%d/%b/%Y %H:%M:%S") 
    print(f"Request {request_id} sent at {current_time}")
    
    response = requests.post(url, json=data)
    end_time = time.time() 
    
    input_length = len(request_content.split(' '))
    output_lenght = len(response.json()['prediction'].split(' '))
    latency = (end_time - start_time) * 1000 / (output_lenght-input_length)
    with latency_lock: 
        latencies.append(latency)
    if response.status_code == 200:
        print(f"Request {request_id} completed successfully. Latency: {latency:.2f} ms")
    else:
        print(f"Request {request_id} failed with status {response.status_code}. Latency: {latency:.2f} ms")


def model_prewarming(urls):
    threads = []
    for url in urls:
        request_content = ' '.join(random.choices(words_list, k=32))
        for i in range(5):
            thread = threading.Thread(target=send_request, args=(i, request_content, url))
            thread.start()
            threads.append(thread)
            time.sleep(1/2)
    
    for thread in threads:
        thread.join()

def thread_manager(num_requests, urls):
    threads = []

    # prewarming
    model_prewarming(urls)
    with open('./workloads/workload/bursty.txt', 'r') as file:
        numbers = file.readlines()
    random_data = [float(num.strip())/3 for num in numbers]
 
    
    random_index = 0
    request_id = 0
    while num_requests>0:
    # for i in range(num_requests):
        rate = int(random_data[random_index])
        random_index += 1
        for _ in range(rate):
            request_content = ' '.join(random.choices(words_list, k=32))
            thread = threading.Thread(target=send_request, args=(request_id, request_content, urls[request_id%len(urls)]))
            thread.start()
            threads.append(thread)
            time.sleep(1 / rate) 
            request_id+=1
            num_requests -= 1
            if num_requests<=0:
                break


    for thread in threads:
        thread.join()
    latencies_array = np.array(latencies[5:])
    print("Latency Mean:", np.mean(latencies_array))
    print("Latency P1:", np.percentile(latencies_array, 1))   
    print("Latency P50:", np.percentile(latencies_array, 50))
    print("Latency P90:", np.percentile(latencies_array, 90))
    print("Latency P95:", np.percentile(latencies_array, 95))
    print("Latency P99:", np.percentile(latencies_array, 99))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-req_rate', '--req_rate', default=1, type=float)
    parser.add_argument('-req_nums', '--req_nums', default=4, type=int)
    args = parser.parse_args()
    
    num_requests = args.req_nums
    urls = [
        "http://localhost:15000/predict",
        # "http://localhost:15001/predict",
        # "http://localhost:15002/predict",
        # "http://localhost:15003/predict",
        # "http://localhost:15004/predict",
        # "http://localhost:15005/predict",
        ]
    thread_manager(num_requests, urls)