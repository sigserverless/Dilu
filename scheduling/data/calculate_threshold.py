import numpy as np

def read_request_data(filepath):
    with open(filepath, 'r') as file:
        data = file.readlines()
    return [float(line.strip())*25 for line in data]

def calculate_intervals(data):
    intervals = []
    current_interval = 0
    for rps in data:
        if rps > 0:
            if current_interval > 0:
                intervals.append(current_interval)
                current_interval = 0
        else:
            current_interval += 1
    if current_interval > 0:
        intervals.append(current_interval)
    return intervals

def calculate_percentiles(intervals, percentiles=[5, 90]):
    if not intervals: 
        return [0 for _ in percentiles]
    return np.percentile(intervals, percentiles)

def compute_pre_warm_keep_alive(data, long_period_length, short_period_length, gamma=0.5):
    long_data = data[:long_period_length]
    short_data = data[:short_period_length]
    
    long_intervals = calculate_intervals(long_data)
    short_intervals = calculate_intervals(short_data)
    
    long_percentiles = calculate_percentiles(long_intervals)
    short_percentiles = calculate_percentiles(short_intervals)
    
    pre_warm_time = gamma * long_percentiles[0] + (1 - gamma) * short_percentiles[0]
    keep_alive_time = gamma * long_percentiles[1] + (1 - gamma) * short_percentiles[1]
    
    return pre_warm_time, keep_alive_time

data = read_request_data('./workloads/bursty.txt')
if data:
    pre_warm, keep_alive = compute_pre_warm_keep_alive(data, long_period_length=858, short_period_length=200)
    print(f"Pre-warm time: {pre_warm} seconds")
    print(f"Keep-alive time: {keep_alive} seconds")
else:
    print("No data available to process.")

