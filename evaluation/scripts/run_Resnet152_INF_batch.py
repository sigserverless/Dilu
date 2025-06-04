from flask import Flask, request, jsonify
import argparse
import torch
import threading
import queue
import cv_models as model_zoo
import time
from concurrent.futures import ThreadPoolExecutor
import os
import io
import pickle
parser = argparse.ArgumentParser(description="Flask server for running a ResNet152 model")
parser.add_argument("--device", default=0, type=int, help="Device to run the model on")
parser.add_argument("--port", default=15000, type=int, help="Port to run the Flask app on")
args = parser.parse_args()

app = Flask(__name__)

model = model_zoo.resnet152(pretrained=False)
model.to(args.device)

batch_queue = []
batch_size = 16  

wait_time = 0.02  
wait_time_lock = threading.Lock() 
exec_time_list = []
exec_time_list_lock = threading.Lock()
MONITOR_DURATION = 1
MIN_WAITING_DURATION = 0.00001


def RPS_monitor():
    global wait_time, wait_time_lock, exec_time_list_lock, exec_time_list
    while True:
       
        with exec_time_list_lock:
            exec_time_list.clear()
        time.sleep(MONITOR_DURATION)
        # update wait_time
        with wait_time_lock and exec_time_list_lock:
            if len(exec_time_list)!=0:
                RPS = len(exec_time_list)/MONITOR_DURATION
                average_et = 0
                for et in exec_time_list:
                    average_et += et
                average_et /= len(exec_time_list)
                wait_time = MIN_WAITING_DURATION if average_et < 1/RPS else 1/10*(average_et) 
                print("MIN_WAITING_DURATION: ", wait_time)
            


def batch_inference():
    global wait_time
    while True:
        if len(batch_queue) != 0:
            start_time = time.time()
            while time.time() - start_time < wait_time and len(batch_queue) < batch_size:
                time.sleep(0.0001) 
            
            with thread_lock:
                data_Xs = [item['image'].unsqueeze(0) for item in batch_queue]  
                result_queues = [item['result_queue'] for item in batch_queue]
                
                data_X = torch.cat(data_Xs)
                batch_queue.clear()

            with torch.no_grad():
                output = model(data_X)
                predictions = output.cpu().numpy().tolist()
                
            for pred, res_queue in zip(predictions, result_queues):
                res_queue.put({'prediction': pred})



# def batch_inference():
#     import torch.nn as nn
#     import torch.optim as optim
#     global model 
#     num_epochs = 1
#     batch_size = 64
#     iters = 1
#     train_X = torch.rand([batch_size*iters, 3, 224, 224])
#     train_Y = torch.randint(1,100,[batch_size*iters])
#     train_loader = torch.utils.data.DataLoader(
#             torch.utils.data.TensorDataset(train_X, train_Y),
#             batch_size=batch_size,
#     )
    
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#     model = model.to("cuda:0")  # Move the model to the appropriate device
#     model.train()
#     for epoch in range(num_epochs):
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to("cuda:0"), labels.to("cuda:0")
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
    
    
#     result_queues = [item['result_queue'] for item in batch_queue]
#     predictions= [ 111 for _ in range(len(result_queues))]
#     for pred, res_queue in zip(predictions, result_queues):
#         res_queue.put({'prediction': pred})


def predict():

    # if 'image' not in request.files:
    #     return jsonify({'error': 'No image provided'}), 400
    # image_file = request.files['image']
    # image_data = io.BytesIO(image_file.read())
    # #  NumPy 
    # image_data.seek(0)
    # image_array = pickle.load(image_data)
    # image_tensor = torch.from_numpy(image_array).float().to(args.device)  
    image_tensor = torch.rand([3, 224, 224]).to(args.device)
    st = time.time()
    result_queue = queue.Queue()
    with thread_lock:
        batch_queue.append({'image': image_tensor, 'result_queue': result_queue})
    prediction = result_queue.get()  # 
    et = time.time()
    with exec_time_list_lock:
        exec_time_list.append(et-st)
    return jsonify({'prediction': prediction})

@app.route('/health', methods=['GET'])
def health_check():
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)  
        print("allocated_memory: ", allocated_memory)
        if allocated_memory > 0.2: 
            return jsonify({'status': 'healthy'}), 200
        else:
            return jsonify({'status': 'unhealthy', 'reason': 'Not enough GPU memory allocated'}), 503
    else:
        return jsonify({'status': 'unhealthy', 'reason': 'GPU not available'}), 503

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    return predict()

@app.route('/shutdown', methods=['GET'])
def shutdown():
    os._exit(0)

if __name__ == '__main__':
    thread_lock = threading.Lock()
    threading.Thread(target=batch_inference, daemon=True).start()
    threading.Thread(target=RPS_monitor, daemon=True).start()
    app.run(debug=False, port=args.port)
