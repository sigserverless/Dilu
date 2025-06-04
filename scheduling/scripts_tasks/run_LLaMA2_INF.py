from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel
import time
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from flask import Flask, request, jsonify
import argparse
import torch
import os
import threading
import time
import queue
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
parser = argparse.ArgumentParser(description="Flask server for running a transformers model")
parser.add_argument("--model_name_or_path", required=True, help="Path to pretrained model or model identifier from huggingface.co/models")
parser.add_argument("--device", default='0,1,2,3', type=str, help="Device to run the model on")
parser.add_argument("--port", default=15000, type=int, help="Port to run the Flask app on")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
print(os.environ["CUDA_VISIBLE_DEVICES"])

app = Flask(__name__)



model_path = args.model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
torch.cuda.synchronize()
time.sleep(60) 
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
model.eval()

batch_queue = []
batch_size = 8  
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
                wait_time = MIN_WAITING_DURATION if average_et < 1/RPS else 1/8*(average_et) 


def batch_inference():
    while True:
        if len(batch_queue)!=0:
            start_time = time.time()
            global wait_time
            while time.time() - start_time < wait_time and len(batch_queue) < batch_size: 
                time.sleep(MIN_WAITING_DURATION) 
            with thread_lock:
                texts_to_process = [item['text'] for item in list(batch_queue)]
                result_queues = [item['result_queue'] for item in list(batch_queue)]
                batch_queue.clear()

            predictions = inference(texts_to_process)
            for pred, res_queue in zip(predictions, result_queues):
                res_queue.put(pred)

def inference(sentences):
    # input_ids = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda:0")
    with torch.no_grad():
        input_ids = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).input_ids 
        # input_ids = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda") # for inf-inf
        generated_text = model.generate(
                input_ids,
                max_new_tokens=16,
                num_return_sequences=1,  
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                max_length=64,
                do_sample=False
            )
    decoded_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_text]
    return decoded_texts


@app.route('/health', methods=['GET'])
def health_check():
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)  
        print("allocated_memory: ", allocated_memory)
        if allocated_memory > 14: 
            return jsonify({'status': 'healthy'}), 200
        else:
            return jsonify({'status': 'unhealthy', 'reason': 'Not enough GPU memory allocated'}), 503
    else:
        return jsonify({'status': 'unhealthy', 'reason': 'GPU not available'}), 503


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        st = time.time()
        data = request.json
        text = data['text']
        result_queue = queue.Queue()
        with thread_lock:
            batch_queue.append({'text': text, 'result_queue': result_queue})
        prediction = result_queue.get()  
        et = time.time()
        with exec_time_list_lock:
            exec_time_list.append(et-st)
        return jsonify({'prediction': prediction})

@app.route('/shutdown', methods=['GET'])
def shutdown():
    os._exit(0)


if __name__ == "__main__":
    thread_lock = threading.Lock()
    threading.Thread(target=batch_inference, daemon=True).start()
    threading.Thread(target=RPS_monitor, daemon=True).start()
    app.run(debug=False, port=args.port)



