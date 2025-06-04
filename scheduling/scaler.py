import threading
import time
import http
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
import random
import requests
import json
from urllib.parse import urlparse, parse_qs
from flask import Flask, request, jsonify
import json
from datetime import datetime
import logging

import logging
from datetime import datetime

logging.basicConfig(filename='logs/dilu-scaler.log', 
                    filemode='w', 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    datefmt='%d/%b/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)
app.logger.disabled = True
services_status = {}
services_dict = {} 
lock = threading.Lock()
scheduler_url = 'http://localhost:5000'


class ServiceManager:
    def __init__(self):
        self.services = {}

    def register_service(self, service_data):
        global services_dict, services_status
        service_id = service_data['service_name']
        if service_id in services_dict:
            return "Service already registered", 409
        # update services_status
        new_service = Service(service_id, service_data['image'], service_data['num_gpus'], service_data['sm_requests'], service_data['sm_limits'], service_data['memory'], service_data['is_llm'], service_data['priority'], service_data['task_type'], service_data['throughput'], service_data['commands'])
        services_dict[service_id] = new_service
        services_status[service_id] = {"requests": 0}
        
        return f"Service {service_id} registered successfully", 200

    def delete_service(self, service_data):
        global services_dict, services_status
        service_id = service_data['service_name']
        if service_id in services_dict:
            del services_dict[service_id]
            del services_status[service_id]
            return f"Service {service_id} deleted successfully", 200
        else:
            return "Service does not exist", 404



@app.route('/register_service', methods=['POST'])
def register_service():
    service_data = request.get_json()
    message, status = service_manager.register_service(service_data)
    return jsonify({'message': message}), status

@app.route('/<service_id>', methods=['POST'])
def handle_predict(service_id):
    if service_id in services_dict:
        services_status[service_id]['requests'] += 1  
        service = services_dict[service_id]
        return service.dispatch(request)  
    else:
        return jsonify({'error': 'Service not found'}), 404




# class RequestHandler(BaseHTTPRequestHandler):
#     def do_POST(self):
#         content_length = int(self.headers['Content-Length'])
#         post_data = self.rfile.read(content_length).decode('utf-8')
#         post_data = json.loads(post_data)

#         if self.path == '/register_service':
#             message, status = service_manager.register_service(post_data)
#             self.send_response(status)
#             self.send_header('Content-type', 'application/json')
#             self.end_headers()
#             response = {'message': message}
#             self.wfile.write(json.dumps(response).encode())
#         else: # Service      
#             parsed_path = urlparse(self.path)
#             service_id = parsed_path.path.strip('/').split('/')[0]
#             if service_id in services_dict:
#                 service = services_dict[service_id]
#                 services_status[service_id]['requests'] += 1 # update 
#                 service.dispatch(self)
#             else:
#                 self.send_error(404, "Service not found")


# def runRH(server_class=HTTPServer, handler_class=RequestHandler, port=14999):
#     server_address = ('0.0.0.0', port)
#     httpd = server_class(server_address, handler_class)
#     print(f"Starting httpd server on port {port}...")
#     httpd.serve_forever()


class Service:
    def __init__(self, service_id, image, num_gpus, sm_requests, sm_limits, memory, is_llm, priority, task_type, throughput, commands):
        self.service_id = service_id
        self.image = image
        self.num_gpus = num_gpus
        self.sm_requests = sm_requests
        self.sm_limits = sm_limits
        self.memory = memory
        self.is_llm = is_llm
        self.priority = priority
        self.task_type = task_type
        self.commands = commands
        self.throughput = throughput
        self.instances = []
        self.round_robin_index = 0
        self.lock = threading.Lock()

        self.scale_out()

    def scale_out(self):
        url = f"{scheduler_url}/schedule"
        data = {
            'num': self.num_gpus,
            'sm_requests': self.sm_requests,
            'sm_limits': self.sm_limits,
            'memory': self.memory,
            'is_llm': self.is_llm,
            'priority': self.priority,
            'type': self.task_type,
            'service_name': self.service_id,
            'image': self.image,
            'COMMAND': self.commands,
        }
        response = requests.post(url, json=data)
        if response.status_code == 200:
            resp_content = response.json()
            instance_id = resp_content.get('instance_id')
            ip_address = resp_content.get('selected_gpus')[0]['ip']
            port = resp_content.get('port')
            instance = {"instance_id": instance_id, "ip_address": ip_address, "port": port, "is_ready": False}
            self.instances.append(instance)
            current_time = datetime.now().strftime("%d/%b/%Y %H:%M:%S")       
            print(f"Instance deployed: {instance_id}, current_time: {current_time}, current_instance_counts: {len(self.instances)}")
            # Start a thread to check readiness
            if "training" not in self.service_id:
                threading.Thread(target=self.check_instance_readiness, args=(instance,)).start()
        else:
            print(f'Instance deploy failed:', response.content)

    def check_instance_readiness(self, instance):
        """Check if the instance is ready to receive requests."""
        while True:
            try:
                response = requests.get(f"http://{instance['ip_address']}:{instance['port']}/health")
                
                if response.status_code == 200:
                    with self.lock:
                        time.sleep(1)
                        instance['is_ready'] = True
                    break
            except requests.RequestException:
                pass
            time.sleep(1)  # Check every second until the instance is ready

    def scale_in(self):
        with self.lock:
            instance_to_remove = self.instances.pop()  # Remove the first instance
        url = f"{scheduler_url}/delete_instance" 
        data = {'instance_id': instance_to_remove['instance_id']}
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            current_time = datetime.now().strftime("%d/%b/%Y %H:%M:%S")      
            print(f"Instance deleted: {instance_to_remove['instance_id']}, current_time: {current_time}, current_instance_counts: {len(self.instances)}")
        else:
            print(f"Instance delete failed:", response.content)


    def dispatch(self, request):
        post_data = request.get_json()
        if not post_data:
            return jsonify({'error': 'No data provided'}), 400
        
        with self.lock:
            ready_instances = [inst for inst in self.instances if inst['is_ready']]
        if not ready_instances:
            return jsonify({'error': 'No ready instances available'}), 503
        
        self.round_robin_index = (self.round_robin_index + 1) % len(ready_instances)
        instance = ready_instances[self.round_robin_index]
        ip = instance['ip_address']
        port = instance['port']

        url = f'http://{ip}:{port}/predict'
        try:
            response = requests.post(url, json=post_data)
            if response.status_code == 200:
                return jsonify(response.json()), 200
            else:
                return jsonify({'error': 'Failed to process prediction', 'details': response.text}), response.status_code
        except requests.exceptions.RequestException as e:
            return jsonify({'error': 'Network error', 'details': str(e)}), 500


class Scaler(threading.Thread):
    def __init__(self, check_interval=1, scale_out_threshold=15, scale_in_threshold=15, history_length=20):
        super().__init__()
        self.check_interval = check_interval
        self.scale_out_threshold = scale_out_threshold
        self.scale_in_threshold = scale_in_threshold
        self.history_length = history_length
        self.request_history = {service_id: [] for service_id in services_status}  #        
        self.daemon = True

    def run(self):
        global services_status, services_dict
        while True:
            time.sleep(self.check_interval)
            
            for service_id, status in services_status.items():
                if "training" in service_id: # do not monitor the training jobs
                    continue 
                
                max_throughput = len(services_dict[service_id].instances) * services_dict[service_id].throughput
                scale_in_throughput = (len(services_dict[service_id].instances)-1) * services_dict[service_id].throughput
                current_time = datetime.now().strftime("%d/%b/%Y %H:%M:%S")     
                print(f"{service_id}, current_time: {current_time}, Requests: {status['requests']}, Max Throughput: {max_throughput}")

                if service_id not in self.request_history:
                    self.request_history[service_id] = []
                if len(self.request_history[service_id]) >= self.history_length:
                    self.request_history[service_id].pop(0)     
                self.request_history[service_id].append(status['requests'])

                status['requests'] = 0

                if self.should_scale_out(service_id, max_throughput):
                    services_dict[service_id].scale_out()
                    self.request_history[service_id].clear()

                if self.should_scale_in(service_id, scale_in_throughput):
                    services_dict[service_id].scale_in()
                    self.request_history[service_id].clear()

    def should_scale_out(self, service_id, max_throughput):
        over_threshold_times = sum(1 for x in self.request_history[service_id] if x > max_throughput)
        return over_threshold_times >= self.scale_out_threshold

    def should_scale_in(self, service_id, scale_in_throughput):
        under_threshold_times = sum(1 for x in self.request_history[service_id] if x < scale_in_throughput)
        return under_threshold_times >= self.scale_in_threshold and len(services_dict[service_id].instances) > 1
    

if __name__ == "__main__":
    service_manager = ServiceManager()
    scaler = Scaler(check_interval=1, scale_out_threshold=20, scale_in_threshold=30, history_length=40)
    scaler.start()
    app.run(debug=False, host='0.0.0.0', port=14999,  threaded=True)

    
