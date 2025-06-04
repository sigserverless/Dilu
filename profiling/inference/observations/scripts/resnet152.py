import time
import argparse
import torch
import cv_models as model_zoo

parser = argparse.ArgumentParser(description="Flask server for running a transformers model")
parser.add_argument("--device", default=0, type=int, help="Device to run the model on")
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
parser.add_argument('--sm', type=int, default=100, help='Batch size for inference')
args = parser.parse_args()

model = model_zoo.resnet152(pretrained=False)
model.eval()
model.to(args.device)
image_tensors = torch.rand([args.batch_size, 3, 224, 224]).to(args.device)
iters = 100
start_time = time.time()
with torch.no_grad():
    for _ in range(iters):
        output = model(image_tensors)
        predictions = output.cpu().numpy().tolist()
end_time = time.time()
eplased_time = end_time-start_time
print('%f,%f,%f,%f' % (args.batch_size, args.sm, eplased_time/iters, iters*args.batch_size/eplased_time))



