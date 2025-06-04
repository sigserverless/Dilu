import time
import statistics
import random
import argparse
import torch
import numpy as np
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

parser = argparse.ArgumentParser(description="Flask server for running a transformers model")
parser.add_argument("--model_name_or_path", required=True, help="Path to pretrained model or model identifier from huggingface.co/models")
parser.add_argument("--device", default=0, type=int, help="Device to run the model on")
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
parser.add_argument('--sm', type=int, default=100, help='Batch size for inference')
args = parser.parse_args()

config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({'eos_token': '[EOS]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id=50256
args.device = "cuda:"+str(args.device)
model.to(args.device)
model.eval()

def generate_random_sentence(vocab, length=256):
    return ' '.join(random.choices(vocab, k=length))

def generate_batch(batch_size, length=256):
    
    vocab = ["the", "and", "of", "to", "in", "is", "you", "that", "it", "he", 
             "for", "was", "on", "are", "as", "with", "his", "they", "I", 
             "at", "be", "this", "have", "from", "or", "one", "had", "by", 
             "word", "but", "not", "what", "all", "were", "we", "when", "your", 
             "can", "said", "there", "use", "an", "each", "which", "she", "do", 
             "how", "their", "if", "will", "up", "other", "about", "out", "many", 
             "then", "them", "these", "so", "some", "her", "would", "make", 
             "like", "him", "into", "time", "has", "look", "two", "more", "write", 
             "go", "see", "number", "no", "way", "could", "people", "my", "than", 
             "first", "water", "been", "call", "who", "oil", "its", "now", "find", 
             "long", "down", "day", "did", "get", "come", "made", "may", "part"]

    return [generate_random_sentence(vocab, length) for _ in range(batch_size)]



iters = 100
texts = generate_batch(args.batch_size, 128)
iter_times = []
start_time = time.time()
for _ in range(iters):
    encoded_data = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=128,
        truncation=True,
        return_tensors='pt',
        padding='longest'  
    )
    input_ids = encoded_data['input_ids'].to(args.device)
    with torch.no_grad():
        outputs = model(input_ids)
        predicted_labels = torch.argmax(outputs.logits, dim=1).cpu().numpy().tolist()
end_time = time.time()
eplased_time = end_time-start_time
# bs, sm, mean_time, throughput
print('%f,%f,%f,%f' % (args.batch_size, args.sm, eplased_time/iters, iters*args.batch_size/eplased_time))