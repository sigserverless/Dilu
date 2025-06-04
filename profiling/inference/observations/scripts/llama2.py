import time
import statistics
import random
import argparse
import torch
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

parser = argparse.ArgumentParser(description="Flask server for running a transformers model")
parser.add_argument("--model_name_or_path", required=True, help="Path to pretrained model or model identifier from huggingface.co/models")
parser.add_argument("--device", default=0, type=int, help="Device to run the model on")
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
parser.add_argument('--sm', type=int, default=100, help='Batch size for inference')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

logging.set_verbosity_error()

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(args.device)
model.eval()


sentences = [
    "The quick brown fox jumps over the lazy dog, and then it runs into the forest, where it meets another fox. They become friends and decide to explore together. They find a hidden cave filled with ancient treasures and mysterious artifacts. The foxes are excited and decide to take some of the treasures back to their den.",
    "In a small village, there lived a kind old man who loved to tell stories to the children. Every evening, the children would gather around him to listen to his tales of adventure and magic. The old man had a magical book that could transport them to different worlds, where they experienced incredible adventures.",
    "Once upon a time, there was a brave knight who set out on a quest to rescue a princess from a dragon. The knight journeyed through dark forests and climbed high mountains. After many challenges, he finally reached the dragon's lair. With courage and skill, he defeated the dragon and rescued the princess, bringing her back to the kingdom safely.",
    "In the bustling city, there was a young girl who dreamed of becoming a famous dancer. She practiced every day, honing her skills and perfecting her moves. One day, she auditioned for a prestigious dance academy and impressed the judges with her talent. She was accepted and began her journey to stardom.",
    "A group of friends decided to go on a camping trip in the mountains. They packed their gear and set off on an adventure. They hiked through beautiful landscapes, set up camp near a crystal-clear lake, and spent the night under a starry sky. The trip was filled with laughter, stories, and unforgettable memories.",
    "There was a young inventor who created amazing gadgets in his small workshop. One day, he built a robot that could help with household chores. The robot became very popular, and soon everyone wanted one. The inventor's life changed overnight, and he became known as the genius who revolutionized daily life.",
    "In a distant kingdom, there was a magical garden that bloomed with the most beautiful flowers. The garden was tended by a gentle fairy who had the power to make flowers bloom all year round. People from all over the kingdom came to see the garden and were mesmerized by its beauty and fragrance.",
    "A young boy discovered a mysterious map in his attic. The map led to a hidden treasure buried deep in the forest. With excitement and curiosity, he set out on an adventure to find the treasure. Along the way, he encountered various challenges but never gave up. Eventually, he found the treasure and returned home as a hero.",
    "In a futuristic city, there was a scientist working on a groundbreaking project. She developed a machine that could transport people to different dimensions. After years of hard work, she successfully created a portal. The scientist became the first person to travel to another dimension, discovering new worlds and possibilities.",
    "A talented artist lived in a quaint town, where she painted beautiful landscapes and portraits. Her art was admired by everyone, and her paintings were displayed in galleries across the country. One day, she painted a masterpiece that captured the essence of her town. The painting became famous, and she was celebrated as one of the greatest artists of her time.",
    "In an ancient forest, there was a wise owl who knew the secrets of the woods. Animals from all over came to seek the owl's wisdom and guidance. The owl shared its knowledge generously, helping the animals solve their problems and live in harmony. The forest flourished, and the owl was respected and loved by all.",
    "A young girl found a magical pendant in her grandmother's attic. When she wore it, she could communicate with animals. She made many animal friends and learned about their lives and challenges. Together, they went on adventures and helped each other. The pendant brought joy and understanding to her life.",
    "In a coastal town, there was a lighthouse keeper who ensured the safety of passing ships. One stormy night, the lighthouse's light went out. The keeper braved the storm to fix the light, saving many ships from disaster. His bravery and dedication were celebrated by the townspeople, who honored him as a hero.",
    "A curious cat lived in a cozy cottage at the edge of the village. The cat loved to explore and often went on little adventures. One day, it discovered a hidden path that led to a secret garden. The garden was filled with magical plants and creatures. The cat made it its special place and visited it often.",
    "In a faraway desert, there was an oasis where travelers could rest and refresh themselves. The oasis was guarded by a kind spirit who ensured that the water remained pure and plentiful. Travelers who visited the oasis were grateful for the spirit's care and often left offerings in thanks. The oasis became a legendary place of respite.",
    "A brilliant musician lived in a bustling city, where she performed in various concerts and events. Her music touched the hearts of many, and she became well-known for her talent. One day, she composed a symphony that captured the spirit of the city. The symphony was performed by the city's orchestra and became an instant classic, loved by all."
]



iters = 50
texts = sentences[:args.batch_size]
start_time = time.time()
for _ in range(iters):
    with torch.no_grad():
        input_ids = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(args.device)
        generated_text = model.generate(
            input_ids,
            max_new_tokens=32,
            num_return_sequences=1,  
            pad_token_id=tokenizer.eos_token_id,
            max_length=64,
            use_cache=True,
            do_sample=False
        )
    decoded_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_text]
end_time = time.time()
elapsed_time = end_time - start_time

# bs, sm, mean_time, throughput
print('%f,%f,%f,%f' % (args.batch_size, args.sm, elapsed_time/32/iters, iters*args.batch_size/elapsed_time))
