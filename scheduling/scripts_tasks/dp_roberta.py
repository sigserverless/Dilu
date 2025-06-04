# torch.DDP 
# NCCL_SOCKET_IFNAME=eno1 python -u /cluster/workloads/dp_roberta.py --nodes 1 --gpus 4 --nr 0  --iteration 100
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import Dataset
import torch.multiprocessing as mp
import time
from datetime import datetime
import statistics

class RandomTextDataset(Dataset):
    def __init__(self, tokenizer, num_samples=1000, max_length=256):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        random_sentence = "This is a sentence {}. ".format(idx) * 60
        inputs = self.tokenizer.encode_plus(random_sentence, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        label = self.labels[idx]
        return inputs, label


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, model_name, args):
    global_rank = args.nr * args.gpus + rank
    setup(global_rank, world_size)

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    model.to(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = AdamW(model.parameters(), lr=5e-5)

    batch_size = args.batch_size
    train_dataset = RandomTextDataset(tokenizer=tokenizer, num_samples=batch_size*args.iterations*4, max_length=128)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        collate_fn=lambda x: ({'input_ids': torch.stack([y[0]['input_ids'].squeeze() for y in x]),
                               'attention_mask': torch.stack([y[0]['attention_mask'].squeeze() for y in x]),
                               'labels': torch.stack([y[1] for y in x])})
    )

    model.train()
    all_st = time.time()
    iteration = 0
    train_total_tokens = 0
    total_tokens = 0  
    start_time = time.time()  
    end_time = time.time()
    statistics_throughput = []
    for batch in train_dataloader:

        batch = {k: v.to(rank) if isinstance(v, torch.Tensor) else torch.Tensor(v).to(rank) for k, v in batch.items()}
        total_tokens += batch['input_ids'].nelement() 
        train_total_tokens += batch['input_ids'].nelement()

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        iteration += 1
        if iteration % 5 == 0:
            end_time = time.time() 
            elapsed_time = end_time - start_time   
            throughput = total_tokens / elapsed_time 
            print(f"Rank {rank}, Iteration {iteration}, Throughput: {throughput:.4f} tokens/second")
            if iteration!=5:
                statistics_throughput.append(throughput)
            total_tokens = 0   
            start_time = time.time()

        # iteration_time = end_time - start_time
        # print(f"Rank {rank}, Iteration {iteration}, Batch size: {batch_size}, Loss: {loss.item()}, Iteration Time: {iteration_time:.4f} seconds")

    all_et = time.time()
    total_eplased_time = all_et - all_st
    print("ALL TIME: =======", total_eplased_time)
    print(f"Rank {rank}, Mean Throughput: {statistics.mean(statistics_throughput):.4f} tokens/second, stdev: {statistics.stdev(statistics_throughput):.4f} tokens/second," )
    cleanup()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=4, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--iterations', default=10, type=int, 
                        metavar='N',
                        help='number of iteration to run of each worker')
    parser.add_argument('--batch_size', default=128, type=int, 
                        metavar='N',
                        help='number of batch_size to run of each worker')
    args = parser.parse_args()
    model_name = '/cluster/datasets/roberta-large'
    args.world_size = args.gpus * args.nodes      
    mp.spawn(train, nprocs=args.gpus, args=(args.world_size, model_name, args, ), join=True)         

