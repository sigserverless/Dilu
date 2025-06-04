import matplotlib.pyplot as plt

file_paths = ['./logs/k8s.txt', './logs/infless-l.txt', './logs/infless-r.txt', './logs/dilu.txt']
baselines =['Exclusive', 'INFless-l', 'INFless-r', 'Dilu']
plt.figure(figsize=(10, 5))
colors = ['r', 'y', 'b','g'] 
line_styles = [':', '--','-.','-'] 

i = 0
for file_path, color, line_style in zip(file_paths, colors, line_styles):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    timestamps = []
    gpu_counts = []
    for line in lines:
        time_str, gpu_count_str = line.split()
        timestamps.append(float(time_str) * 60)
        gpu_counts.append(int(gpu_count_str))
    
    # baseline
    plt.plot(timestamps, gpu_counts, linestyle=line_style, color=color, label=baselines[i])
    i+=1
plt.legend()
plt.xlabel('Time (minutes)')
plt.ylabel('Active GPUs')
plt.title('Active GPU Count Over Time Across Baselines')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('multiple-baselines-timeline.png', dpi=300)
