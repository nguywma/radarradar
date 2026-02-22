import re
import matplotlib.pyplot as plt

# Parse log file
log_file = "log.txt"
with open(log_file, "r") as f:
    log_lines = f.readlines()

mean_recalls = []
val_losses = []
avg_train_losses = []

for line in log_lines:
    # Match Mean Recall
    match_recall = re.search(r"===> Mean Recall on OORD\s*:\s*([\d.]+)", line)
    if match_recall:
        mean_recalls.append(float(match_recall.group(1)))
    
    # Match Validation Loss
    match_val_loss = re.search(r"===> Mean Validation Loss:\s*([\d.]+)", line)
    if match_val_loss:
        val_losses.append(float(match_val_loss.group(1)))

    # Match Average Training Loss
    match_avg_train_loss = re.search(r"===> Epoch \d+ Complete: Avg. Loss:\s*([\d.]+)", line)
    if match_avg_train_loss:
        avg_train_losses.append(float(match_avg_train_loss.group(1)))

# Plotting
epochs = range(1, len(avg_train_losses) + 1)

plt.figure(figsize=(12, 6))
plt.plot(epochs, avg_train_losses, label='Avg Training Loss')
plt.plot(epochs, val_losses, label='Mean Validation Loss')
# plt.plot(epochs, mean_recalls, label='Mean Recall on OORD')
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Metrics over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

all_sequences = [
    'Bellmouth_1', 'Bellmouth_2', 'Bellmouth_3', 'Bellmouth_4',
    'Hydro_1', 'Hydro_2', 'Hydro_3',
    'Maree_1', 'Maree_2',
    'Twolochs_1', 'Twolochs_2'
]