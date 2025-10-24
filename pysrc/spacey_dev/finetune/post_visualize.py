# requires the state log_history from trainer
from matplotlib import pyplot as plt
import json
import numpy as np

with open(f"./finetuned_models/roberta-base-squad2-nq-best/log_history.json", "r") as f:
# with open(f"./finetuned_models/roberta-base-squad2-nq-nasa/log_history.json", "r") as f:

    log_history = json.load(f)

train_steps, train_losses = [], []
eval_steps, eval_losses   = [], []
train_epochs, val_epochs = [], []

steps = []
f1_values, em_values, eval_steps = [], [], []

for i, log in enumerate(log_history):
    if "loss" in log:
        train_losses.append(log["loss"])
        train_steps.append(log["step"])
        train_epochs.append(log['epoch'])
    
    if "eval_f1" in log:
        f1_values.append(log["eval_f1"])
        em_values.append(log["eval_exact"])
        eval_steps.append(log.get("step", log.get("epoch", len(f1_values))))
        eval_losses.append(log["eval_loss"])
        val_epochs.append(log["epoch"])

interp_val = np.interp(train_steps, eval_steps, eval_losses) # interpol

plt.figure()
#plt.title(f'RoBERTa Fine Tune Loss Curves')
plt.plot(train_steps, train_losses)
plt.plot(train_steps, interp_val, 'o-')
plt.grid(alpha=0.3)
plt.xlabel('steps')
plt.ylabel("Loss")
plt.legend(labels=['Training Loss', "Validation Loss"])

interp_epoch_val = np.interp(train_epochs, val_epochs, eval_losses) # interpol

plt.figure()
#plt.title(f'RoBERTa Fine Tune Loss Curves')
plt.plot(train_epochs, train_losses)
plt.plot(train_epochs, interp_epoch_val, 'o-')
plt.grid(alpha=0.3)
plt.xlabel('epoch')
plt.ylabel("Loss")
plt.legend(labels=['Training Loss', "Validation Loss"])


plt.figure()
#plt.title("Validation Performance")
plt.plot(val_epochs, f1_values, label="F1 (val)")
plt.plot(val_epochs, em_values, label="EM (val)")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.show()