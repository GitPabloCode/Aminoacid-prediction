import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm  
from utils import *
from models import LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if os.path.exists("training_batches.pt"):
    batches = torch.load("training_batches.pt")
else:
    batches = create_batches("training_sequences.txt", 64)
    torch.save(batches, "training_batches.pt")

print("Batches created and start training")

vocab_size = 26
embedding_size = 10
hidden_size = 64
model = LSTM(vocab_size, embedding_size, hidden_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []
n_epochs = 4
for epoch in range(n_epochs):
    total_loss = 0
    with tqdm(batches, desc=f"Epoch {epoch}") as tqdm_iterator:
        for sequences in tqdm_iterator:
            sequences = sequences.to(device)
            optimizer.zero_grad()
            output, _ = model(sequences)
            output = output.permute(0, 2, 1)
            loss = criterion(output[:, :, :-1], sequences[:, 1:])   # Confrontiamo l'output del modello con il successivo amminoacido nella sequenza
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            tqdm_iterator.set_postfix({"loss": loss.item()})

    average_loss = total_loss / len(batches)
    losses.append(average_loss)
    print(f'Average Loss: {average_loss}')

plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.savefig('training_loss_plot.png')
path = "LSTM_weights.pth"
torch.save(model.state_dict(), path)
