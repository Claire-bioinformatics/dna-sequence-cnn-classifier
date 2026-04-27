import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

import matplotlib.pyplot as plt

import os

os.makedirs("outputs", exist_ok=True)

def generate_random_dna_sequence(length=100):
    return ''.join(random.choices(['A', 'C', 'G', 'T'], k=length))


def insert_motif(sequence, motif="TATAAA"):
    position = random.randint(0, len(sequence) - len(motif))
    return sequence[:position] + motif + sequence[position + len(motif):]


def generate_dataset(n_samples=2000, sequence_length=100, motif="TATAAA"):
    sequences = []
    labels = []

    for _ in range(n_samples):
        seq = generate_random_dna_sequence(sequence_length)

        if random.random() < 0.5:
            seq = insert_motif(seq, motif)
            label = 1
        else:
            label = 0

        sequences.append(seq)
        labels.append(label)

    return pd.DataFrame({
        "sequence": sequences,
        "label": labels
    })


df = generate_dataset()
df.head()

def one_hot_encode_sequence(sequence):
    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1]
    }

    return np.array([mapping[base] for base in sequence])


X = np.array([one_hot_encode_sequence(seq) for seq in df["sequence"]])
y = df["label"].values

X.shape, y.shape

X = np.transpose(X, (0, 2, 1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class DNASequenceCNN(nn.Module):
    def __init__(self):
        super(DNASequenceCNN, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=4,
            out_channels=16,
            kernel_size=6
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.fc1 = nn.Linear(16 * 47, 32)
        self.fc2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


model = DNASequenceCNN()
model

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20

train_losses = []
train_accuracies = []

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    predictions = (outputs >= 0.5).float()
    accuracy = (predictions == y_train).float().mean().item()

    train_losses.append(loss.item())
    train_accuracies.append(accuracy)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

model.eval()

with torch.no_grad():
    y_prob = model(X_test)
    y_pred = (y_prob >= 0.5).float()

test_accuracy = accuracy_score(y_test.numpy(), y_pred.numpy())
test_auc = roc_auc_score(y_test.numpy(), y_prob.numpy())

print("Test accuracy:", test_accuracy)
print("ROC-AUC:", test_auc)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test.numpy(), y_pred.numpy())

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_test.numpy(), y_pred.numpy()))

for i in range(5):
    print("True:", int(y_test[i].item()),
          "Predicted:", int(y_pred[i].item()),
          "Prob:", float(y_prob[i].item()))

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test.numpy(), y_prob.numpy())

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {test_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("outputs/roc_curve.png", dpi=300)
plt.show()

plt.figure()
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.savefig("outputs/training_loss_curve.png", dpi=300)
plt.show()

plt.figure()
plt.plot(train_accuracies)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy Curve")
plt.savefig("outputs/training_accuracy_curve.png", dpi=300)
plt.show()
