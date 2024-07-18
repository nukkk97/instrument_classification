import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import csv
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Params

sample_rate=22050
n_mels=128
n_fft=1024
hop_length=512
fixed_length=512
epochs=40
learning_rate=0.001
batch_size=512
model_path="./models/best_model.pth"



# Mel spectrogram transformation
class AudioTransform:
    def __init__(self, sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fixed_length=fixed_length):
        self.fixed_length = fixed_length
        self.hop_length = hop_length
        self.transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        #self.resample = T.Resample(orig_freq=sample_rate, new_freq=sample_rate)

    def __call__(self, waveform):
        #waveform = self.resample(waveform)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # we select the middle segment
        middle = waveform.size(1) // 2
        half_length = self.fixed_length * self.hop_length // 2
        start = max(0, middle - half_length)
        end = min(waveform.size(1), middle + half_length)
        waveform = waveform[:, start:end]

        mel_spec = self.transform(waveform)
        # Ensure [channel, freq, time]
        if mel_spec.size(2) < self.fixed_length:
            # Pad the tensor if shorter than the fixed length
            pad_size = self.fixed_length - mel_spec.size(2)
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_size), "constant", 0)
        elif mel_spec.size(2) > self.fixed_length:
            # Truncate the tensor if longer than the fixed length
            mel_spec = mel_spec[:, :, :self.fixed_length]
        return mel_spec

# Dataset
class AudioDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
        
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        label = self.labels[idx]
        waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
        if self.transform:
            waveform = self.transform(waveform)
        '''print(waveform)
        print(label)
        print(waveform.size())'''
        return waveform, label

# Load Data
def load_data(data_dir, mode):
    file_list = []
    labels = []
    csv_file = "Metadata_Train.csv" if mode == 'train' else "Metadata_Test.csv"
    
    with open(csv_file, mode='r') as file:
        csv_reader = csv.DictReader(file)
        metadata = {row['FileName']: row['Class'] for row in csv_reader}
    
    for filename in os.listdir(data_dir):
        audio_path = os.path.join(data_dir, filename)
        correspondence = metadata.get(filename, None)
        
        # Check and Encode
        if audio_path.endswith('.wav'):
            file_list.append(audio_path)
            encode = None
            if correspondence == 'Sound_Guitar':
                encode = 0
            elif correspondence == 'Sound_Drum':
                encode = 1
            elif correspondence == 'Sound_Violin':
                encode = 2
            else:
                encode = 3
            labels.append(encode)
    
    return file_list, labels

# Model
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        
        # Calculate the flattened size after convolutions and pooling
        self.flattened_size = self.calculate_flattened_size()
        
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 4)

    def calculate_flattened_size(self):
        # Dummy input to calculate the size of the flattened feature map
        dummy_input = torch.zeros(1, 1, n_mels, fixed_length)
        x = self.pool(self.relu(self.conv1(dummy_input)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return x.numel()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Paths
train_dir = './Train_submission/Train_submission/'
test_dir = './Test_submission/Test_submission/'

# Load data
train_file_list, train_labels = load_data(train_dir, "train")
test_file_list, test_labels = load_data(test_dir, "test")

audio_transform = AudioTransform(fixed_length=fixed_length)
train_dataset = AudioDataset(train_file_list, train_labels, transform=audio_transform)
test_dataset = AudioDataset(test_file_list, test_labels, transform=audio_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioClassifier().to(device)
directory = os.path.dirname(model_path)
if not os.path.exists(directory):
    os.makedirs(directory)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Model loaded!")
else:
    print("Model doesn't exist.")

# Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Start training
num_epochs = epochs
best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_diff = 0
    total = 0
    for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)):
        if inputs is None:
            continue
        # print(labels)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, test_predicted = torch.max(outputs, 1)
        
        differences = test_predicted != labels
        num_differences = differences.sum().item()
        total_diff += num_differences
        total += labels.numel()
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')
    print(f'Error Rate: {total_diff/total}')
    
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), model_path)

# Evaluating
model.eval()
predictions = []

with torch.no_grad():
    for batch, (inputs, labels) in enumerate(test_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        batch_filenames = [test_dataset.file_list[i + batch * batch_size] for i in range(len(inputs))]
        for filename, pred in zip(batch_filenames, predicted):
            filename = filename.split('/')[-1]
            decode = None
            if pred.item() == 0:
                decode = 'Sound_Guitar'
            elif pred.item() == 1:
                decode = 'Sound_Drum'
            elif pred.item() == 2:
                decode = 'Sound_Violin'
            else:
                decode = 'Sound_Piano'
            predictions.append([filename, decode])

# Save a CSV
predictions_df = pd.DataFrame(predictions, columns=['FileName', 'PredictedClass'])
predictions_df.to_csv('predictions.csv', index=False)

print('Predictions saved to predictions.csv')
