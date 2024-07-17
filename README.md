# Instrument Classification
This is a simple self-practice of utilizing torchaudio to perform instrument sound classification task. The core concepts are Mel spectrogram and convolutional neural network.
##Ｄataset
The related dataset is here: 
https://www.kaggle.com/datasets/soumendraprasad/musical-instruments-sound-dataset
This dataset consists of four classes: violin, guitar, piano, and drums. The relevant labels of filenames are stored in Metadata_Train.csv and Metadata_Test.csv. However, some of the data from the author is totally mislabeled, so I tried my best to fix them. The fixed files are attached within this repository.
##Requirements
Please refer to requirements.txt! Also, You should use venv as a virtual environment.
```bash
python -m venv /path/to/new/virtual/environment
```
Activate the environment:
```bash
source env_name/bin/activate
```
Install dependencies:
```bash
pip install -r /path/to/requirements.txt
```
Run the code:
```bash
python main.py
```
##Model Architecture
You can find the following session in main.py. The layers are mainly consist of cnn and fcs.
```python
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        # other layers

    def forward(self, x):
        # forwarding
        return x
```
##Training
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
	
The parameters are adjustable in main.py.
##Testing Accuracy
The script is able to check the total accuracy on the test set after you run main.py.
```bash
python correctness.py
```