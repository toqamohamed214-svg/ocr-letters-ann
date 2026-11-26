# Letter recognition (A-Z) with an Artificial Neural Network

Trains a small ANN (16 inputs -> 128 -> 64 -> 26 letters) to recognise capital English letters.

**Data:** UCI Letter Recognition (20,000 samples, 16 numeric features per letter image), included in `data/`.
Source: https://archive.ics.uci.edu/dataset/59/letter+recognition
Note: the features are pre-extracted from images of letters, so this is the classic feature-based version of the task. To work on raw handwritten images, swap in EMNIST Letters and feed pixels to the same kind of network.

**Run**
```
pip install -r requirements.txt
python main.py
```
Output: test accuracy, per-letter report, sample predictions (saved example in `run_output.txt`).
