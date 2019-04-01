# An efficient end-to-end toolkit for ASR implemented with TensorFlow

##  How to use
1. Prepare Data
    Transfer your raw audio data into tfdata for fast load during training.
    ```python
    python ../../dataset.py -c configs/demo.yaml
    ```
    Supporting `.ark` file which is the standard type in Kaldi.

    Another thing this script will do is summarize the dataset and gives a proper bucket setting. This will iter your dataset. The bucket setting is related to your raw feature length and the frame skipping strategy (setting in your `.yaml`). So you need to reset you bucket boundary if you change them.

2. Configure your `.yaml` file
3. Train the model
    ```python
    python ../../main.py -c configs/demo.yaml
    ```
4. Infer with a trained model
    ```python
    python ../../main.py -m infer --gpu 0 -c configs/demo.yaml
    ```
# Model Implements
- Transformer
- CTC
- LAS
- Extended-RNA
- CTC-LM

# Results
## AIshell2
|Model| Discription | Cost | dev | test |
|:-----:|-------------|---|:-----:| :-----: |
| Transformer  |  6 blockes for each side; <br> skip 3 frames | 4 GPU/21h/55k steps| 8.8 | 8.5 |
|CTC   | CONV_BLSTM with 1600 hiddens states (800 for each direction); <br> confidence penalty 0.3 | 4 GPU/23h/33k steps |9.7  | |

## HKUST
|Model| Discription | Cost | train_dev | dev |
|:-----:|-------------|---|:-----:| :-----: |
| Transformer  |  6 blockes for each side; <br> skip 3 frames | 4 GPU/13h/35k steps| 32.6 | |
|CTC   | CONV_BLSTM with 1600 hiddens states (800 for each direction); <br> confidence penalty 0.3 | 4 GPU/14.5h/21.5k steps | 26.9  | 27.8 |

## LibriSpeech
|Model| Discription | Cost | dev-clean | test-clean |
|:-----:|-------------|---|:-----:| :-----: |
|CTC   | CONV_BLSTM with 1600 hiddens states (800 for each direction); <br> confidence penalty 0.3 | 4 GPU/1d2h/28k steps |  | 6.9 |


# Projects

the projects has
- arguments.py
- datasets.py
- /models
- /exps

each data has a folder in the exp folder, which contains:
- /models
- /configs
- vocab.txt
if you run an experiment, you just need to enter the corresponding folder and run:
```python
python ../../main.py configs/standard.yaml
```
If you need to check the learing curve, you can run this command in the folder:
```python
tensorboard --logdir models --port 8888
```

Please look Projects for more details.

# Coding Standard
all the model files and utility files are here. We need to additionally prepare the project folder, where the batch loop and log pytho commands exist.

# Acknowledgements
https://github.com/chqiwang/transformer \\
https://github.com/vrenkens/nabu
...
