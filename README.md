# StanceRel

## Requirements

* Python 3.8.10
* PyTorch 1.21.1
* transformers 4.10.0

## Inference

You can directly test the model through 
```
python run_roberta_rel.py
```

Then you can obtain the results in the test_reuslts.txt such as 

results {'disagree-f1': 0.7458323215682784, 'neutral-f1': 0.557027589856837, 'agree-f1': 0.7170068128058874, 'macro-f1': 0.6732889080770009, 'micro-f1': 0.6875956752519801, 'precision': 0.6876456876456877}


## Training

You need 
```
python preprocess_graph.py
python train_and_extract_graph_features.py
python run_roberta_rel.py (adding --do_train to the args)
```

Trained model can be downloaded from :



Dataset can be downloaded from : https://scale.com/open-av-datasets/oxford

