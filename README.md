# OnML (Ontology-based Interpretable Machine Learning)

This repository contains the code used for the paper ...

The model comes with instructions to train:
+ Prediction model (LSTM/SVM)
+ OnML Interpretable model 

If you use this code or our results in your research, please cite as appropriate:

```
@article{abc,
  title={{Ontology-based Interpretable Machine Learningfor Textual Data}},
  author={Phung, Lai and NhatHai, Phan and Han, Hu and Anuja, Badeti and David, Newman and Dejing Dou},
  journal={International Joint Conference on Neural Networks},
  year={2020}
}
```

## Software Requirements

Python 3 is used for the current codebase.

Tensorflow 1.1 or later

## Experiments
To reproduce the results in the paper:
+ `python3 main.py`

To train the model from scratch:
+ `python3 lstm_cc.py`
+ Call the trained model in the `lstm_cc.py`
+ `python3 main.py`
