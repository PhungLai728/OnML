# OnML (Ontology-based Interpretable Machine Learning)

This is the official implementation of the [Ontology-based Interpretable Machine Learningfor Textual Data](https://arxiv.org/pdf/2004.00204.pdf) in Tensorflow.

If you use this code or our results in your research, please cite as appropriate:

```
@article{abc,
  title={{Ontology-based Interpretable Machine Learning for Textual Data}},
  author={Phung, Lai and NhatHai, Phan and Han, Hu and Anuja, Badeti and David, Newman and Dejing Dou},
  journal={International Joint Conference on Neural Networks},
  year={2020}
}
```


## Software Requirements

Python 3 is used for the current codebase.

Tensorflow 1.1 or later

Protégé for Ontology


## Experiments
The repository comes with instructions to reproduce the results in the paper or to train the model from scratch:

To view Ontology:
+ ConSo and DrugAO ontologies used in the paper are provided in folder `ontologies`.
+ Open ontology by [Protégé](https://protege.stanford.edu/products.php) or Import it to [WebVOWL](http://vowl.visualdataweb.org/webvowl.html).

To reproduce the results:
+ Clone or download the folder from this repository.
+ Some large-size data or pretrained models are provided in [Google Drive folder](https://drive.google.com/drive/my-drive).
+ Run `python3 main_cc.py`
+ Note: Due to the privacy requirements of Drug data, this repository only provide data and code for consumer complaints. 

To play with the model, you can:
+ Change classifier: In this code, the pretrained models are provided in folder `model/` in `h5`, `json`, and `pickle` format. You can train your own classifier and load it in the main function.
+ Customize data, the ontology concepts, relations among ontology concepts, your anchor list, stopword lists in `data/`.


## Issues
If you have any issues while running the code or further information, please send email directly to the first author of this paper. 
