# OnML (Ontology-based Interpretable Machine Learning for Textual Data)

This is the official implementation of the [Ontology-based Interpretable Machine Learning for Textual Data](https://arxiv.org/pdf/2004.00204.pdf) in Tensorflow.

If you use this code or our results in your research, please cite as appropriate:

```
@article{lai2020ontology,
  title={Ontology-based Interpretable Machine Learning for Textual Data},
  author={Lai, Phung and Phan, NhatHai and Hu, Han and Badeti, Anuja and Newman, David and Dou, Dejing},
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
+ Open ontology locally by [Protégé](https://protege.stanford.edu/products.php).
+ Open ontology online by importing the ontology on [WebVOWL](http://vowl.visualdataweb.org/webvowl.html).

To reproduce the results:
+ Clone or download the folder from this repository.
+ Some large-size data or pretrained models of any file needed to run, if you cannot find it here, please find it on [Google Drive folder](https://drive.google.com/drive/folders/17w6RLR5pTG8BfXN-039YWBMnJWrYGKmK?usp=sharing). 
+ Go to folder `src/` and Run `python3 main_cc.py`
+ Note: Due to the privacy requirements of Drug data, this repository only provides data and code for consumer complaints. 

To play with the model, you can:
+ Change classifier: In this code, the pretrained models are provided in folder `model/` in `h5` and `json` format. You can train your own classifier, save it in the same format, and then load it in the main function.
+ Customize data, ontology concepts, relations among ontology concepts, anchor list, and stopword list in `data/`.

To customize the code with your data or train the prediction model:
+ Go to `reproduce` folder. Instructions are provided there. 

## Issues
If you have any issues while running the code or further information, please send email directly to the first author of this paper (`tl353@njit.edu`). 
