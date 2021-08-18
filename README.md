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

Python 3.5 is used for the current codebase.

Tensorflow 1.1 or later (Tensor 1 only)

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

## Common issues reported
I received several emails about problems while running the code, but in general, I found that is just the version compatibility issues. The common errors and how to solve it are as follows:

+ `module has no attribute "placeholder"`: This is because you are using Tensorflow 2, while the code is using Tensorflow 1. Please [disable TF2](https://www.tensorflow.org/api_docs/python/tf/compat/v1/disable_v2_behavior) behaviors by adding the line `tf.compat.v1.disable_v2_behavior()` after the line of importing TF in all the files. 

+ `object of type "NoneType" has no len`: I guess you are using Python 3.7. Note that this code uses part of [LIME](https://github.com/marcotcr/lime) for visualization, and that visualization does not work with Python 3.7  ([error](https://github.com/marcotcr/lime/issues/294)). So I recommend you to use Python3 < 3.7 to be able to run the whole code; otherwise, you can run OnML but no visualization.

To avoid install/uninstall the package, I recommend you to create an environment while using the code, so that it will not affect other code's running.
1. Create an environment by: `conda create -n name-of-environment python=version-of-python`, e.g., conda create -n my_py37 python=3.7
2. Activate the created environment by: `conda activate name-of-environment`, e.g., conda activate my_py37
3. Install needed packages to run the code (If we dont know which packages to install, just run the code, the error about needed packages will show up)
4. Deactivate the environment after using: `conda deactivate`

## Potential issues 
If you have any issues while running the code or further information, please send email directly to the first author of this paper (`tl353@njit.edu`). 
