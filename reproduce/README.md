# OnML (Ontology-based Interpretable Machine Learning for Textual Data)

This is the official implementation of the [Ontology-based Interpretable Machine Learning for Textual Data](https://arxiv.org/pdf/2004.00204.pdf) in Tensorflow.

If you use this code or our results in your research, please cite as appropriate:

```
@article{phung2020ontology,
  title={Ontology-based Interpretable Machine Learning for Textual Data},
  author={Phung, Lai and NhatHai, Phan and Han, Hu and Anuja, Badeti and David, Newman and Dejing Dou},
  journal={International Joint Conference on Neural Networks},
  year={2020}
}
```

## Customize OnML with your data, model, and ontology

Folder `AMT_evaluation`:
+ This folder is to help you create data for uploading Amazon Mechanical Turk,
+ Data (images) is in `screen_shots`,
+ Run `python3 gen_CSV.py` to generate script for AMT.

Folder `OLLIE`:
+ Generate triples using OLLIE. Some examples are provided,
+ Run `python3 create_txt.py` to create input files for OLLIE,
+ Run `bash run.sh` to get output of OLLIE. Do not forget to download `ollie-app-latest.jar` from [Google Drive folder](https://drive.google.com/drive/folders/17w6RLR5pTG8BfXN-039YWBMnJWrYGKmK?usp=sharing).

Folder `Preprocessing_data`:
+ Run `python3 preprocess.py` to do data processing,
+ Run `python3 onto.py` to get information from ontology. Note that `ConSo_onto.csv` was generated directly from an ontology in Protégé (Protégé 5.5.0-beta-9 was used in this project) and an add-on `Export to CSV`,
+ Run `python3 gen_vocab_matrix.py` to generate w2v matrix of dictionary,
+ Run `python3 get_X.py` to get data, which is used in `main_cc.py` and `lstm_cc.py`.

Folder `Prediction_model`:
+ Run `python3 lstm_cc.py` for model prediction (The LSTM model is used as an original prediction model in OnML).

Note:
+ While running OnML model, you see some files, such as `concepts_property_cc_071119.csv`, `abstract_concepts_CC_071119`, etc. They are manually generated based on the relations of concepts on the ontology. If you are interested in how to generate them, please contact the first author of this paper!

