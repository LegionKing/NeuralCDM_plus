# NeuralCDM_plus
The implementation of NeuralCDM+ in ["Neural Cognitive Diagnosis for Intelligent Education Systems"](http://staff.ustc.edu.cn/~qiliuql/files/Publications/Fei-Wang-AAAI2020.pdf). This is also the CNCD-Q model in "[NeuralCD: A General Framework for Cognitive Diagnosis](https://ieeexplore.ieee.org/abstract/document/9865139)".



If this code helps with your studies, please kindly cite the following publication:

```
@article{wang2020neural,
  title={Neural Cognitive Diagnosis for Intelligent Education Systems},
  author={Wang, Fei and Liu, Qi and Chen, Enhong and Huang, Zhenya and Chen, Yuying and Yin, Yu and Huang, Zai and Wang, Shijin},
  booktitle={Thirty-Fourth AAAI Conference on Artificial Intelligence},
  year={2020}
}
```

or

```
@article{wang2022neuralcd,
  title={NeuralCD: A General Framework for Cognitive Diagnosis},
  author={Wang, Fei and Liu, Qi and Chen, Enhong and Huang, Zhenya and Yin, Yu and Wang, Shijin and Su, Yu},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2022},
  publisher={IEEE}
}
```





## Files that need to be prepared in advance

- result/word2vec.model
- result/word2vec.model.wv.vectors.npy
- result/word2vec.model.trainables.syn1neg.npy

These are the saved model files trained with gensim. They will be used for further preprocessing in the function prepare_embedding()  (in net_knowledge.py).



- data/net_knowledge_train.json

The data file used to train the knowledge concept prediction model. An example data file is provided to illustrate the format.



- data/net_knowledge_pred.json

The data file that contains the exercies of which the knowledge concepts need to be predicted. Exercise texts are contained in the file. An example data file is provided to illustrate the format. 



- data/train_set.json

The training set for NeuralCDM+. Response logs of different students are combined and shuffled. An example data file is provided.



- data/val_set.json

The validation set for NeuralCDM+. An example data file is provided.



- data/test_set.json

The test set for NeuralCDM+. The data format is the same with val_set.json.



## Dependencies:

- python 3.6
- pytorch >= 1.0 (pytorch 0.4 might be OK but pytorch<0.4 is not applicable)
- numpy
- json
- sklearn
- bintrees



## Usage

Use `python net_knowledge.py` to train the knowledge concept prediction model and prepare the files needed for NeuralCDM+.



Then, run `python main.py` to train and test NeuralCDM+. 

