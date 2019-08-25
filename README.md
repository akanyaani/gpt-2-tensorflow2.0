# GPT-2 Pre-training and text generation, implemented in Tensorflow2.0


**This repositry has Openai GPT-2 pre-training implementation in tensorflow2.0, I am also working on text -generation using this model, 
I will push that code after couple of days.**

**Requirements**
*  python >= 3.6
*  tensorflow==2.0
*  sentencepiece
*  ftfy
*  click
*  tqdm

You can pre-train the model using sample data availbe in repository or you can download the data using this github repo https://github.com/eukaryote31/openwebtext

Pre-Training model on sample data avialable in repositry
```
>> python pre_process.py
```

Pre-Training model on openwebtext or any other data

```
>> python pre_process.py --data-dir data_directory
```

```
>> python train_model.py
```
