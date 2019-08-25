# GPT-2 Pre-training and text generation, implemented in Tensorflow2.0


**This repository has OpenAi GPT-2 pre-training implementation in tensorflow2.0, I am also working on text -generation using this model, 
I will push that code after couple of days.**

**Requirements**
*  python >= 3.6
*  tensorflow==2.0
*  sentencepiece
*  ftfy
*  click
*  tqdm

You can pre-train the model using sample data availbe in repository or you can download the data using this github repo https://github.com/eukaryote31/openwebtext

Pre-Training model on sample data available in repository
```
$ python pre_process.py --help

Options:
  --data-dir TEXT        training data path  [default: /data/scraped]
  --vocab-size INTEGER   byte pair vocab size  [default: 32000]
  --min-seq-len INTEGER  minimum sequence length  [default: 15]
  --max-seq-len INTEGER  minimum sequence length  [default: 512]
  --help                 Show this message and exit.
  
  
>> python pre_process.py
```

Pre-Training model on openwebtext or any other data

```
>> python pre_process.py --data-dir=data_directory
```

```
$ python train_model.py --help

Options:
  --num-layers INTEGER      No. of decoder layers  [default: 8]
  --embedding-size INTEGER  Embedding size  [default: 768]
  --num-heads INTEGER       Number of heads  [default: 8]
  --dff INTEGER             Filter Size  [default: 3072]
  --max-seq-len INTEGER     Seq length  [default: 515]
  --vocab-size INTEGER      Vocab size  [default: 50000]
  --optimizer TEXT          optimizer type  [default: adam]
  --batch-size INTEGER      optimizer type  [default: 8]
  --learning-rate FLOAT     learning rate  [default: 0.001]
  --distributed BOOLEAN     distributed training  [default: False]
  --help                    Show this message and exit.
  
  
>> python train_model.py
```

For distributed training on multiple gpu.
```
>> python train_model.py --distributed Ture
```
**Computation Graph of GPT-2 Model.**

                              **Decoder**
<img src="/images/GPT-2_Decoder.jpg" alt="Decoder Graph" height="750" width="900"/>
                              **GPT-2**
<img src="/images/GPT-2_Graph.jpg" alt="GPT-2_Graph" height="850" width="900"/>


