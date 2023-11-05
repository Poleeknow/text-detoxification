# Text-detoxification
Polina Bazhenova, DS21-01

p.bazhenova@innopolis.university

# Guideline

1. First of all, you need to clone the repository.
2. Next, you should download all required packages and modules specified in `requirements.txt` file.
3. To load the dataset you need to launch `src/data/download.py`
4. In `src` directory you can find all nessesary scripts:
    * `src/data` contains `metric_model.py` with script of training and evaluating model which is used to estimate main models, and `preprocess.py` with script of data preprocessing for the main models.
    * `src/models` contains `seq2seq_train.py` with train script of Seq2Seq LSTM model, `t5-paraph_train.py` and `t5-paraph_predict.py` with train and prediction scripts  of [t5 paraphraseing model](https://huggingface.co/mrm8488/t5-small-finetuned-quora-for-paraphrasing) correspondently, `t5-small_train.py` and `t5-small_predict.py` with strain and prediction scripts of [t5 small model](https://huggingface.co/t5-small) correspondently.
5. Additionally, you can use Jupiter notebooks from the `notebooks` directory for better visual perception.
6. And finally, in the `reports` directory you can find reports describing this project. `First report.pdf` describes path in solution creation process. And `Final report.pdf` describes your final solution.