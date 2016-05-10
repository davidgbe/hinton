# Hinton

## Getting started 
1. Clone this repo
1. Install virtualenv using pip
  * You can do this with `pip install virtualenv`
1. Activate virtual env
  * `source venv/bin/activate`
  * `deactivate` to deactivate virtualenv
1. Install requirements using pip
  * To do this: `pip install -r ./requirements.txt`
1. Run the code!
```
python ./run.py word2vec
````
or 
````
python ./run.py crf
```

We have also implemented logistic and hmm classifers. They can be run with

```
python ./run_logistic.py
```
and
```
python ./run_hmm.py
```
