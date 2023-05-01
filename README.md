# Tweets Sentiment Classification
Using Data Science with Machine Learning and NLP techniques to detect the Tweet Sentiment using significant features given by the most linked features that extraction from that are taken into consideration when evaluating the Sentiment of Tweet.

## Table of Contents
* [Introduction](#introduction)
* [Dataset General info](#dataset-general-info)
* [Evaluation](#evaluation)
* [Technologies](#technologies)
* [Setup](#setup)
* [Features](#features)
* [Run Example](#run-example)
* [Sources](#sources)

## Introduction
Natural Language Processing has had a great impact as one of the most influential and used languages in the fields of Artificial Intelligence, which have been helped by the development of Natural Language Processing techniques and its algorithms and hardware/resources, one of these areas of Natural Language Processing is the sentiments analysis, which take the rule of feelings identification, which I present to you.\
Based on this introduction, I present to you my project in solving the problem of Tweets Sentiment Classification. In this project, I tried to expand the use of Machine Learning algorithms only with NLP techniques, with the use of three types of vectorizers in these operations, which are among the most popular types in Machine Learning and NLP. and my suggestions for solving it with the best possible ways and the current capabilities using Data Science, Machine Learning and NLP.\
Hoping to improve it gradually in the coming times.

## Dataset General info
**General info about the dataset:**
* About:\
This data set contains three columns:
1. Index -- (Integer type)
2. message to examine -- (String type)
3. label (depression result) -- (Integer type, had label encoder process, or labelled 0, 1 indicated the depression)
* Context:\
Finding if a person is depressed from their use of words on social media can definitely help in the cure!

## Evaluation

1. Confusion matrix analysis.
2. Features Selection.
3. Compute ROC of the best performance.
4. Cross-Validation.
5. Learning Curve.

## Technologies
* Programming language: Python.
* Libraries: numpy, python-math, collection, more-itertools, DateTime, regex, strings, matplotlib, pandas, seaborn, beautifulsoup4, nltk, wordcloud, gensim, scikit-learn, scipy, imblearn, xgboost. 
* Application: Jupyter Notebook.

## Setup
To run this project setup the following libraries on your local machine using pip on the terminal after installing Python:\
'''\
pip install numpy\
pip install python-math\
pip install collection\
pip install more-itertools\
pip install DateTime\
pip install regex\
pip install strings\
pip install matplotlib\
pip install pandas\
pip install seaborn\
pip install beautifulsoup4\
pip install nltk\
pip install wordcloud\
pip install gensim\
pip install scikit-learn\
pip install scipy\
pip install imblearn\
pip install xgboost\
'''\
To install these packages with conda run:\
'''\
conda install -c anaconda numpy\
conda install -c conda-forge mpmath\
conda install -c lightsource2-tag collection\
conda install -c conda-forge more-itertools\
conda install -c trentonoliphant datetime\
conda install -c conda-forge re2\
conda install -c conda-forge r-stringi\
conda install -c conda-forge matplotlib\
conda install -c anaconda pandas\
conda install -c anaconda seaborn\
conda install -c anaconda bs4\
conda install -c anaconda nltk\
conda install -c conda-forge wordcloud\
conda install -c anaconda gensim\
conda install -c anaconda scikit-learn\
conda install -c anaconda scipy\
conda install -c conda-forge imbalanced-learn\
conda install -c anaconda py-xgboost\
'''

## Features
* I present to you my project solving the problem of Tweets Sentiment Classification using a lot of effective algorithm and techniques with a good analysis (EDA), and comparing between them using logical thinking, and put my suggestions for solving it in the best possible ways and the current capabilities using Data Science, Machine Learning and NLP.

### To Do:
**Briefly about the process of the project work, here are (some) insights that I took care of it:**

* Explored the dataset
* Doing some EDA
* Data Cleaning
1. Remove HTMLs.
2. Remove URLs.
3. Remove Images.
4. Remove Mentions.
5. Remove Emoji.
5. Remove Emoticons.
6. Remove non-ASCII character.
7. Remove Punctuation.
8. Remove Extra Alphabatic characters.
9. Remove First and End spaces.
10. Remove Numbers.
11. Convert text words to lower case.
12. Remove Single character.

* Data Preprocessing
1. Remove stop words.
2. Lemmatization.
3. Stemming.
4. Perform Training data and Validation sets.

* Deep EDA
1. Find Sents, Tokens, Lemma, POS, NER.
2. Find the best N-Grams.
3. Checking the most frequent words and their counts.

* Preprocess and modeling
1. Processing imbalanced target in data.
trying to handle imbalanced data through up-sampling techniques.
2. Vectorizing using:\
2.1 Count Vectorizer.\
2.2 TF-IDF Vectorizer.\
2.3 Word2Vec.

* Checking the best N-Grams and find if it's possible to take the count on each word into consideration to select the best performance with the target.
  
 *Used various classifying algorithms to aim best results:

* Decision Tree, Random Forest, Extra Trees, KNN, SVC, XGBoost, SGD, Voting, Multinomial NB, Ada Boost, Gradient Boosting, Bagging, Logistic Regression, Stacking, and Ridge Classifier.

*Also making analysis using:
1. Confusion matrix analysis.
2. Features Selection.
3. Compute ROC of the best performance.
4. Cross-Validation.
5. Learning Curve.

**Finally:**

* Using Halving Grid Search CV to tuning the best models parameters.
## Run Example

To run and show analysis, insights, correlation, and results between any set of features of the dataset, here is a simple example of it:

* Note: you have to use a jupyter notebook to open this code file.

1. Run the importing stage.

2. Load the dataset.

3. Select which cell you would like to run and show its output.

4. Run Selection/Line in Python Terminal command (Shift+Enter).

## Sources
This data was taken from Kaggle:\
(https://www.kaggle.com/datasets/gargmanas/sentimental-analysis-for-tweets)
