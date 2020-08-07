import glob
import itertools
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer 
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import os
import preprocessor as p
import re
import shutil
import xml.etree.ElementTree as ET

list_remove_tokens = ["amp"]
p.set_options(p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER)
tokenizer_ = TweetTokenizer(strip_handles=False, reduce_len=True, preserve_case=False)

lang2stemmer = {
    "spanish": SnowballStemmer("spanish", ignore_stopwords=True),
    "english": PorterStemmer(),
}

def read_file(file_name, encoding='utf-8'):
    with open(file_name, 'r', encoding=encoding) as file:
        data = file.read().rstrip("\n").split("\n")
    return data

def get_labels(file_name, encoding='utf-8'):
    dict_ = dict()
    data_labels = read_file(file_name)
    for line in data_labels:
        label = int(line[-1])
        user = line[:-4]
        dict_[user] = label
    return dict_

def _list_to_txt(file_name, list_, encoding='utf-8'):
    with open(file_name, 'w', encoding=encoding) as file_:
        for element_ in list_:
            file_.write(f"{element_}\n")

def combination_hyperparameters(list_hyperparameters):
    return list(itertools.product(*list_hyperparameters))

def save_cv_folds(data, data_dir):
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)
    for (idx_fold, data_fold) in enumerate(data):
        data_dir_fold = os.path.join(data_dir, f"{idx_fold}")
        os.makedirs(data_dir_fold)
        fname_data_fold = os.path.join(data_dir_fold,"data.txt")
        fname_label_fold = os.path.join(data_dir_fold ,"truth.txt")
        tweets = [sample[0] for sample in data_fold]
        _list_to_txt(fname_data_fold, tweets)
        labels = [sample[1] for sample in data_fold]
        _list_to_txt(fname_label_fold, labels)

def read_dir(dir_name, fname_labels, encoding='utf-8'):
    user2label = get_labels(fname_labels)
    docs_ = []
    for file_name  in glob.glob(f"{dir_name}/*.txt"):
        user_ = os.path.basename(file_name)[:-4]
        try :
            label_ = user2label[user_]
            readed_doc = (read_file(file_name)[0], label_) # (string, int)
            docs_.append(readed_doc)
        except:
            print(f"User: {user_} has not a label.")
    return docs_

def prediction_xml_file(output_dir, user_id, lang, label):
    author = ET.Element("author",
        {
            'id': f"{user_id}",
            'lang': f"{lang}",
            'type': f"{label}",
        }
    )
    tree = ET.ElementTree(author)
    file_name = os.path.join(output_dir, f"{user_id}.xml")
    tree.write(file_name)

def _clean_tweet(tweet):
    # removes duplicate spaces and new lines
    tweet_ = re.sub(' +', ' ', tweet)
    tweet_ = re.sub('\n', '', tweet_)
    return tweet_

def parse_xml(raw_input, parse_output, encoding='utf-8'):
    tree = ET.parse(raw_input)
    with open(parse_output, 'w', encoding=encoding) as f:
        for tweets in tree.getroot():
            for tweet in tweets:
                tweet_ = _clean_tweet(tweet.text)
                f.write(f"{tweet_}\n")

def parse_xml_(raw_input):
    lines = []
    tree = ET.parse(raw_input)
    for tweets in tree.getroot():
        for tweet in tweets:
            tweet_ = _clean_tweet(tweet.text)
            lines.append(tweet_)
    return lines

def _preprocess_tags(tweet):
    tweet = tweet.replace("RT #USER#:", "rt")
    tweet = tweet.replace("#USER#", "user")
    tweet = tweet.replace("#URL#", "url")
    tweet = tweet.replace("#HASHTAG#", "hashtag")
    return tweet

def _preprocess_tweet(tweet, stop_words, stemmer):
    tweet_ = _preprocess_tags(tweet)
    tweet_ = p.tokenize(tweet_) # emoji, smiley, number
    tweet_ = tokenizer_.tokenize(tweet_)
    def https2url(token):
        if "https" in token:
            return "url"
        else:
            return token
    tweet_ = list(map(https2url, tweet_))
    # removes all not word token:
    tweet_ = list(filter(lambda token: token.isalpha() and len(token) > 1, tweet_))
    # removes all the stop words:
    tweet_ = list(filter(lambda token: token not in stop_words, tweet_))
    # stemming:
    tweet_ = [stemmer.stem(token) for token in tweet_]
    tweet_ = " ".join(tweet_)
    return tweet_

def preprocess_txt(parse_input, preprocess_output, language, encoding='utf-8'):
    stop_words = frozenset(stopwords.words(language) + list_remove_tokens)
    stemmer = lang2stemmer[language]
    parse_tweets = read_file(parse_input, encoding=encoding)
    with open(preprocess_output, 'w', encoding=encoding) as f:
        for tweet in parse_tweets[:-1]:
            processed_tweet = _preprocess_tweet(tweet, stop_words, stemmer) 
            f.write(f"{processed_tweet} eot ") # end of tweet
        processed_tweet = _preprocess_tweet(parse_tweets[-1], stop_words, stemmer) 
        f.write(f"{processed_tweet} eot") # end of tweet

def preprocess_(parsed_tweets, language):
    stop_words = frozenset(stopwords.words(language) + list_remove_tokens)
    stemmer = lang2stemmer[language]
    lines = []
    for tweet in parsed_tweets[:-1]:
        processed_tweet = _preprocess_tweet(tweet, stop_words, stemmer)
        lines.append(f"{processed_tweet} eot ") # end of tweet
    processed_tweet = _preprocess_tweet(parsed_tweets[-1], stop_words, stemmer)
    lines.append(f"{processed_tweet} eot") # end of tweet
    return "".join(lines)