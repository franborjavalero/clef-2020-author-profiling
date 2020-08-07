from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer 
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
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