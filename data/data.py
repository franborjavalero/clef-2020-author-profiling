from joblib import dump
import os
import numpy as np
import random
import shutil
import scipy.sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from utils import read_dir

def create_cv_data_pipeline(dir_data, fname_labels, num_folds=10, seed=42):
    random.seed(seed)
    total_corpus = read_dir(dir_data, fname_labels)
    random.shuffle(total_corpus)
    num_examples = len(total_corpus)
    num_examples_fold = num_examples // num_folds
    data_folds = []
    for i in range(0, num_examples, num_examples_fold):
        j = min(i + num_examples_fold, num_examples)
        data_folds.append(total_corpus[i:j])
    return data_folds

def _get_input_label(list_tuples):
    inputs_ = [ tuple_[0] for tuple_ in list_tuples]
    labels_ = [ tuple_[1] for tuple_ in list_tuples]
    return inputs_, labels_

def _remove_outer_list(old_list):
    new_list = [element for list_ in old_list for element in list_]
    return new_list

def _dimensionality_reduction(dir_features_full, list_n_components, features_train, features_dev, seed=42):
    list_n_components_ = list(filter(lambda dim: dim < features_train.shape[1], list_n_components))
    print(list_n_components_)
    print(features_train.shape[1])
    for n_components_ in list_n_components_:
        print(f"\t{n_components_}")
        dir_features_reduced = f"{dir_features_full}_{n_components_}"
        os.makedirs(dir_features_reduced)
        svd = TruncatedSVD(n_components=n_components_, random_state=seed)
        features_train_reduced = svd.fit_transform(features_train)
        features_dev_reduced = svd.transform(features_dev)
        np.save(os.path.join(dir_features_reduced, "train"), features_train_reduced)
        np.save(os.path.join(dir_features_reduced, "dev"), features_dev_reduced)

def extract_features_cv_fold(data_folds, output_dir, n_gram_words, n_gram_chars, max_dfs, min_dfs):
    num_fold_ = len(data_folds)
    analyzers = ["word", "char"]
    analyzer_to_ngram = {
        "word": n_gram_words,
        "char": n_gram_chars,
    }
    methods = [
        #"count", 
        "tfid",
    ]
    method_to_func = {
        #"count": CountVectorizer,
        "tfid": TfidfVectorizer,
    }
    for i in range(num_fold_):
        dir_features_fold = os.path.join(output_dir, f"{i}")
        if os.path.isdir(dir_features_fold):
            shutil.rmtree(dir_features_fold)
        os.makedirs(dir_features_fold)  
        """ Cross-validation split """
        if i == 0:
            data_train = data_folds[1:]
            data_dev =  data_folds[0]
        elif i < (num_fold_-1):
            data_train = data_folds[:i] + data_folds[i+1:]
            data_dev =  data_folds[i]
        else:
            data_train = data_folds[:-1]
            data_dev =  data_folds[-1]
        data_train = _remove_outer_list(data_train)
        train_X, train_Y = _get_input_label(data_train)
        dev_X, dev_Y = _get_input_label(data_dev)
        train_Y = np.array(train_Y) 
        dev_Y = np.array(dev_Y)
        dir_truth_fold = os.path.join(dir_features_fold, "truth")
        os.makedirs(dir_truth_fold)
        np.save(os.path.join(dir_truth_fold, "train"), train_Y)
        np.save(os.path.join(dir_truth_fold, "dev"), dev_Y)
        """ Extract features """
        for method in methods:
            function = method_to_func[method]
            for analyzer in analyzers:
                n_grams = analyzer_to_ngram[analyzer]
                for n_gram in n_grams:
                    for max_df in max_dfs:
                        for min_df in min_dfs:
                            features_name = f"{method}_{analyzer}_ngram{n_gram[0]}-{n_gram[1]}_maxdf{max_df}_mindf{min_df}"
                            parameters = {
                                "ngram_range": n_gram, 
                                "analyzer": analyzer, 
                                "lowercase": True, 
                                "strip_accents": 'unicode', 
                                "dtype": np.float64, 
                                "decode_error": 'replace',
                                "max_df": max_df,
                                "min_df": min_df,
                            }
                            try:
                                vectorizer = function(**parameters)
                                features_train = vectorizer.fit_transform(train_X)
                                features_dev = vectorizer.transform(dev_X)
                                print(f"{i} | {features_name}")
                                # save features
                                dir_features_full = os.path.join(dir_features_fold, features_name)
                                os.makedirs(dir_features_full)
                                scipy.sparse.save_npz(os.path.join(dir_features_full, "train"), features_train)
                                scipy.sparse.save_npz(os.path.join(dir_features_full, "dev"), features_dev)
                            except ValueError:
                                continue