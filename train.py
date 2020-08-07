import argparse
import glob
import os
from hyperopt import STATUS_OK, hp, tpe, Trials, fmin
import math
import numpy as np
from joblib import dump
import json
import scipy.sparse
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import shutil
from utils import parse_xml_, preprocess_, get_labels

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dataset", type=str, help="Directory contains the XML of the datasets of english and spanish.")
    parser.add_argument('output_dir', type=str, help="Directory to store vectorizer, scaler, SVD and trained mododels of both languages.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    # logistic_regression,tfid_word_ngram1-2_maxdf0.6_mindf0.03_4329,0.681,0.76
    en_to_configuration = {
        "tokenizer_lang": "english",
        "vectorizer": [
            ("word", (1, 2), 0.6, 0.03),
        ],
        "svd": None,
        "scaler": True,
        "normalizer": False,
        "model": "logistic_regression",
    }

    #  logistic_regression,tfid_char_ngram1-3_maxdf0.7_mindf0.05-tfid_word_ngram1-2_maxdf0.7_mindf0.05_5768,0.797,0.876
    es_to_configuration = {
        "tokenizer_lang": "spanish",
        "vectorizer": [
            ("char", (1, 3), 0.7, 0.05),
            ("word", (1, 2), 0.7, 0.05),
        ],
        "svd": None,
        "scaler": True,
        "normalizer": False,
        "model": "logistic_regression",
    }

    languages_to_resources = {
        "es": es_to_configuration,
        "en": en_to_configuration,
    }

    input_dir = args.input_dataset
    output_dir = args.output_dir

    def tfid_vectorizer(list_text, ngram_range, analyzer, dtype=np.float64, max_df=1, min_df=1):
        parameters = {
            "ngram_range": ngram_range, 
            "analyzer": analyzer, 
            "lowercase": True, 
            "strip_accents": 'unicode', 
            "dtype": dtype, 
            "decode_error": 'replace',
            "max_df": max_df,
            "min_df": min_df,
        }
        vectorizer = TfidfVectorizer(**parameters)
        features = vectorizer.fit_transform(list_text)
        return vectorizer, features

    for language in languages_to_resources.keys():
        output_dir_lang = os.path.join(output_dir, language)
        
        if not os.path.isdir(output_dir_lang):
            os.makedirs(output_dir_lang)
        
        tokenizer_lang, vectorizer_, svd_, scaler_, normalizer_, model_ = languages_to_resources[language].values()
        input_dir_lang = os.path.join(input_dir, language)
        file_labels = os.path.join(input_dir_lang, "truth.txt")
        author_to_label = get_labels(file_labels)
        authors_id = [os.path.splitext(os.path.basename(xml_file))[0] for xml_file in sorted(glob.glob(f"{input_dir_lang}/*.xml"))]
        labels = np.empty(len(authors_id), dtype=int)
        preprocessed_list = []
        
        for (i, author_id) in enumerate(authors_id):
            labels[i] = author_to_label[author_id]
            xml_file = os.path.join(input_dir_lang, f"{author_id}.xml")
            # 1. Parse
            parsed_lines = parse_xml_(xml_file)
            # 2. Preprocess
            preprocessed_author = preprocess_(parsed_lines, tokenizer_lang)
            preprocessed_list.append(preprocessed_author)
        
        # 3. Vectorize
        if len(vectorizer_) == 2:
            (analyzer_a, ngram_range_a, max_df_a, min_df_a) = vectorizer_[0]
            (analyzer_b, ngram_range_b, max_df_b, min_df_b) = vectorizer_[1]
            vectorizer_a, features_a = tfid_vectorizer(preprocessed_list, ngram_range_a, analyzer_a, max_df=max_df_a, min_df=min_df_a)
            vectorizer_b, features_b = tfid_vectorizer(preprocessed_list, ngram_range_b, analyzer_b, max_df=max_df_b, min_df=min_df_b)
            features = scipy.sparse.hstack([features_a, features_b])
            dump(vectorizer_a, os.path.join(output_dir_lang, f"vectorizer_{analyzer_a}_{ngram_range_a[0]}_{ngram_range_a[1]}_{max_df_a}_{min_df_a}"))
            dump(vectorizer_b, os.path.join(output_dir_lang, f"vectorizer_{analyzer_b}_{ngram_range_b[0]}_{ngram_range_b[1]}_{max_df_b}_{min_df_b}"))
        else:
            (analyzer, ngram_range, max_df, min_df) = vectorizer_[0]
            vectorizer, features = tfid_vectorizer(preprocessed_list, ngram_range, analyzer, max_df=max_df, min_df=min_df)
            dump(vectorizer, os.path.join(output_dir_lang, f"vectorizer_{analyzer}_{ngram_range[0]}_{ngram_range[1]}_{max_df}_{min_df}"))
        
        # 4. Reduce dimensionality (svr, logistic_regression)
        if svd_:
            svd = TruncatedSVD(n_components=svd_, random_state=args.seed)
            features = svd.fit_transform(features)
            dump(svd, os.path.join(output_dir_lang, f"svd_{svd_}"))
        
        # 5. Scale (svr, logistic_regression)
        if scaler_:
            scaler = preprocessing.StandardScaler().fit(features.toarray())
            features = scaler.transform(features.toarray())
            dump(scaler, os.path.join(output_dir_lang, f"scaler"))
        
        # 6. Normalize (multinomial_nb)
        if normalizer_:
            features = preprocessing.Normalizer(copy=False).fit_transform(features)
        
        # 7. Train
        if model_ == "logistic_regression":
            space_grid = {
                'C': hp.choice('C', [0.25, 0.5, 1.0]),
            }
            def objective(space):
                clf = LogisticRegression(
                    random_state=args.seed,
                    C=space['C'],
                )
                clf.fit(features, labels)
                hipotesys = clf.predict(features)
                acc = accuracy_score(labels, hipotesys)
                loss = 1 - acc
                return {'loss':loss, 'status': STATUS_OK, 'acc': acc, "clf" : clf}
        elif model_ == "svr":
            space_grid = {
                'C': hp.loguniform('C', math.log(1e-5), math.log(1e5)),
                'tol': hp.loguniform('tol', math.log(1e-5), math.log(1e-2)),
                'intercept_scaling': hp.loguniform('intercept_scaling', math.log(1e-1), math.log(1e1)),
            }
            def objective(space):
                clf = LinearSVC(random_state=args.seed, tol=space['tol'], C=space['C'], intercept_scaling=space['intercept_scaling'])
                clf.fit(features, labels)
                hipotesys = clf.predict(features)
                acc = accuracy_score(labels, hipotesys)
                loss = 1 - acc
                return {'loss':loss, 'status': STATUS_OK, 'acc': acc, "clf" : clf}
        else: #multinomial_nb
            space_grid = {
                'alpha': hp.loguniform('alpha', -3, 5),
            }
            def objective(space):
                clf = MultinomialNB(alpha=space['alpha'])
                clf.fit(features, labels)
                hipotesys = clf.predict(features)
                acc = accuracy_score(labels, hipotesys)
                loss = 1 - acc
                return {'loss':loss, 'status': STATUS_OK, 'acc': acc, "clf" : clf}
        
        trials = Trials()
        fmin(fn=objective, space=space_grid, algo=tpe.suggest,verbose=False,
            max_evals=1, trials=trials, rstate=np.random.RandomState(args.seed))
        
        dump(trials.results[0]['clf'], os.path.join(output_dir_lang, f"clf"))

        print(f"{language} - Accuracy: {trials.results[0]['acc']:.3f}")
        
        configuration_str = json.dumps(languages_to_resources[language])
        with open(os.path.join(output_dir_lang, f"configuration.json"), "w") as fjson:
            fjson.write(configuration_str)
        
if __name__ == "__main__": 
    main()

    