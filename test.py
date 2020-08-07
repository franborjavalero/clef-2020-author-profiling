import argparse
import glob
import os
from joblib import load
import scipy.sparse
from sklearn import preprocessing
import shutil
from utils import parse_xml_, preprocess_, prediction_xml_file

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="$inputDataset")
    parser.add_argument('-o', type=str, help='$outputDir')
    parser.add_argument('-r', type=str, default=None, help='$inputRun')
    parser.add_argument('--dir_resources', type=str, default="./resources", help='Directoy where are located: vectorizer, scaler, model...; all the fited models')
    args = parser.parse_args()

    # 1: logistic_regression,tfid_word_ngram1-2_maxdf0.6_mindf0.03_4329,0.681,0.76
    en_to_configuration = {
        "tokenizer_lang": "english",
        "vectorizer": [
            ("word", (1, 2), 0.6, 0.03),
        ],
        "svd": None,
        "scaler": True,
        "normalizer": False,
    }
    
    # 2: logistic_regression,tfid_char_ngram1-3_maxdf0.7_mindf0.05-tfid_word_ngram1-2_maxdf0.7_mindf0.05_5768,0.797,0.876
    es_to_configuration = {
        "tokenizer_lang": "spanish",
        "vectorizer": [
            ("char", (1, 3), 0.7, 0.05),
            ("word", (1, 2), 0.7, 0.05),
        ],
        "svd": None,
        "scaler": True,
        "normalizer": False,
    }
    
    languages_to_configuration = {
        "es": es_to_configuration,
        "en": en_to_configuration,
    }

    input_dir = args.c
    output_dir = args.o
    resources_dir = args.dir_resources

    for language in languages_to_configuration.keys():
        tokenizer_lang, vectorizer_, svd_, scaler_, normalizer_ = languages_to_configuration[language].values()
        input_dir_lang = os.path.join(input_dir, language)
        resources_dir_lang = os.path.join(resources_dir, language)
        authors_id = [os.path.splitext(os.path.basename(xml_file))[0] for xml_file in sorted(glob.glob(f"{input_dir_lang}/*.xml"))]
        preprocessed_list = []
        
        for author_id  in authors_id:
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
            vectorizer_a = load(os.path.join(resources_dir_lang, f"vectorizer_{analyzer_a}_{ngram_range_a[0]}_{ngram_range_a[1]}_{max_df_a}_{min_df_a}"))
            vectorizer_b = load(os.path.join(resources_dir_lang, f"vectorizer_{analyzer_b}_{ngram_range_b[0]}_{ngram_range_b[1]}_{max_df_b}_{min_df_b}"))
            features_a, features_b = vectorizer_a.transform(preprocessed_list), vectorizer_b.transform(preprocessed_list)
            features = scipy.sparse.hstack([features_a, features_b])
        else:
            (analyzer, ngram_range, max_df, min_df) = vectorizer_[0]
            vectorizer = load(os.path.join(resources_dir_lang, f"vectorizer_{analyzer}_{ngram_range[0]}_{ngram_range[1]}_{max_df}_{min_df}"))
            features = vectorizer.transform(preprocessed_list)
        
        # 4. Reduce dimensionality (svr, logistic_regression)
        if svd_:
            svd = load(os.path.join(resources_dir_lang, f"svd_{svd_}"))
            features = svd.transform(features)
        
        # 5. Scale (svr, logistic_regression)
        if scaler_:
            scaler = load(os.path.join(resources_dir_lang, f"scaler"))
            features = scaler.transform(features.toarray())
        
        # 6. Normalize (multinomial_nb)
        if normalizer_:
            features = preprocessing.Normalizer(copy=False).fit_transform(features)
        
        # 7. Classify
        model = load(os.path.join(resources_dir_lang, f"clf"))
        hypothesis = model.predict(features)
        
        # 8. Export hypothesis
        output_dir_lang = os.path.join(output_dir, language)
        if os.path.isdir(output_dir_lang):
            shutil.rmtree(output_dir_lang)
        os.makedirs(output_dir_lang)  
        for (idx, author_id) in enumerate(authors_id):
            prediction_xml_file(output_dir_lang, author_id, language, hypothesis[idx])


if __name__ == "__main__": 
    main()

    