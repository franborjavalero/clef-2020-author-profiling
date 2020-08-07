# Logistic regression with TF-IDF features
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is an open source implementation of our solution to the competition: [PAN@CLEF 2020 Author Profiling](https://pan.webis.de/clef20/pan20-web/author-profiling.html).

Our approach uses a logistic regressor model with character and word n-grams TF-IDF features.

## Dependencies

- Python 3.7
- We need the following packages (using pip):

```
pip install hyperopt
pip install joblib
pip install scikit-learn
pip install nltk
pip install tweet-preprocessor
```
## Results

Our approach achieves the third best solution in the private test, the results are shown in the table below:

| LANG | ACC  |
|------|------|
| ES   | 0.78 |
| EN   | 0.73 |

Our team is **deborjavalero20**, you can check the full ranking in this [link](https://pan.webis.de/clef20/pan20-web/author-profiling.html#evaluation)

## Usage

The commands below show how to replicate the experiments. 

The train.py script trains the Spanish and English models using the corpus located at `DATA_DIR` and stores the trained models on  `RESOURCES_DIR`.

```
python3 train.py DATA_DIR RESOURCES_DIR
```

The test.py script generates the Spanish and the English hypothesis. The argument `DATA_DIR` is the folder of the input data, and the argument `HYPOTHESIS_DIR` will be the directory to store own hypothesis.
```
python3 test.py -c DATA_DIR -o HYPOTHESIS_DIR
```

## License
 
The MIT License (MIT)

Copyright (c) 2020 Francisco de Borja Valero
