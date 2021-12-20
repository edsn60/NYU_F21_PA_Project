import stanza
import numpy as np
import json
from typing import List, Dict
import math
import pickle
from stopwords import stop_words
from preprocessing import save_model_path, save_preprocessed_data_path
from collections import Counter
import glob
import os
from pathlib import Path
from datetime import datetime

if not os.path.exists(os.path.join(Path.home(), 'stanza_resources')):
    stanza.download()

class Predict:
    def __init__(self, filename_suffix: str = ""):
        self._models = []
        for i in glob.glob(os.path.join(save_model_path, f"model_tfidf_*_{filename_suffix}.pickle")):
            with open(i, "rb") as fp:
                self._models.append(pickle.load(fp))
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Model loaded")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Loading all tokens from disk...")
        with open(os.path.join(save_preprocessed_data_path, f"all_token_{filename_suffix}.json"), "r") as fp:
            self._all_token: Dict = json.loads(fp.read())
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} All tokens loaded")
        self._pipeline: stanza.Pipeline = stanza.Pipeline("en", processors="tokenize, pos, lemma, ner")
        self._idf = np.genfromtxt(os.path.join(save_preprocessed_data_path, f"idf_{filename_suffix}.csv"), delimiter=",")

    @staticmethod
    def _remove_stop_words(s: str) -> str:
        res = []
        s1 = s.split(" ")
        for i in s1:
            if i and i not in stop_words.keys():
                res.append(i)
        return " ".join(res)

    def _get_idf(self, tf) -> List:
        tf_T = tf.T.tolist()
        idf: List = []
        for token in tf_T:
            num_zeros: np.ndarray = token.count(0)
            total_nums: int = len(token)
            idf.append(math.log2((total_nums) / (1 + total_nums - num_zeros)))
        return idf

    def _preprocess_new_twitter(self, new_tweet) -> np.ndarray:

        doc: stanza.Document = self._pipeline(new_tweet)
        new_tf: List = [0.0 for i in range(len(self._all_token))]
        for sent in doc.sentences:
            for token in sent.tokens:
                if token.ner == "O" and token.words[0].lemma in self._all_token.keys() and token.words[0].lemma not in stop_words.keys():
                    new_tf[self._all_token[token.words[0].lemma]] += 1
            for ner in sent.ents:
                if ner.text in self._all_token.keys() and ner.text not in stop_words.keys():
                    new_tf[self._all_token[ner.text]] += 1

        for i in range(len(self._idf)):
            new_tf[i] *= self._idf[i]

        return np.array(new_tf)

    def predict(self, new_tweet: str):
        x = self._preprocess_new_twitter(new_tweet)
        pred = []
        for model in self._models:
            pred.append(model.predict(np.array([x]))[0])

        if 1 not in pred:
            return 0
        if 0 not in pred:
            return 1

        counter = Counter(pred)
        return 1 if counter[1] >= counter[0] else 0
