import stanza
from stopwords import stop_words
import numpy as np
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
import json
from sklearn import metrics
from typing import List, Dict
import pickle
import os
from datetime import datetime
from pathlib import Path
from warnings import warn

if not os.path.exists(os.path.join(Path.home(), 'stanza_resources')):
    stanza.download()

save_model_path = "./models"
save_preprocessed_data_path = "./preprocessed_data"
raw_dataset_path = "./datasets"

gnb = GaussianNB()
mnb = MultinomialNB()
random_forest = RandomForestClassifier()
logistic_regression = LogisticRegression()


class Preprocessing:
    def __init__(self, csv_file_path: List[str], filename_suffix: str = "", from_beginning=True):
        self._stanza_pipepline: stanza.Pipeline = stanza.Pipeline("en", processors="tokenize, pos, lemma, ner")
        self._raw_data = csv_file_path
        self._processor: List = []
        self._from_beginning = from_beginning
        self._filename_suffix = filename_suffix
        self._tfidf = None
        self._is_pipeline_constructed = False

    @staticmethod
    def _remove_stop_words(s: str) -> str:
        res = []
        s1 = s.split(" ")
        for i in s1:
            if i and i not in stop_words.keys():
                res.append(i)
        return " ".join(res)

    def _construct_pipeline(self):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Constructing Pipeline...")
        input_docs = [stanza.Document([], text=t) for t in self._raw_data]
        self._processor = self._stanza_pipepline(input_docs)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Pipeline Constructed")

    def get_all_tokens(self) -> Dict:
        all_token_path = os.path.join(save_preprocessed_data_path, f"all_token_{self._filename_suffix}.json")

        if os.path.exists(all_token_path):
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Getting all tokens from file...")
            with open(all_token_path, "r") as fp:
                all_tokens = json.loads(fp.read())
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} All tokens got")
            return all_tokens
        self._construct_pipeline()
        self._is_pipeline_constructed = True
        all_tokens = {}
        idx = 0
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Getting all tokens...")
        for doc in self._processor:
            for sent in doc.sentences:
                for token in sent.tokens:
                    if token.ner == "O":
                        if token.words[0].upos != "SYM" and token.words[0].upos != "PUNCT" and token.words[0].lemma not in all_tokens and token.words[0].lemma not in stop_words.keys():
                            all_tokens.update({token.words[0].lemma: idx})
                            idx += 1

                for ner in sent.ents:
                    if ner.text not in all_tokens and ner.text not in stop_words.keys():
                        all_tokens.update({ner.text: idx})
                        idx += 1
        with open(os.path.join(save_preprocessed_data_path, f"all_token_{self._filename_suffix}.json"), "w") as fp:
            fp.write(json.dumps(all_tokens))
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} All tokens got")
        return all_tokens

    def get_tfidf(self) -> np.ndarray:
        all_tokens = self.get_all_tokens()
        if not self._is_pipeline_constructed:
            self._construct_pipeline()
            self._is_pipeline_constructed = True
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Computing TF-IDF matrix...")
        tf = []
        for doc in self._processor:
            doc_tf = [0.0 for i in range(len(all_tokens))]
            for sent in doc.sentences:
                for token in sent.tokens:
                    if token.ner == "O" and token.words[0].upos != "SYM" and token.words[0].upos != "PUNCT" and token.words[0].lemma in all_tokens.keys():
                        doc_tf[all_tokens[token.words[0].lemma]] += 1

                for ner in sent.ents:
                    if ner.text in all_tokens.keys():
                        doc_tf[all_tokens[ner.text]] += 1

            tf.append(doc_tf)
        tf = np.array(tf)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} TF computed")
        tf_T = tf.T
        tf_T = tf_T.tolist()
        if not os.path.exists(os.path.join(save_preprocessed_data_path, f"idf_{self._filename_suffix}.csv")):
            idf = []
            for token in tf_T:
                num_zeros = token.count(0)
                total_nums = len(token)
                idf.append(math.log2((total_nums) / (1 + total_nums - num_zeros)))
            np.savetxt(os.path.join(save_preprocessed_data_path, f"idf_{self._filename_suffix}.csv"), np.array(idf),
                       delimiter=",")
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IDF computed")
        else:
            idf = np.genfromtxt(os.path.join(save_preprocessed_data_path, f"idf_{self._filename_suffix}.csv"))
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IDF restored")
        tfidf = tf.T
        for i in range(len(tfidf)):
            tfidf[i] *= idf[i]

        tfidf = tfidf.T
        idf = np.array(idf)

        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} TF-IDF computed")
        self._tfidf = tfidf
        return tfidf

    def feature_selection(self, x: np.ndarray, y: np.ndarray, k: int = 1000):
        if x.shape[1] < 1000:
            warn(f"The provided 'k' is larger than the number of features provided, which is {x.shape[1]}. Using {x.shape[1]} instead.")
            k = x.shape[1]
        select = SelectKBest(chi2, k=k)
        x = select.fit_transform(x, y)
        feature_mask = select.get_support(indices=True)
        all_token = self.get_all_tokens()
        all_token_lst = list(all_token.keys())
        selected_tokens = {}
        selected_idf = []
        idf = np.genfromtxt(os.path.join(save_preprocessed_data_path, f"idf_{self._filename_suffix}.csv"))
        for i in range(len(feature_mask)):
            token_id = feature_mask[i]
            token = all_token_lst[token_id]
            selected_tokens[token] = i
            selected_idf.append(idf[token_id])

        np.savetxt(os.path.join(save_preprocessed_data_path, f"idf_selected_{k}.csv"), selected_idf, delimiter=",")

        with open(os.path.join(save_preprocessed_data_path, f"all_tokens_selected_{k}.json"), "w") as fp:
            fp.write(json.dumps(selected_tokens))

        return x


def calculate_metric(model: str, label, pred):
    total = len(label)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(total):
        if label[i] == 0:
            if pred[i] == 0:
                TN += 1
            else:
                FP += 1
        else:
            if pred[i] == 0:
                FN += 1
            else:
                TP += 1
    print(f"{model}: \n"
          f"accuracy: {metrics.accuracy_score(label, pred)}\n"
          f"precision: {metrics.precision_score(label, pred)}\n"
          f"recall: {metrics.recall_score(label, pred, zero_division=1)}\n"
          f"F-1 score: {metrics.f1_score(label, pred)}\n"
          f"confusion matrix: \n{metrics.confusion_matrix(label, pred)}\n"
          f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}\n")


def train_model(x, y, filename_suffix: str, epoch: int = 0):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training...")
    try:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} gnb")
        gnb.fit(x, y)
        with open(os.path.join(save_model_path, f"model_tfidf_gnb_{filename_suffix}_{str(epoch)}.pickle"), "wb") as fp:
            pickle.dump(gnb, fp)
    except:
        print("gnb train error!\n")

    try:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} mnb")
        mnb.fit(x, y)
        with open(os.path.join(save_model_path, f"model_tfidf_mnb_{filename_suffix}_{str(epoch)}.pickle"), "wb") as fp:
            pickle.dump(mnb, fp)
    except:
        print("mnb train error!\n")

    try:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} random forest")
        random_forest.fit(x, y)
        with open(os.path.join(save_model_path, f"model_tfidf_random_forest_{filename_suffix}_{str(epoch)}.pickle"), "wb") as fp:
            pickle.dump(random_forest, fp)
    except:
        print("randon_forest train error!\n")

    try:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} logistic regression")
        logistic_regression.fit(x, y)
        with open(os.path.join(save_model_path, f"model_tfidf_logistic_regression_{filename_suffix}_{str(epoch)}.pickle"),
                  "wb") as fp:
            pickle.dump(logistic_regression, fp)
    except:
        print("logistic_regression train error!\n")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training finished")


def test_model(x, y):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Testing...")
    try:
        random_forest_pred = random_forest.predict(x)
        calculate_metric("random_forest", y, random_forest_pred)
    except:
        print("random_forest predict error!\n")

    try:
        gnb_pred = gnb.predict(x)
        calculate_metric("gnb", y, gnb_pred)
    except:
        print("gnb predict error!\n")

    try:
        mnb_pred = mnb.predict(x)
        calculate_metric("mnb", y, mnb_pred)
    except:
        print("mnb predict error!\n")

    try:
        logistic_regression_pred = logistic_regression.predict(x)
        calculate_metric("logistic_regression", y, logistic_regression_pred)
    except:
        print("logistic_regression predict error!\n")
