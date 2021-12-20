import preprocessing
import pandas as pd
from copy import deepcopy
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
import os
import json
from sklearn.feature_selection import SelectKBest, chi2

# with open("./datasets/train_E6oV3lV.csv", "r") as fp:
#     with open("./datasets/hate.csv", "a") as fp1:
#         with open("./datasets/non_hate.csv", "a") as fp2:
#             a = fp.readline()
#             for line in fp:
#                 line1 = line.split(",")
#                 if line1[1] == "1":
#                     fp1.write(",".join(line1))
#                 else:
#                     fp2.write(",".join(line1))

# hate = pd.read_csv("./datasets/hate.csv", delimiter=",")
# non_hate = pd.read_csv("./datasets/non_hate.csv", delimiter=",")
#
# pre = preprocessing.Preprocessing("./datasets/hate.csv", filename_suffix="hate")
# pre._get_all_tokens()

hate = []
hate_label = []
for row in pd.read_csv("./datasets/hate.csv", delimiter=",").iterrows():
    hate.append(row[1]["tweet"])
    hate_label.append((row[1]["label"]))
hate_len = len(hate)

non_hate = []
non_hate_label = []
for row in pd.read_csv("./datasets/non_hate.csv", delimiter=",").iterrows():
    non_hate.append(row[1]["tweet"])
    non_hate_label.append((row[1]["label"]))
non_hate_len = len(non_hate)

data = deepcopy(hate)
data.extend(non_hate)
label = deepcopy(hate_label)
label.extend(non_hate_label)

pre = preprocessing.Preprocessing(data, "hate")
all_token_path = os.path.join(preprocessing.save_preprocessed_data_path, f"all_token_hate.json")
with open(all_token_path, "r") as fp:
    all_tokens = json.loads(fp.read())
tfidf = pre.get_tfidf()
label = np.array(label)
tfidf = sklearn.preprocessing.minmax_scale(tfidf)
x = pre.feature_selection(tfidf, label, k=1000)
x_train, x_test, y_train, y_test = train_test_split(x, label, train_size=0.85)
preprocessing.train_model(x_train, y_train, "selected_1000")
preprocessing.test_model(x_test, y_test)

# a = np.array([[1, 2, 3], [4, 5, 6],[9, 100, 2]])
# b = np.array([1, 0, 1])
# select = SelectKBest(chi2, k=2)
# x = select.fit_transform(a, b)
# y = select.get_support(indices=True)
# print(y, type(y))



# epoch = 10
# start = 0
# end = hate_len
# for i in range(epoch):
#     data = deepcopy(hate)
#     label = deepcopy(hate_label)
#     if end <= non_hate_len:
#         data.extend(non_hate[start: end: 1])
#         label.extend(non_hate_label[start: end: 1])
#         start += hate_len
#         end += hate_len
#     else:
#         data.extend(non_hate[start:: 1])
#         data.extend(non_hate[0: end - non_hate_len: 1])
#         label.extend(non_hate_label[start:: 1])
#         label.extend(non_hate_label[0: end - non_hate_len: 1])
#         start = end - non_hate_len
#         end = start + hate_len
#
#     pre = preprocessing.Preprocessing(data, "hate")
#     tfidf = pre.get_tfidf()
#     label = np.array(label)
#     select = SelectKBest(chi2, k=1000)
#     x = select.fit_transform(tfidf, label)
#     x_train, x_test, y_train, y_test = train_test_split(tfidf, label)
#     print(f"training epoch {i}")
#     preprocessing.train_model(x_train, y_train, "hate", epoch=i)
#     preprocessing.test_model(x_test, y_test)
