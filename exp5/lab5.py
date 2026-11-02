import pandas as pd 
import numpy as np 

from sklearn.datasets import fetc_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,classification_report
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction import CountVectorizer

categories = [
    "comp.graphics",
    "rec.sport.baseball",
    "sci.space",
    "talk.politics.misc"
]

data = fetc_20newsgroups(
    subset = "all",
    categories = categories,
    remove = ("headers","footers","quotes")
)

x = data.data
y = data.target

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size = 0.2,
    random_state = 42
)

count = CountVectorizer(
    stop_words = "english",
    max_features = 5000
)

x_train_count = count.fit_transform(x_train)
x_test_count = count.transform(x_test)

binary = CountVectorizer(
    stop_words = "english",
    max_features = 5000
    binary = True 
)

x_train_binary = binary.fit_transform(x_train)
x_test_binary = binary.transfrom(x_test)

mnb = MultinomialNB()
mnb.fit(x_train_count,y_train)
y_pred_mnb = mnb.predict(x_test_count)

print("MultinomialNB")
print("Accuracy:",accuracy_score(y_test,y_pred_mnb))
print("f1:",f1_score(y_test,y_pred_mnb,average="weighted"))

bnb = BernoulliNB()
bnb.fit(x_train_binary,y_train)
y_pred_bnb = bnb.predict(x_test_binary)

print("BernoulliNB")
print("Accuracy:",accuracy_score(y_test,y_pred_bnb))
print("f1:",f1_score(y_test,y_pred_bnb,average="weighted"))

result = pd.DataFrame({
    "Model":["MultinomialNB","BernoulliNB"],
    "Accuracy":[accuracy_mnb,accuracy_bnb],
    "f1":[f1_mnb,f1_bnb]
})

print("\ncomparison")
print(result)

print("MultinomialNB report")
print(classification_report(y_test,y_pred_mnb, target_names = data.target_names))

print("BernoulliNB report")
print(classification_report(y_test,y_pred_bnb, target_names = data.target_names))



