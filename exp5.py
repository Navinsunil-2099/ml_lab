import pandas as pd 
from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,f1_score,classification_report

categories = [
    'comp.graphics',
    'rec.sport.baseball'
    'sci.space',
    'talk.politics.misc'
]

newsgroups = fetch_20newsgroups(
    subset='all',
    categories = categories,
    remove = ('headers','footers','quotes')
)

x_text = newsgroups.data 
y = newsgroups.target

x_train_text, x_test_text, y_train, y_test = train_test_split(
    x_text,
    y,
    test_size = 0.2,
    random_state = 42
)

count_vectorizer = CountVectorizer(
    stop_words = 'english',
    max_features = 5000
)

x_train_count = count_vectorizer.fit_transform(x_train_text)

x_test_count = count_vectorizer.transform(x_test_text)

binary_vectorizer = CountVectorizer(
    stop_words = 'english',
    max_features = 5000,
    binary = True 
)

x_train_binary = binary_vectorizer.fit_transform(x_train_text)

x_test_binary = binary_vectorizer.transform(x_test_text)

# creating the actual model mnb 

mnb = MultinomialNB()
#training the mnb model using input data + input target/labels
mnb.fit(x_train_count,y_train)

y_pred_mnb = mnb.predict(x_test_count)

acc_mnb = accuracy_score(y_test,y_pred_mnb)
f1_mnb = f1_score(y_test, y_pred_mnb, average = 'weighted')

bnb = BernoulliNB()

bnb.fit(x_train_count,y_train)

y_pred_bnb = bnb.predict(x_test_binary)

acc_bnb = accuracy_score(y_test,y_pred_bnb)
f1_bnb = f1_score(y_test,y_pred_bnb,average = 'weighted')

#results
print("MultinomialNB")
print("Accuracy:",acc_mnb)
print("F1_score:",f1_mnb)

print("\BernoulliNB")
print("Accuracy:",acc_bnb)
print("F1_score:",f1_bnb)


results = pd.DataFrame({
    'Model':['MultinomialNB','BernoulliNB'],
    'Accuracy':[acc_mnb,acc_bnb],
    'F1_score':[f1_mnb,f1_bnb]
})

print("\nComparison")
print(results)

print("\MultinomialNB report")

print(classification_report(
    y_test,
    y_pred_mnb,
    target_names = newsgroups.target_names 
))

print("\BernoulliNB report")

print(classification_report(
    y_test,
    y_pred_bnb,
    target_names = newsgroups.target_names
))


