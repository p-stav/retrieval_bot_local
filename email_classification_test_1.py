import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from IPython import display



df = pd.read_csv('20190512_campaign_replies_clean.csv')
# df.head()

# remove nulls
df = df[pd.notnull(df['Last conversation summary'])]
df = df[pd.notnull(df['Status'])]


# create categorical variable for Status
df['status_id'] = df['Status'].factorize()[0]
df.drop_duplicates()

# create focused table
col  =  ['Last conversation summary', 'Status']
foused_df =  df[col]
conversation_to_status = dict(df[['Last conversation summary', 'Status']].values)
status_to_conversation = dict(df[['Status', 'Last conversation summary']].values)

status_id_df = df[['Status', 'status_id']].drop_duplicates().sort_values('status_id')
status_to_id = dict(status_id_df.values)
id_to_status = dict(status_id_df[['status_id', 'Status']].values)


fig = plt.figure(figsize=(8,6))
df.groupby('Status').Email.count().plot.bar(ylim=0)
plt.show()

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df['Last conversation summary']).toarray()
labels = df.Status
features.shape # (10867, 5322) --> this means that 10867 conversations are represented by 5433 features, representing the tfidf score for different unigrams and bigrams


# find the terms most correlated with each
    N = 5
    for status, status_id in sorted(status_to_id.items()):
      features_chi2 = chi2(features, labels == status)
      indices = np.argsort(features_chi2[0])
      feature_names = np.array(tfidf.get_feature_names())[indices]
      unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
      bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
      print("# '{}':".format(status))
      print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:]).encode('utf-8').strip()))
      print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:]).encode('utf-8').strip()))



# FIRST PRINTOUT
"""
#not_interested
  . Most correlated unigrams:
. email
. interested
. thanks
. thank
. sorry
  . Most correlated bigrams:
. thank reaching
. currently office
. thanks reaching
. hi maryn
. thank email

# 'auto_reply':
  . Most correlated unigrams:
. access
. returning
. monday
. office
. return
  . Most correlated bigrams:
. office monday
. immediate assistance
. access email
. limited access
. currently office

# 'auto_reply_referral':
  . Most correlated unigrams:
. brien
. contact
. longer
. com
. maternity
  . Most correlated bigrams:
. currently maternity
. direct email
. office maternity
. leave contact
. maternity leave

# 'do_not_contact':
  . Most correlated unigrams:
. address
. unsubscribe
. longer
. remove
. list
  . Most correlated bigrams:
. list thank
. mailing list
. remove email
. email list
. remove list

# 'handed_off':
  . Most correlated unigrams:
. send
. demo
. chat
. yes
. brochures
  . Most correlated bigrams:
. hi sheila
. send information
. interested learning
. send brochures
. yes interested

# 'handed_off_with_questions':
  . Most correlated unigrams:
. interesting
. scott
. fee
. come
. examples
  . Most correlated bigrams:
. insurance company
. currently process
. hi scott
. hi ali
. hi steve

# 'interested':
  . Most correlated unigrams:
. passed
. marketing
. ali
. middle
. curren
  . Most correlated bigrams:
. hi curren
. great hear
. marketing director
. hi mark
. forward information

# 'not_interested':
  . Most correlated unigrams:
. maryn
. time
. thanks
. thank
. interested
  . Most correlated bigrams:
. need services
. hi maryn
. interested thank
. interested thanks
. interested time

# 'referral':
  . Most correlated unigrams:
. sally
. org
. kathleen
. julia
. com
  . Most correlated bigrams:
. building manager
. contact scott
. hey thanks
. person reach
. com assistance

# 'sent_meeting_invite':
  . Most correlated unigrams:
. 812
. speaking
. curren
. yes
. quote
  . Most correlated bigrams:
. insurance carriers
. information need
. march 25
. yes interested
. yes iâ
"""

X_train, X_test, y_train, y_test = train_test_split(df['Last conversation summary'], df['Status'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)



#### Random Forrest Classifier
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()





# dig into logistic regression
model = LogisticRegression(random_state=0)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=status_id_df.status_id.values, yticklabels=status_id_df.status_id.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()



# show mismatches
for predicted in status_id_df.status_id:
  for actual in status_id_df.status_id:
    if predicted != actual and conf_mat[actual, predicted] >= 10:
      print("'{}' predicted as '{}' : {} examples.".format(id_to_status[actual], id_to_status[predicted], conf_mat[actual, predicted]))
      display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['status_id', 'Last conversation summary']])
      print('')
