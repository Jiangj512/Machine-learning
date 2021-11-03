from utils import load_corpus, stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from utils import processing

TRAIN_PATH = "./data/weibo2018/train.txt"
TEST_PATH = "./data/weibo2018/test.txt"

# 分别加载训练集和测试集
train_data = load_corpus(TRAIN_PATH)
test_data = load_corpus(TEST_PATH)

df_train = pd.DataFrame(train_data, columns=["words", "label"])
df_test = pd.DataFrame(test_data, columns=["words", "label"])
df_train.head()

### 特征编码（词袋模型）

vectorizer = CountVectorizer(token_pattern='\[?\w+\]?',
                             stop_words=stopwords)
X_train = vectorizer.fit_transform(df_train["words"])
y_train = df_train["label"]

X_test = vectorizer.transform(df_test["words"])
y_test = df_test["label"]

clf = MultinomialNB()
clf.fit(X_train, y_train)

# 在测试集上用模型预测结果
y_pred = clf.predict(X_test)

# 测试集效果检验

print(metrics.classification_report(y_test, y_pred))
print("准确率:", metrics.accuracy_score(y_test, y_pred))

### 手动输入句子，判断情感倾向

strs = ["终于收获一个最好消息", "哭了, 今天怎么这么倒霉"]
words = [processing(s) for s in strs]
vec = vectorizer.transform(words)

output = clf.predict(vec)
print(output)
