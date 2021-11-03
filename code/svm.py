from utils import load_corpus, stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
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

### 特征编码（Tf-Idf模型）

vectorizer = TfidfVectorizer(token_pattern='\[?\w+\]?',
                             stop_words=stopwords)
X_train = vectorizer.fit_transform(df_train["words"])
y_train = df_train["label"]

X_test = vectorizer.transform(df_test["words"])
y_test = df_test["label"]

### 训练模型&测试

clf = svm.SVC()
clf.fit(X_train, y_train)

# 在测试集上用模型预测结果
y_pred = clf.predict(X_test)

# 测试集效果检验

print(metrics.classification_report(y_test, y_pred))
print("准确率:", metrics.accuracy_score(y_test, y_pred))

### 手动输入句子，判断情感倾向

strs = ["只要流过的汗与泪都能化作往后的明亮，就值得你为自己喝彩", "烦死了！为什么周末还要加班[愤怒]"]
words = [processing(s) for s in strs]
vec = vectorizer.transform(words)

output = clf.predict(vec)
output
