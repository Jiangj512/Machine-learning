from utils import load_corpus, stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
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


vectorizer = CountVectorizer(token_pattern='\[?\w+\]?',
                             stop_words=stopwords,
                             max_features=2000)
X_train = vectorizer.fit_transform(df_train["words"])
y_train = df_train["label"]

X_test = vectorizer.transform(df_test["words"])
y_test = df_test["label"]

### 训练模型&测试

param = {
    'booster': 'gbtree',
    'max_depth': 6,
    'scale_pos_weight': 0.5,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'eval_metric': 'error',
    'eta': 0.3,
    'nthread': 10,
}
dmatrix = xgb.DMatrix(X_train, label=y_train)
model = xgb.train(param, dmatrix, num_boost_round=200)

# 在测试集上用模型预测结果
dmatrix = xgb.DMatrix(X_test)
y_pred = model.predict(dmatrix)

# 测试集效果检验

auc_score = metrics.roc_auc_score(y_test, y_pred)  # 先计算AUC
y_pred = list(map(lambda x: 1 if x > 0.5 else 0, y_pred))  # 二值化
print(metrics.classification_report(y_test, y_pred))
print("准确率:", metrics.accuracy_score(y_test, y_pred))
print("AUC:", auc_score)

### 手动输入句子，判断情感倾向（1正/0负）

strs = ["哈哈哈哈哈笑死我了", "我也是有脾气的!"]
words = [processing(s) for s in strs]
vec = vectorizer.transform(words)
dmatrix = xgb.DMatrix(vec)

output = model.predict(dmatrix)
output
