import json
from transformers import BertJapaneseTokenizer, BertModel
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

with open('input.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# レビューをトークナイズしてエンベディングを取得
reviews = [d['review'] for d in data]
labels = [d['label'] for d in data]

# BERT日本語版のトークナイザーとモデルを読み込む
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v3')
bert_model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-v3')

# レビューをトークナイズしてエンベディングを取得
inputs = tokenizer(reviews, return_tensors='pt', padding=True, truncation=True, max_length=512)
with torch.no_grad():
    outputs = bert_model(**inputs)

# 最後の隠れ層のベクトルを平均して特徴量とする
embeddings = outputs.last_hidden_state.mean(dim=1).numpy()

# データセットをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# データのスケーリング
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ロジスティック回帰モデルを使って学習
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# モデルの評価
accuracy = model.score(X_test_scaled, y_test)
print(f'Accuracy: {accuracy}')

# 学習済みモデルとトークナイザーの保存
joblib.dump(model, 'logistic_regression_model.pkl')
torch.save(bert_model.state_dict(), 'bert_model.pt')
tokenizer.save_pretrained('.')
