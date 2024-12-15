import json
from transformers import BertJapaneseTokenizer, BertModel
import torch
import joblib

# 学習済みモデルとトークナイザーの読み込み
model = joblib.load('logistic_regression_model.pkl')
bert_model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-v3')
bert_model.load_state_dict(torch.load('bert_model.pt', weights_only=True))
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v3')

# 新しいレビューを入力
new_review = ["会社では同じものを何年も前から使っていましたが、在宅勤務の時はなしで過ごしていたら、手首や肘が痛くなってきたので、自宅用に注文しました。"]

# レビューをトークナイズしてエンベディングを取得
inputs = tokenizer(new_review, return_tensors='pt', padding=True, truncation=True, max_length=512)
print(inputs)
with torch.no_grad():
    outputs = bert_model(**inputs)

# 最後の隠れ層のベクトルを平均して特徴量とする
new_embedding = outputs.last_hidden_state.mean(dim=1).numpy()

# 学習済みのロジスティック回帰モデルを使用して予測確率を取得
prediction_proba = model.predict_proba(new_embedding)
sakura_review_probability = prediction_proba[0][1] * 100  # 1の確率をパーセンテージに変換

print(f'サクラレビュー度: {sakura_review_probability:.2f}%')
