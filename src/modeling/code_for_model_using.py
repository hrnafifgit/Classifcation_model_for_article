import joblib
import pandas as pd

# تحميل النموذج والـTF-IDF والـSelector
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
selector = joblib.load("feature_selector.pkl")

# بيانات جديدة
new_texts = pd.Series([


])

# تحويل النصوص
features = vectorizer.transform(new_texts)
features_selected = selector.transform(features)

# التنبؤ
predictions = model.predict(features_selected)
print(predictions)
