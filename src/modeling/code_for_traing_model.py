# ============================================
# 🧠 مشروع تصنيف نصوص باستخدام الذكاء الاصطناعي
# ============================================

# 🩵 [1] تحميل وتجهيز البيانات
import pandas as pd
data = pd.read_csv(r'cleaned_no_stopwords.csv')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data['content'], data['category'], test_size=0.3, random_state=42
)

# 🩵 [2] تحويل النصوص إلى ميزات رقمية (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 🩵 [3] اختيار الميزات الأهم (Feature Selection)
from sklearn.feature_selection import SelectKBest, chi2
# chi2 يقيس مدى ارتباط كل ميزة بالفئة
selector = SelectKBest(chi2, k=250)
X_train_sel = selector.fit_transform(X_train_tfidf, y_train)
X_test_sel = selector.transform(X_test_tfidf)

# 🩵 [4] بناء النموذج وتدريبه
from sklearn.linear_model import LogisticRegression
# يعطي احتمالات ثنائية وليس مجرد صح أو خطأ
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_sel, y_train)

# 🩵 [5] تقييم النموذج
from sklearn.metrics import classification_report
predictions = model.predict(X_test_sel)
print(classification_report(y_test, predictions))

# 🩵 [6] حفظ النموذج والأدوات
import joblib
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(selector, "selector.pkl")
print("✅ تم!")
