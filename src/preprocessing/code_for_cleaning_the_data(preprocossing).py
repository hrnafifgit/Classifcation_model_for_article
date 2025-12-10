import pandas as pd, re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean(t):
    if pd.isna(t) or not t: return ""
    c = re.sub(r'[^a-zA-Z\s]', '', str(t))
    c = re.sub(r'\s+', ' ', c).strip()
    return ' '.join([w for w in c.split() if w.lower() not in stop_words])

df = pd.read_csv(r"databefor_cleaning.csv")
df['category'], df['content'] = df['category'].apply(clean), df['content'].apply(clean)
df = df[(df['category'] != '') & (df['content'] != '')]
df[['category', 'content']].to_csv(r"dataafter_cleaning.csv", index=False)

print(f"✅ {len(df)} سجل")