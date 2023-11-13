import os
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt

#nltk.download('punkt') #obligatory
#nltk.download('stopwords')

count = 0

def preprocess_resume_str(txt):
    global count
    print(count)
    count += 1
    # convert all characters in the string to lower case
    txt = txt.lower()
    # remove non-english characters, punctuation and numbers
    txt = re.sub('[^a-zA-Z]', ' ', txt)
    txt = re.sub('http\S+\s*', ' ', txt)  # remove URLs
    txt = re.sub('RT|cc', ' ', txt)  # remove RT and cc
    txt = re.sub('#\S+', '', txt)  # remove hashtags
    txt = re.sub('@\S+', '  ', txt)  # remove mentions
    txt = re.sub('\s+', ' ', txt)  # remove extra whitespace
    # tokenize word
    txt = nltk.tokenize.word_tokenize(txt)
    # remove stop words
    txt = [w for w in txt if not w in nltk.corpus.stopwords.words('english')]

    return txt


df_filename = "data/resume_tokenized.csv"
if (not os.path.isfile(df_filename)):
    df = pd.read_csv("data/resume.csv")
    df = df.drop("Resume_html", axis=1)
    df["Resume_str"] = df["Resume_str"].apply(lambda x: preprocess_resume_str(x))
    df.to_csv(df_filename, index=False)
else:
    df = pd.read_csv(df_filename)

print(df.head())
print(df.columns.values)
#print(df["Resume_str"][0])
df['Category'].value_counts().sort_index().plot(kind='bar', figsize=(12, 6))
plt.show()



