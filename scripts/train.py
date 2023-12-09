import torch
import numpy as np
import scipy as sp
import pandas as pd
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import shap


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

df = pd.read_csv("data/resume.csv")
resume_str_0 = df["Resume_str"][0]

inputs = tokenizer(resume_str_0, return_tensors="pt")
print("after tokenizing inputs")

def pred(x):
    print("before tv")
    tv = torch.tensor(
        [
            tokenizer.encode(v, padding="max_length", max_length=500, truncation=True)
            for v in x
        ]
    )
    print("after making tv tensor")
    outputs = model(tv)[0].numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:, 1])  # use one vs rest logit units
    return val

# build an explainer using a token masker
print("before building explainer")
explainer = shap.Explainer(pred, tokenizer)
print("after building explainer")

shap_values = explainer(df["Resume_str"][:10], fixed_context=1)
shap.plots.text(shap_values[3])

