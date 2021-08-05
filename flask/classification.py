import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

topic_dict = pd.read_csv('topic_dict.csv')

model = AutoModelForSequenceClassification.from_pretrained('classification_model', num_labels=7)
tokenizer = AutoTokenizer.from_pretrained('classification_model', local_files_only=True)

mapping = {}
for i in range(7):
  mapping[i] = topic_dict['topic'][i]

def text_clear(text):
  text = text.replace("\\", "").replace("n"," ")
  text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》→·]', ' ', text)
  text = re.sub(r'[0-9]+' , '' ,text)
  return text

def get_prediction(text):
    text = text_clear(text)
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    return mapping[probs.argmax().item()]