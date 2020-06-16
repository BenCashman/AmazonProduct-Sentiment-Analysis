
#Our import list
import streamlit as st
import numpy as np
import pandas as pd
import torch
import transformers
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
#Libraries to pull
from transformers import BertModel
from torch import nn
from torch.nn import functional as F
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

#for added flavor and color
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

#lets set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def main():

    #global variables to be used in script
    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    class_names = ['negative', 'positive']


    class Model(nn.Module):
        def __init__(self,*args,**kwargs):
            super(Model, self).__init__()


    #develop a class for the Sentiment Classifier
    class SentimentClassifier(nn.Module):
        def __init__(self, n_classes):
            super(SentimentClassifier, self).__init__()
            self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
            self.drop = nn.Dropout(p=0.3)
            self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        def forward(self, input_ids, attention_mask):
            _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
                )
            output = self.drop(pooled_output)
            return self.out(output)


    #Generate a title for our webpage
    st.title('Sentiment analysis and product reviews.')
    #createing a sidebar for our webpage
    st.sidebar.title("Sentiment Analysis Web App")
    #little comment for our sidebar section
    st.sidebar.markdown("😃Is your review positive or negative?😞")

    #Here we will load the data into a cache to prevent repeated work)
    def load_data():
        #Function to pull in data from our Amazon s3 Bucket
        data = pd.read_csv('https://amazonproductdata.s3-us-west-1.amazonaws.com/train.csv')
        return data

    #let's ingest our raw data here
    raw_data = load_data()
    #let's also use a smaller subset to work with to make things a bit more light weight
    df = raw_data.sample(2000)
    #let's also remove the null values as there are very few in out data set
    df.dropna(inplace=True)

    #A function for loading models incase we include other models later
    def load_model(filepath):
        model = SentimentClassifier(len(class_names))
        device = torch.device('cpu')
        model.load_state_dict(torch.load(filepath, map_location=device))
        return model

    #conisder loading this into text box classifica.ion funtion
    model = load_model('./model/BERT_trained_model')

    #here we have the ability to plot data metrics
    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)

    def BERT_inference(review_text):
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            #Now we must encode the use text
            encoded_review = tokenizer.encode_plus(
            review_text,
            max_length=300,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            )

            input_ids = encoded_review['input_ids'].to(device)
            attention_mask = encoded_review['attention_mask'].to(device)

            output = model(input_ids, attention_mask)
            _, prediction = torch.max(output, dim=1)

            st.write(f'Review text: {review_text}')
            st.write(f'Sentiment  : {class_names[prediction]}')

    #sidebar options to add more rich features to our app
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Amazon Review Sentiment Analysis. (Polarity Classification)")
        st.table(df)
    #Generating a textbox for user input
    if st.sidebar.checkbox("Input text for inference", False):
        st.subheader("Amazon Review Dataset for Sentiment Analysis. (Inference Demonstration.)")
        user_input = st.text_area("Please provide a review here.")
        if user_input:
            #Let's process the users input
            print(user_input)
            BERT_inference(user_input)



if __name__ == '__main__':
    main()
