import streamlit as st
import numpy as np
import pandas as pd
import torch
import transformers



def main():

    #Generate a title for our webpage
    st.title('Sentiment analysis and product reviews.')
    #createing a sidebar for our webpage
    st.sidebar.title("Sentiment Analysis Web App")
    #little comment for our sidebar section
    st.sidebar.markdown("😃Is your review positive or negative?😞")

    #Here we will load the data into a cache to prevent repeated work
    @st.cache(persist = True)
    #Function to pull in data from our Amazon s3 Bucket
    def load_data():
        data = pd.read_csv('https://amazonproductdata.s3-us-west-1.amazonaws.com/train.csv')
        return data
    df = load_data()

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Amazon Review Dataset for Sentiment Analysis. (Polarity Classification)")
        st.write(df)


if __name__ == '__main__':
    main()
