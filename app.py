import streamlit as st
import numpy as np
import pandas as pd
import torch
import transformers

def main():

    st.title('Sentiment analysis and product reviews')
    st.sidebar.title("Sentiment Analysis Web App")
    st.sidebar.markdown("😃Is your review positive or negative?😞")

if __name__ == '__main__':
    main()
