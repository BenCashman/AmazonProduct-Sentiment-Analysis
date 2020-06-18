# Sentiment analysis for Amazon products
## (Ben Cashman, June 17, 2020)
### Table of Contents
1. Why this Software is needed

   * Businesses have difficulty measuring the success of a product at          launch.

   * There is a high saturation of useless Amazon product reviews.

   * Not all users rate products with the same weighting, regarding rating scores.  
2. Environment Setup
3. System Requirements
4. Model Training and Data Exploration
5. Running Streamlit
6. Docker
7. Deployment

### Why is this software needed?
There are hundreds of thousands of products on the Amazon marketplace,  many of which have hundreds if not thousands of reviews. Often the most informative metric to a product owner is the user review rating. This however means a user is missing out on valuable and rich information in the written user reviews.

One solution might be to hire someone to read these reviews however this generates significant overhead. As such a natural language processing algorithm utilized for sentiment analysis will provide a user with an overall sentiment score which can then be used to assess a product's performance in the marketplace saving time and money in the long term.  
### Environment Setup
To be able to run locally clone and download this repo:

Git clone https://github.com/BenCashman/AmazonProduct-Sentiment-Analysis
cd AmazonProduct-Sentiment-Analysis

### System Requirements

The following packages and dependencies are required to run this app.

```
streamlit==0.60.0

pandas==0.25.1

numpy==1.17.2

torch==1.5.0

transformers==2.11.0

seaborn==0.10.1

scikit-learn==0.23.1  

scipy==1.4.1

gdown==3.11.1
```

### Installation:
Run the following command to install the necessary dependencies.

`pip install -r requirements.txt`
### Model Training and Exploration
To be able to reproduce my work with data exploration, model training, and model evaluation you can navigate to the AMZ_Products_review.ipynb notebook in this repo and click the open in colab button.
## Running Streamlit
For a local build:  
streamlit run app.py

### Streamlit in Docker
Docker build:  
docker build -t streamlit:v4 -f Dockerfile .

Docker run:   
docker run -p 8501:8501 streamlit:v4
### Deployment:
