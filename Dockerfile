# Dockerfile for building streamlit app

#Install python 3.7
FROM python:3.7

#copy local files into container
COPY app.py /tmp/
COPY requirements.txt /tmp/
COPY assets/ /tmp/assets

# change directory
WORKDIR /tmp

# install dependencies
RUN apt-get update && apt-get install -y vim g++
RUN pip install -r requirements.txt


#COPY data /tmp/data
# .streamlit for something to do with making enableCORS=False
COPY .streamlit .streamlit

#ENV PORT 8080



# run commands
CMD ["streamlit", "run", "app.py"]
