FROM python:3.8

WORKDIR /streamlit

COPY . /streamlit
RUN apt-get update -y && apt-get install -y libgl1-mesa-dev
RUN pip install -r requirements.txt

CMD streamlit run app.py