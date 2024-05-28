
import streamlit as st
import pandas as pd
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import streamlit as st
import nltk

model_name = 'morca/t5-ft'

def load_model():
    print("loading model...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    nltk.download('punkt')
    print("Model loaded!")
    return tokenizer, model


st_model_load = st.text("model is loading")
tokenizer, model = load_model()
st.success('model loaded!')
st_model_load.text("")

def preprocess_text(text):
    text = text.replace("?", '"').replace("?", "'")
    text = text.strip().replace("\n", " ")

    if not text.endswith("."):
        text = text + "."

    t5_prepared_Text = "summarize: " + text

    return t5_prepared_Text

def generate_summary_t5(text):
    text = preprocess_text(text)
    inputs = tokenizer.encode(text, max_length=1024, truncation='longest_first', return_tensors="pt")

    outputs = model.generate(inputs,
                             num_beams=4,
                             no_repeat_ngram_size=3,
                             min_length=60,
                             max_length=128,
                             length_penalty=2.0,
                             temperature=0.5)
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    summary = nltk.sent_tokenize(decoded_output.strip())
    summary = '\n'.join(summary)
    if len(summary) > 0:
        with st.container():
            st.subheader("Generated summary")
            st.markdown(st.session_state.summary)

def news(key, new):
    st.subheader(str(key + 1) + ". " + new[1])
    st.button("Generate summary for story " + str(key + 1) + " with T5", on_click=generate_summary_t5(new[2]))
    st.button("Generate summary for story " + str(key + 1) + " with Pegasus")


st.header('News Summarizer')
df = pd.read_excel('news.xlsx')
new = df.loc[0, :].values.tolist()
news(0, new)


# AND in st.sidebar!
with st.sidebar:
    st.subheader('Türkçe için...')
    st.button('we will be there...')
