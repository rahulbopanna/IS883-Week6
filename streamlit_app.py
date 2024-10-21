import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

### Title of app on Streamlit
st.title("Rahul Week6 HW")

### Enter BUID
BUID = 59385965

### OpenAI Secret Key
my_secret_key = st.secrets['Week6']

### Load GPT2 from hugging face
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

### Prompt input on streamlit app by user
prompt = st.text_input("What is your prompt today?", "Enter your prompt here")
max_tokens = st.number_input("Enter the number of tokens for the response", min_value=1, max_value=100, value=50)

### Engineer the types of responses based on temperature
if st.button("Generate Response"):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs_high = model.generate(
        inputs['input_ids'], 
        max_length=max_tokens,
        do_sample=True, 
        temperature=1.9
    )
    highly_creative_response = tokenizer.decode(outputs_high[0], skip_special_tokens=True)
    st.write("Highly Creative Response:")
    st.write(highly_creative_response)

    outputs_low = model.generate(
        inputs['input_ids'], 
        max_length=max_tokens, 
        do_sample=True, 
        temperature=0.5
     )
    highly_predictable_response = tokenizer.decode(outputs_low[0], skip_special_tokens=True)
    st.write("Highly Predictable Response:")
    st.write(highly_predictable_response)
