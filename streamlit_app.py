import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Application Title
st.title("Week 6 App")

# BU ID
bu_id = 59385965  

# Load the pre-trained GPT-2 model
model_name = "gpt2"
gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# User Input Function
def generate_text(prompt, max_length, creativity):
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors="pt")
    generated = gpt2_model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=creativity
    )
    return gpt2_tokenizer.decode(generated[0], skip_special_tokens=True)

# User prompt and number of tokens
prompt_input = st.text_input("Enter your text prompt:", "")
token_limit = st.number_input("Select the number of tokens for output:", min_value=1, max_value=100, value=50)

# Generate responses
if st.button("Generate"):
    if prompt_input:
        # Generate creative response
        creative_response = generate_text(prompt_input, token_limit, creativity=1.9)
        st.subheader("Creative Response:")
        st.write(creative_response)

        # Generate predictable response
        predictable_response = generate_text(prompt_input, token_limit, creativity=0.6)
        st.subheader("Predictable Response:")
        st.write(predictable_response)

        # Display the user ID for tracking purposes
        st.write(f"User ID: {bu_id}")
    else:
        st.warning("Please enter a prompt to generate responses.")
