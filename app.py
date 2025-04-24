import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

@st.cache_resource
def load_model():
    model = GPT2LMHeadModel.from_pretrained("habeebapisec/physics-gpt-qa")
    tokenizer = GPT2Tokenizer.from_pretrained("habeebapisec/physics-gpt-qa")
    return model, tokenizer

model, tokenizer = load_model()

st.title("🤖 Physics GPT Assistant (Hugging Face Model)")

user_input = st.text_input("Ask your Physics Question (start with 'Q:')", "Q: What is potential energy?\nA:")

if st.button("Generate Answer"):
    inputs = tokenizer.encode(user_input, return_tensors="pt")
    attention_mask = torch.ones_like(inputs)

    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=150,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.success(result)
