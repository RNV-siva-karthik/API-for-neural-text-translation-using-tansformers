# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 08:53:12 2023

@author: R.N.V Siva Karthik
"""
import uvicorn
from fastapi import FastAPI
import torch
from transformers import MarianMTModel, MarianTokenizer
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
app = FastAPI()
@app.get("/{input_text}")
async def root(input_text:str):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    translation = model.generate(**inputs, max_length=1500, num_beams=4, length_penalty=2.0, no_repeat_ngram_size=3, early_stopping=False)
    translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)
    return {'text':translated_text}
if __name__ == "__main__":
   uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
