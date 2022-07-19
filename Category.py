import numpy as np 
import pandas as pd
import PyPDF2
import re
import nltk
import tensorflow as tf
import torch as t
import transformers
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

nltk.download('tagsets')

pdf_file = open('Lionel Andrés Messi.pdf','rb')
read_pdf = PyPDF2.PdfFileReader(pdf_file)
no_of_pages=read_pdf.getNumPages()
no_of_pages
pdf_text=[]

from transformers import pipeline
classifier = pipeline('zero-shot-classification')
input_text= pdf_text
candidate_label =['Sports','Football','Food',]
classifier (input_text,candidate_label)
highest = max(candidate_label)
print("Category type is : " , highest)