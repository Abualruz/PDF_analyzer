import numpy as np 
import pandas as pd
import PyPDF2
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('tagsets')
from transformers import BertForQuestionAnswering
pdf_file = open('Lionel Andrés Messi.pdf','rb')
read_pdf = PyPDF2.PdfFileReader(pdf_file)
no_of_pages=read_pdf.getNumPages()
no_of_pages
pdf_text=[]
for i in range(no_of_pages):
    text=(read_pdf.getPage(i).extractText())
    text=re.sub(r'\n',' ',text)
    pdf_text.append(text)
    #print(pdf_text)
text= "\n".join (pdf_text)
print (text)
print('\n')
print('Length of pdf_text List: {}'.format(len(pdf_text)))
Questions=[
    "who inhabited northern Colombia in the isolated Andes ?",   
    "who expanded their empire on the southwest part of Colombia ?",
    "when did he sign a contract with Paris Saint-Germain ?"
]
model= BertForQuestionAnswering.from_pretrained ("deepset/bert-base-cased-squad2")
from transformers import AutoTokenizer
tokenizer= AutoTokenizer.from_pretrained ("deepset/bert-base-cased-squad2")
tokenizer.encode(Questions[2], truncation=True)
from transformers import pipeline
nlp= pipeline('question-answering', model=model ,tokenizer=tokenizer)
nlp({
    'question': Questions[2],
    'context': text
})
import nltk
nltk.download('punkt')
import sumy.utils
import sumy.nlp.stemmers
import sumy.nlp.tokenizers
import sumy.summarizers.lsa
import sumy.parsers.plaintext
import sumy.summarizers.text_rank

LANGUAGE= "english"
SENTENCE_COUNT= 2

tokenizer= sumy.nlp.tokenizers.Tokenizer(LANGUAGE)
parser= sumy.parsers.plaintext.PlaintextParser.from_string(text,tokenizer)
stemmer = sumy.nlp.stemmers.Stemmer(LANGUAGE)
summarizer= sumy.summarizers.text_rank.TextRankSummarizer(stemmer)
summarizer.stop_words = sumy.utils.get_stop_words(LANGUAGE)

summary_list= [str(s) for s in summarizer(parser.document, SENTENCE_COUNT)]
summary= "\n".join(summary_list)
print(summary)
import tensorflow as tf
import torch as t
import transformers
from transformers import pipeline
classifier = pipeline('zero-shot-classification')
input_text= pdf_text
candidate_label =['Sports','Football','Food',]
classifier (input_text,candidate_label)
highest = max(candidate_label)
print("Category type is : " , highest)
import fitz
file = fitz.open("pictures with text.pdf")
import os, sys
# iterate over PDF pages and find all the images location inside the pdf
import io
from PIL import Image

page = file[i]

for i in range(len(file)):
    # get the page itself
    page = file[i]
    #image_list = page.getImageList()
    image_list= page.get_images()
    # printing number of images found in this page
    if image_list:
        print(f"[+] Found a total of {len(image_list)} images in page {i}")
    else:
        print("[!] No images found on page", i)
    for image_index, img in enumerate(page.get_images(), start=1):
        # get the XREF of the image
        xref = img[0]
        # extract the image bytes
        base_image = file.extract_image(xref)
        image_bytes = base_image["image"]
        # get the image extension
        image_ext = base_image["ext"]
        # load it to PIL
        image = Image.open(io.BytesIO(image_bytes))
        # save it to local disk
        image.save(open(f"image{i+1}_{image_index}.{image_ext}", "wb"))
        import os

directory = r'C:\Users\habua\Desktop\Project VSC'
for filename in os.listdir(directory):
    if filename.endswith(".jpeg") or filename.endswith(".png"):
        print(os.path.join(directory, filename))
    else:
        continue
import easyocr
reader= easyocr.Reader(['en'])
results= reader.readtext('thumbnail_Image-1.png')
print (results)
import os
path_of_the_directory = directory
ext = ('jfif','png',)
for files in os.listdir(path_of_the_directory):
    if files.endswith(ext):
        print(files)  
        results= reader.readtext(files)
        dialogue= ''
        for result in results:
            dialogue+= result[1]+''
            #print (dialogue)
        print (dialogue)
    else:
        continue
import camelot
import camelot.io as camelot
# PDF file to extract tables from
file = "table_sample.pdf"
tables = camelot.read_pdf("Food Calories List sample.pdf",pages='1-end',flavor="stream")
print (tables)
#Printing all the tables inside the pdf file as a concatinated list

for x in range(len(tables)):
    print (tables[x].df),
tables.export("camelot_tables.csv", f = "csv") 
import torch
from transformers import pipeline
import pandas as pd

#tqa = pipeline(task="table-question-answering", 
 #              model="google/tapas-base-finetuned-wtq")
#table = pd.read_csv("camelot_tables-page-8-table-1.csv")
#table = table.astype(str)               
#table

table = pd.read_csv("camelot_tables-page-8-table-1.csv")
table = table.astype(str)
query = "what is the Calories mount per piece for olives?"
from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd
tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base-finetuned-wikisql")
model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-base-finetuned-wikisql")
encoding = tokenizer(table=table, query=query, return_tensors="pt")
outputs = model.generate(**encoding)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))




