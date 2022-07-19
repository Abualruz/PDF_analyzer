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

pdf_text=[]
for i in range(no_of_pages):
    text=(read_pdf.getPage(i).extractText())
    text=re.sub(r'\n',' ',text)
    pdf_text.append(text)
    #print(pdf_text)
text= "\n".join (pdf_text)

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