import camelot
import camelot.io as camelot
# PDF file to extract tables from
file = "table_sample.pdf"
tables = camelot.read_pdf("Food Calories List sample.pdf",pages='1-end',flavor="stream")
#print (tables)
#Printing all the tables inside the pdf file as a concatinated list

#for x in range(len(tables)):
   # print (tables[x].df),
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