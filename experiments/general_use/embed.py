import pandas as pd
import os
from pingkit import embed_dataset



train_df = pd.read_csv('simpleqa_train.csv')
test_df= pd.read_csv('simpleqa_test.csv')

set_base = "/home/berkowitzj2/scratch/simpleqa"
model="google/gemma-2-9b-it"
model_base=model.split('/')[-1]
train_dir=set_base+"_train"
test_dir=set_base+"_test"
text_col='prompt'


embed_dataset(train_df, input_col=text_col, output_dir=train_dir, model_name=model, device="auto",pooling="last",filter_non_text=True, parts=["rs"])

embed_dataset(test_df, input_col=text_col, output_dir=test_dir, model_name=model, device="auto",pooling="last",filter_non_text=True, parts=["rs"])

