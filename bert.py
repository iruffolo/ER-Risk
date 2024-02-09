import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline

from load_data.database import DataLoader


tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# x = tokenizer.encode(text, return_tensors='pt')
# print(x)
# print(tokenizer.convert_ids_to_tokens(x[0].tolist()))
# print(model)


if __name__ == "__main__":

    dataloader = DataLoader("data/sqlite/db.sqlite")

    print(dataloader.get_data())
