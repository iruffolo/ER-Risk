import torch
from transformers import AutoTokenizer, BertConfig, BertModel


def _get_embeddings(data, tokenizer, device):
    """
    Tokenize the data using the provided tokenizer
    """

    # Create encoding from tokenizer
    encoding = tokenizer.batch_encode_plus(
        data.values.tolist(),
        max_length=512,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        return_token_type_ids=True,
        truncation=True,
        padding="longest",
        return_attention_mask=True,
    )
    # convert lists to tensors
    seq = torch.tensor(encoding["input_ids"]).to(device)
    mask = torch.tensor(encoding["attention_mask"]).to(device)
    token_ids = torch.tensor(encoding["token_type_ids"]).to(device)

    return seq, mask, token_ids


def get_bert_embeddings(
    data,
    model_name="emilyalsentzer/Bio_ClinicalBERT",
    batch_size=128,
):
    """
    Get embeddings from BERT model
    """

    # Set device GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load BERT model
    bert = BertModel.from_pretrained(model_name).to(device)

    # Load Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the data
    seq, mask, token_ids = _get_embeddings(data, tokenizer, device)

    last_hidden_state = []
    pooler_output = []
    with torch.no_grad():

        # Get predictions in batches
        for i in range(0, len(seq), batch_size):
            print(
                f"Processing inputs {i} of {len(seq//batch_size)} for Bert embeddings."
            )
            bert_output = bert(
                input_ids=seq[i : i + batch_size],
                attention_mask=mask[i : i + batch_size],
                token_type_ids=token_ids[i : i + batch_size],
            )

            # last_hidden_state.append(bert_output.last_hidden_state.cpu())
            pooler_output.append(bert_output.pooler_output.cpu())

    # last_hidden_state = torch.cat(last_hidden_state, dim=0)
    pooler_output = torch.cat(pooler_output, dim=0)

    return {"last_hidden_state": last_hidden_state, "pooler_output": pooler_output}
