import torch
from transformers import DistilBertModel, DistilBertTokenizer, CLIPTokenizer, CLIPTextModel


def get_tokenizer_and_model(model_type, device, eval_mode=True):
    assert model_type in ('bert', 'clip'), "Text model can only be one of clip or bert"
    if model_type == 'bert':
        text_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        text_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    else:
        text_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
        text_model = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch16').to(device)
    if eval_mode:
        text_model.eval()
    return text_tokenizer, text_model
    

def get_text_representation(text, text_tokenizer, text_model, device,
                            truncation=True,
                            padding='max_length',
                            max_length=77):
    token_output = text_tokenizer(text,
                                  truncation=truncation,
                                  padding=padding,
                                  return_attention_mask=True,
                                  max_length=max_length)
    indexed_tokens = token_output['input_ids']
    att_masks = token_output['attention_mask']
    tokens_tensor = torch.tensor(indexed_tokens).to(device)
    mask_tensor = torch.tensor(att_masks).to(device)
    text_embed = text_model(tokens_tensor, attention_mask=mask_tensor).last_hidden_state
    return text_embed


def get_text_and_coord_representation(prompt: list, text_tokenizer, text_model, device):
    """
    prompt: list로 받아들이고 있음
    """

    prompt_split = prompt[0].split(', ')
    # print(prompt_split)
    assert len(prompt_split) ==3, "[ERROR] You might have missed to include 2 commas in the prompt"
    text_part = prompt_split[2]
    coord_part = (int(prompt_split[0].strip('[()]')), int(prompt_split[1].strip('[()]')))
    text_embed = get_text_representation([text_part], text_tokenizer, text_model, device)

    return text_embed, coord_part

