from transformers import BertModel, BertConfig,  BertTokenizerFast
from transform import SelectionJoinTransform
from encoder import PolyEncoder
import torch
import os

def Load_Model_Tokenizer(model_path):

    bert_config = BertConfig.from_json_file(os.path.join(model_path, 'config.json'))
    previous_model_file = os.path.join(model_path, 'pytorch_model.bin')
    print(previous_model_file)

    model_state_dict = torch.load(previous_model_file, map_location="cpu")

    tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=True, clean_text=False)
    tokenizer.add_tokens(['\n'], special_tokens=True)
    context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=128)

    bert = BertModel(bert_config)
    bert.resize_token_embeddings(len(tokenizer))

    model = PolyEncoder(bert_config, bert=bert, poly_m=16)
    model.load_state_dict(model_state_dict)
    model.eval()

    print('모델 준비 끝')
    return model, tokenizer