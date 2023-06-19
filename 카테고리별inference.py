import torch
from transform import SelectionSequentialTransform, SelectionJoinTransform
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

class Category_Callcenter(): 
    def __init__(self, model, tokenizer, emb_df, device): 
        self.context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=128)
        self.response_transform = SelectionSequentialTransform(tokenizer=tokenizer, max_len=128)
        self.model = model
        self.emb_df = emb_df
        self.device = device

    def cosine_score(self, query, idx): 
        with torch.no_grad():
            response =[query]
            responses_token_ids_list, responses_input_masks_list = self.response_transform(response)  # [token_ids],[seg_ids],[masks]
            long_tensors = [responses_token_ids_list, responses_input_masks_list]
            responses_token_ids_list, responses_input_masks_list = (torch.tensor(t, dtype=torch.long, device=self.device) for t in long_tensors)

            ids = responses_token_ids_list.unsqueeze(1)
            masks = responses_input_masks_list.unsqueeze(1)
            seq_length = ids.shape[-1]

            ids = ids.view(-1, seq_length)
            masks = masks.view(-1, seq_length)

            question_emb = self.model.bert(ids, masks)[0][:, 0, :].to('cpu').detach().numpy()
            # print(question_emb.shape)
            compare_question_embs = torch.stack(list(self.emb_df.iloc[idx][['q1', 'q2', 'q3', 'q4']].values), dim=1)
            compare_question_embs = compare_question_embs.to('cpu').detach().numpy()[0]
            # print(compare_question_embs.shape)
            # 코사인 유사도 계산
            similarity = cosine_similarity(question_emb, compare_question_embs)[0]

            print("코사인 유사도min:", np.min(similarity))
            print("코사인 유사도mean:", np.mean(similarity))
            print("코사인 유사도max:", np.max(similarity))
            return np.max(similarity)

    def inference(self, query): 
        def context_input(context):
            context_input_ids, context_input_masks = self.context_transform(context)
            contexts_token_ids_list_batch, contexts_input_masks_list_batch = [context_input_ids], [context_input_masks]
            long_tensors = [contexts_token_ids_list_batch, contexts_input_masks_list_batch]
            contexts_token_ids_list_batch, contexts_input_masks_list_batch = (torch.tensor(t, dtype=torch.long, device=self.device) for t in long_tensors)
            return contexts_token_ids_list_batch, contexts_input_masks_list_batch
        
        def embs_gen(contexts_token_ids_list_batch, contexts_input_masks_list_batch):
            ctx_out = self.model.bert(contexts_token_ids_list_batch, contexts_input_masks_list_batch)[0]  # [bs, length, dim]
            poly_code_ids = torch.arange(self.model.poly_m, dtype=torch.long).to(self.device)
            poly_code_ids = poly_code_ids.unsqueeze(0).expand(1, self.model.poly_m)
            poly_codes = self.model.poly_code_embeddings(poly_code_ids) # [bs, poly_m, dim]
            embs = self.model.dot_attention(poly_codes, ctx_out, ctx_out) # [bs, poly_m, dim]
            return embs
        
        def score(embs, cand_emb):
            ctx_emb = self.model.dot_attention(cand_emb, embs, embs) # [bs, res_cnt, dim]
            dot_product = (ctx_emb*cand_emb).sum(-1)
            return dot_product

        with torch.no_grad(): 
            answers = self.emb_df
            cand_embs = torch.stack(answers['response'].tolist(), dim=1).to(self.device)
            embs = embs_gen(*context_input([query]))
            embs = embs.to(self.device)
            s = score(embs, cand_embs)
            idx = int(s.argmax(1)[0])
            self.cosine_score(query, idx)
            print(idx)
            # if self.cosine_score(query, idx)<0.93:
            #     return idx, '죄송합니다 시민님, 답을 찾기 어려운 질문입니다. 다시 질문해 주시겠습니까?'
            return idx