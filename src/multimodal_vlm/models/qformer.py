import torch
import torch.nn as nn
import torch.nn.functional as F
import accelerate
import numpy as np

class Qformer(nn.Module):
    def __init__(self, hidden_size=768, vision_width=1408, llm_channel_width: int = 1024, num_query_tokens=32, embd_dim=256, bert_model=None, tokenizer=None):
        super().__init__()

        self.num_query_tokens = num_query_tokens

        # config = BertConfig.from_pretrained("bert-base-uncased")
        # config.hidden_size = hidden_size
        # config.encoder_width = vision_width
        # config.add_cross_attention = True
        # config.is_decoder = True
        # config.cross_attention_freq = 2

        self.qformer = bert_model
        self.tokenizer = tokenizer

        self.embedding_layer = self.qformer.get_input_embeddings()

        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, hidden_size)).to(
            device="cuda")  # 32, hidden_size
        self.vision_proj_layer = nn.Linear(vision_width, hidden_size)

        self.vision_proj = nn.Linear(hidden_size, embd_dim)
        self.text_proj = nn.Linear(hidden_size, embd_dim)

        self.match_proj = nn.Linear(hidden_size, 2)

        self.temp = nn.Parameter(torch.ones([]) * 0.07)

        # LLM projection
        self.llm_project = nn.Linear(hidden_size, llm_channel_width)


    def _get_cross_attention_mask(self, B, query_len, num_patches, text_len, device):

        cross_attention_mask = torch.ones((B, num_patches), device=device, dtype=torch.long)
        return cross_attention_mask


    def _get_itc_attention_mask(self, B, bert_input_mask, query_len, num_patches, text_len, device):

        query_self_attention_mask = torch.ones((B, query_len), device=device, dtype=torch.long)
        text_self_attention_mask = torch.ones((B, text_len), device=device, dtype=torch.long)

        # Mask padding
        if bert_input_mask is not None:
            text_self_attention_mask = text_self_attention_mask & bert_input_mask

        cross_attention_mask = torch.ones((B, num_patches), device=device, dtype=torch.long)

        return query_self_attention_mask, text_self_attention_mask, cross_attention_mask

    # def _get_itc_attention_mask(self, B, query_len, num_patches, text_len, device):
    #     seq_len = query_len + text_len
    #     query_self_attention_mask = torch.ones(query_len, device=device, dtype=torch.long)
    #     text_self_attention_mask = torch.ones(text_len, device=device, dtype=torch.long)

    #     cross_attention_mask = torch.ones(num_patches, query_len, device=device, dtype=torch.long)

    #     return query_self_attention_mask.expand(B, -1), text_self_attention_mask.expand(B, -1), cross_attention_mask.expand(B, -1, -1)

    def _get_itm_attention_mask(self, B, bert_input_mask, query_len, num_patches, text_len, device):
        seq_len = query_len + text_len
        attention_mask = torch.ones((B, seq_len), device=device, dtype=torch.long)

        # Mask padding
        if bert_input_mask is not None:
            attention_mask[:, query_len:] = attention_mask[:, query_len:] & bert_input_mask
        cross_attention_mask = torch.ones((B, num_patches), device=device, dtype=torch.long)  # all is attending to patches

        return attention_mask, cross_attention_mask

    # def _get_itg_attention_mask(self, B, query_len, num_patches, text_len, device):
    #     seq_len = query_len + text_len
    #     attention_mask = torch.ones((B, seq_len), device=device, dtype=torch.long)
    #     cross_attention_mask = torch.ones((B, num_patches), device=device, dtype=torch.long)
    #
    #     return attention_mask, cross_attention_mask

    def _get_itg_attention_mask(self, B, bert_input_mask, query_len, num_patches, text_len, device):
        seq_len = query_len + text_len
        attention_mask = torch.ones((B, seq_len), device=device, dtype=torch.long)
        # attention_mask[:, :query_len, :query_len] = 1

        # causal_mask = torch.tril(torch.ones(text_len, text_len, device=device, dtype=torch.long))
        # attention_mask[:, query_len:] = causal_mask

        # Mask padding
        if bert_input_mask is not None:
            attention_mask[:, query_len:] = attention_mask[:, query_len:] & bert_input_mask

        cross_attention_mask = torch.ones((B, num_patches), device=device, dtype=torch.long)  # all is attending to patches
        # cross_attention_mask[:, query_len:, :] = 0

        return attention_mask, cross_attention_mask

    def _get_Bert_tokenizer_embedding(self):
        return self.tokenizer, self.embedding_layer

    # @torch.no_grad()
    def forward(self, image_embeds, bert_inputs, bert_input_mask=None, labels=None):
        if self.tokenizer is None or self.qformer is None:
            return None

        B = image_embeds.size(0)
        device = image_embeds.device

        # tokenized_text_input = self.tokenizer(
        #     bert_inputs,
        #     return_tensors="pt",
        #     padding=True,
        #     truncation=True
        # )

        # input_ids = tokenized_text_input["input_ids"].to(device)

        # with torch.no_grad():

        bert_embeds = self.embedding_layer(bert_inputs)

        query_tokens = self.query_tokens.expand(B, -1, -1)  # (B, 32, hidden_size)

        combined_input = torch.cat([query_tokens, bert_embeds], dim=1)  # (B, query_len + text_len, hidden_size)

        image_embeds_proj = self.vision_proj_layer(image_embeds)  # (B, num_patches, hidden_size)

        query_len = query_tokens.size(1)
        text_len = bert_embeds.size(1)
        seq_len = query_len + text_len
        num_patches = image_embeds.size(1)

        # ITC

        itc_query_attention_mask, itc_text_attention_mask, itc_cross_attention_mask = self._get_itc_attention_mask(B,
                                                                                                                   bert_input_mask,
                                                                                                                   query_len,
                                                                                                                   num_patches,
                                                                                                                   text_len,
                                                                                                                   device)

        ## Splitting on two forward passes to separate restrict the self-attention and the cross-attention

        itc_query_output = self.qformer.bert(inputs_embeds=query_tokens,
                                             attention_mask=itc_query_attention_mask,
                                             encoder_hidden_states=image_embeds_proj,
                                             encoder_attention_mask=itc_cross_attention_mask,
                                             return_dict=True,
                                             )
        itc_text_output = self.qformer.bert(inputs_embeds=bert_embeds,
                                            attention_mask=itc_text_attention_mask,
                                            return_dict=True,
                                            )


        itc_query_output = itc_query_output.last_hidden_state
        itc_text_output = itc_text_output.last_hidden_state

        cls_text_token = itc_text_output[:, 0, :]
        # image_feat = itc_query_output.mean(dim=1)  # B, hidden_size
        image_feat = itc_query_output[:, 0, :] # also trying with the cls from the vit

        cls_text_token = self.text_proj(cls_text_token)  # B, embd_dim
        image_feat = self.vision_proj(image_feat)  # B, embd_dim

        cls_text_token = F.normalize(cls_text_token, dim=-1)
        image_feat = F.normalize(image_feat, dim=-1)

        sim_i2t = (image_feat @ cls_text_token.T) * self.temp
        sim_t2i = (cls_text_token @ image_feat.T) * self.temp

        itc_targets = torch.arange(B, device=device)

        itc_loss = (F.cross_entropy(sim_i2t, itc_targets) + F.cross_entropy(sim_t2i, itc_targets)) / 2

        # ITM

        ## for ITM we can combine the query tokens and text tokens in self-attention

        itm_self_attention, itm_cross_attention = self._get_itm_attention_mask(B,
                                                                               bert_input_mask,
                                                                               query_len,
                                                                               num_patches,
                                                                               text_len,
                                                                               device)

        itm_output = self.qformer.bert(inputs_embeds=combined_input,
                                       attention_mask=itm_self_attention,
                                       encoder_hidden_states=image_embeds_proj,
                                       encoder_attention_mask=itm_cross_attention,
                                       return_dict=True)

        itm_query_output = itm_output.last_hidden_state # [:, :query_len, :] # B, 32, hidden_size
        itm_text_output = itm_output.last_hidden_state # [:, query_len:, :]

        logits = self.match_proj(itm_query_output)  # B, 32, 2
        logits = logits.mean(dim=1)  # B, 2

        # Negatives
        shuffle = torch.randperm(B, device=device)
        shuffled_logits  = logits[shuffle]

        ## get the false positives
        hard_negatives_list = []
        for i in range(B):
            false_positives_logits = torch.cat((logits[:i,:], logits[i+1:,:]), dim=0)
            hard_negs = false_positives_logits[false_positives_logits[:, 0] > false_positives_logits[:, 1]]
            if hard_negs.shape[0] > 0:
                hard_negatives_list.append(hard_negs)


        # negative_logits = self.match_proj(negatives)  # B, 32, 2
        # negative_logits = negative_logits.mean(dim=1)  # B, 2

        if hard_negatives_list:
            hard_negatives = torch.cat(hard_negatives_list, dim=0)
        else:
            hard_negatives = torch.tensor([], device=device, dtype=logits.dtype).reshape(0, 2)

        itm_final_logits = torch.cat([logits, hard_negatives], dim=0) # B*2, 2

        itm_targets = torch.cat(
            [torch.ones(B, dtype=torch.long, device=device), torch.zeros(hard_negatives.shape[0], dtype=torch.long, device=device)], dim=0) # B*2,


        itm_loss = F.cross_entropy(itm_final_logits, itm_targets)

        # ITG
        itg_self_attention, itg_cross_attention = self._get_itg_attention_mask(B,
                                                                               bert_input_mask,
                                                                               query_len,
                                                                               num_patches,
                                                                               text_len,
                                                                               device)

        labels = torch.full((B, seq_len), -100, dtype=torch.long, device=device)

        # Set labels to the actual token IDs
        labels[:, query_len:] = bert_inputs


        lm_outputs = self.qformer(inputs_embeds=combined_input,
                                  attention_mask=itg_self_attention,
                                  encoder_hidden_states=image_embeds_proj,
                                  encoder_attention_mask=itg_cross_attention,
                                  return_dict=True,
                                  labels=labels,
                                  output_hidden_states=True
                                  )


        itg_loss = lm_outputs.loss
        total_loss = itm_loss + itc_loss + itg_loss
        print("itm_loss:", itm_loss)
        print("itc_loss:", itc_loss)
        print("itg_loss:", itg_loss)

        query_hidden_state = lm_outputs.hidden_states[-1][:, :query_len, :]
        qformer_final_output = self.llm_project(query_hidden_state)

        return qformer_final_output, total_loss

        ## final forward to get queries
        # final_query_attention_mask = torch.ones(query_len, device=device, dtype=torch.long)
        # final_query_cross_attention = torch.ones(num_patches, query_len, device=device, dtype=torch.long)
        # query_output = self.qformer.bert(inputs_embeds=query_tokens,
        #                       attention_mask=final_query_attention_mask,
        #                       encoder_hidden_states=image_embeds_proj,
        #                       encoder_attention_mask=final_query_cross_attention,
        #                       return_dict=True,
        #                       )

        # return itc_query_output, total_loss

    @torch.no_grad()
    def _forward(self, image_embeds, bert_inputs=None, labels=None):
        B = image_embeds.size(0)
        device = image_embeds.device

        # bert_embeds = self.embedding_layer(bert_inputs)
        query_tokens = self.query_tokens.expand(B, -1, -1)  # (B, 32, hidden_size)
        image_embeds_proj = self.vision_proj_layer(image_embeds)  # (B, num_patches, hidden_size)

        query_len = query_tokens.size(1)
        # text_len = bert_embeds.size(1)
        # seq_len = query_len + text_len
        num_patches = image_embeds.size(1)

        final_query_attention_mask = torch.ones((B, query_tokens.size(1)),
                                                device=device,
                                                dtype=torch.long)
        final_query_cross_attention = torch.ones((B, image_embeds.size(1)),
                                                 device=device,
                                                 dtype=torch.long)

        query_output = self.qformer(
            inputs_embeds=query_tokens,
            attention_mask=final_query_attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=final_query_cross_attention,
            return_dict=True,
        )

        ## TODO: Project for LLM
        return query_output

