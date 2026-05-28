import torch
import torch.nn as nn
import torch.nn.functional as F
import accelerate

class MultiModal(nn.Module):
    def __init__(self, vit, llm, qformer):
        super().__init__()
        self.vit = vit
        self.llm = llm
        self.q_former = qformer

        ## making sure qformer is trainable
        for param in self.q_former.parameters():
            param.requires_grad = True

        ## freezing llm
        for param in self.llm.parameters():
            param.requires_grad = False

        ## freezing vit
        for param in self.vit.parameters():
            param.requires_grad = False


        self.llm_embedding_layer = self.llm.get_input_embeddings()

    def forward(self, image_inputs, bert_inputs, bert_input_mask=None, input_ids=None, attention_mask=None):

        ## Stage 1

        ## ViT output
        # print(image_inputs)
        with torch.no_grad():
            vision_features = self.vit(pixel_values=image_inputs)
        image_embeds = vision_features.last_hidden_state

        ## Q-Former output
        qformer_output, qformer_loss = self.q_former(image_embeds=image_embeds, bert_inputs=bert_inputs, bert_input_mask=bert_input_mask)
        # qformer_output = qformer_output.to(dtype=torch.bfloat16)

        # Extract query tokens from Qformer output
        # if isinstance(qformer_output, dict):
        #     query_tokens = qformer_output['query_output']  # (B, 32, 1024)
        # else:
        #     query_tokens = qformer_output  # Fallback

        ## Stage 2

        # if input_ids is not None:
        #     # Get LLM embeddings for text
        #     text_embeds = self.llm_embedding_layer(input_ids).to(dtype=torch.bfloat16)  # (B, seq_len, hidden_size)

        #     # Concatenate query tokens with text embeddings
        #     llm_input_embeds = torch.cat([query_tokens, text_embeds], dim=1)

        #     # Adjust attention mask to include query tokens
        #     if attention_mask is not None:
        #         query_mask = torch.ones(
        #             query_tokens.size(0),
        #             query_tokens.size(1),
        #             dtype=attention_mask.dtype,
        #             device=attention_mask.device
        #         )
        #         llm_attention_mask = torch.cat([query_mask, attention_mask], dim=1)
        #     else:
        #         llm_attention_mask = None
        # else:
        #     llm_input_embeds = query_tokens
        #     llm_attention_mask = None

        # with torch.no_grad():
        #     llm_output = self.llm(
        #         inputs_embeds=llm_input_embeds,
        #         attention_mask=llm_attention_mask,
        #         return_dict=True,
        #     )

        # return {
        #     'query_output': query_tokens,
        #     'llm_logits': llm_output.logits[:, query_tokens.size(1):, :],
        # }

        return {
            "qformer_output": qformer_output,
            "qformer_loss": qformer_loss,
            # 'query_output': query_tokens,
            # 'llm_logits': llm_output.logits[:, query_tokens.size(1):, :],
        }
