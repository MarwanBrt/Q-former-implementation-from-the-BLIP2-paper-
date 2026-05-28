from torch.utils.data import Dataset


class VisionTextDataProcessor(Dataset):
    def __init__(self, dataset, processor, bert_tokenizer, llm_tokenizer):
        super().__init__()
        self.dataset = dataset
        self.processor = processor
        self.bert_tokenizer = bert_tokenizer
        self.llm_tokenizer = llm_tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        image = item["image"]
        caption = item["caption"]

        messages = [
            {"role": "user", "content": f"describe the image"},
            {"role": "assistant", "content": caption}
        ]

        # -----------------------

        # Tokenize text for q-former
        tokenized_BERT_inputs = self.bert_tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )

        bert_input_ids = tokenized_BERT_inputs["input_ids"].to('cuda').squeeze(0)
        bert_input_mask = tokenized_BERT_inputs["attention_mask"].to('cuda').squeeze(0)

        # ==> already done in qformer
        # embedding_layer = BertModel.get_input_embeddings()
        # Bert_text_embeds = embedding_layer(input_ids)

        # -----------------------

        # Tokenize text for LLM
        llm_text = self.llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        llm_inputs = self.llm_tokenizer(llm_text, return_tensors='pt',
                                        padding="max_length",
                                        truncation=True,
                                        max_length=128,
                                        )

        # -----------------------

        # Process image
        image_inputs = self.processor(images=image, return_tensors='pt').to(device="cuda")
        pixel_values = image_inputs["pixel_values"].squeeze(0)  # (C, H, W)

        llm_input_ids = llm_inputs["input_ids"].squeeze(0)  # (llm_seq_len,)
        llm_attention_mask = llm_inputs["attention_mask"].squeeze(0)

        # return {
        #     # "text_inputs": caption,
        #     "BertInputs": input_ids,
        #     "image_inputs": image_inputs,
        #     "input_ids": tokenized_text_inputs["input_ids"].squeeze(0),
        #     "attention_mask": tokenized_text_inputs["attention_mask"].squeeze(0),
        # }

        return {
            "pixel_values": pixel_values,  # (C, H, W)
            "BertInputs": bert_input_ids,  # (bert_len,)
            "BertInputMask": bert_input_mask,
            "llm_input_ids": llm_input_ids,  # (llm_seq_len,)
            "llm_attention_mask": llm_attention_mask,  # (llm_seq_len,)
            "caption": caption,
        }