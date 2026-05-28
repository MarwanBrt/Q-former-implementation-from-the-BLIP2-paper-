from transformers import BertTokenizer, BertConfig, BertLMHeadModel, BertModel
import accelerate

def setup_BERT(model_name="bert-base-uncased", hidden_size=768, vision_width=1408, cross_freq=2):

    config = BertConfig.from_pretrained(model_name)
    config.hidden_size = hidden_size

    ## Add cross-attention for ViT embeddings
    config.encoder_width = vision_width
    config.add_cross_attention = True
    config.cross_attention_freq = cross_freq
    config.is_decoder = True
    config.use_cache = True

    qformer = BertLMHeadModel.from_pretrained(
        model_name,
        config=config, device_map="cuda"
    )
    tokenizer = BertTokenizer.from_pretrained(model_name, device_map="cuda")
    return qformer, tokenizer