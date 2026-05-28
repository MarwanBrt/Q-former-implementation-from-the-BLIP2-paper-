print("Starting...")

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import BertTokenizer, BertConfig, BertLMHeadModel, BertModel
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer, DistilBertForMaskedLM
import accelerate

from src.multimodal_vlm.models.components import setup_BERT
from src.multimodal_vlm.models.qformer import Qformer
from src.multimodal_vlm.models.trainer import Trainer
from src.multimodal_vlm.models.multimodal import MultiModal
from src.multimodal_vlm.data.dataset import VisionTextDataProcessor

from transformers import ViTImageProcessor, ViTForImageClassification, ViTModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import requests

from huggingface_hub import login

login("hf_token")

print("Loading ViT...")

ViT_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224", device_map="cuda")
ViT_model = ViTModel.from_pretrained("google/vit-base-patch16-224", device_map="cuda")

## Test Image to extract hidden state and make sure the ViT is working

image_url = "https://media.istockphoto.com/id/517188688/de/foto/berglandschaft.jpg?s=612x612&w=0&k=20&c=o-aMrF8VR-eelasAPmp7gFtbVy3ssL8UDZff5fTran8="

image = Image.open(requests.get(image_url, stream=True).raw)

inputs = ViT_processor(images=image, return_tensors='pt').to(device="cuda")
# print(inputs)
with torch.no_grad():
    outputs = ViT_model(**inputs)

vit_hidden_state = outputs.last_hidden_state

print("ViT Loaded and working")

print("Loading LLM...")

LLM_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device_map="cuda")
text_processor = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype="auto", device_map="cuda")

# Test LLM and get input embedding dim

messages = [{"role": "user", "content": "hi! how are you"}]
input = text_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = text_processor(input, device="cuda", return_tensors='pt').to("cuda")

# with torch.no_grad():
#     output = LLM_model.generate(**inputs, max_new_tokens=50)
# result = text_processor.decode(output[0], skip_special_tokens=True)

imbed = LLM_model.get_input_embeddings()
llm_embed_dim = imbed(inputs["input_ids"]).shape[-1]

print("LLM Loaded and working")


print("Setting VLM...")

MyBertModel, MyBertTokenizer = setup_BERT()
q_former = Qformer(hidden_size=768, llm_channel_width=llm_embed_dim, bert_model=MyBertModel, tokenizer=MyBertTokenizer, vision_width=vit_hidden_state.shape[-1], num_query_tokens=32, embd_dim=256).to(device="cuda")

print("Loading the data...")

dataset = load_dataset("RIW/small-coco")

train_set = VisionTextDataProcessor(
    dataset["train"].select(range(3000)),
    processor=ViT_processor,
    bert_tokenizer=MyBertTokenizer,
    llm_tokenizer=text_processor
)

val_set = VisionTextDataProcessor(
    dataset["train"].select(range(3000,3500)),
    processor=ViT_processor,
    bert_tokenizer=MyBertTokenizer,
    llm_tokenizer=text_processor
)

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)

print("Data successfully loaded")

complete_model = MultiModal(ViT_model, LLM_model, q_former)
trainer = Trainer(complete_model, train_dataloader, val_dataloader, epochs=100, lr=3e-3)

print("Training the model...")

## Train on the first stage
trainer.train()

print("Training Finished")