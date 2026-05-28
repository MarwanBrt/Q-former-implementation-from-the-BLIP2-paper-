import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import BertTokenizer, BertConfig, BertLMHeadModel, BertModel
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer, DistilBertForMaskedLM
import accelerate

from datasets  import load_dataset

from transformers import ViTImageProcessor, ViTForImageClassification, ViTModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import requests

from huggingface_hub import login

from dotenv import load_dotenv
load_dotenv()

## add huggingface token
login(os.environ["HF_TOKEN"])




