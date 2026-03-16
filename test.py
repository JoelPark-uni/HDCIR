# pip install accelerate
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import open_clip
pretraining = {
'ViT-B-32':'laion2b_s34b_b79k',
'ViT-B-16':'laion2b_s34b_b88k',
'ViT-L-14':'laion2b_s32b_b82k',
'ViT-H-14':'laion2b_s32b_b79k',
'ViT-g-14':'laion2b_s34b_b88k',
'ViT-bigG-14':'laion2b_s39b_b160k'
}
device = 'cuda'
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained="laion2b_s34b_b79k")
clip_model = clip_model.eval().requires_grad_(False).to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip_model.tokenizer = tokenizer

text = ""
text_input = clip_model.tokenizer(text)
print(clip_model.encode_text(text_input.to(device)).shape)

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", device_map="auto")

# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# question = "how many dogs are in the picture?"
# inputs = processor(raw_image, question, return_tensors="pt").to("cuda")

# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True))
