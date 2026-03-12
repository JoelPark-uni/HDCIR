오리지널 코드는 https://github.com/ExplainableML/Vision_by_Language 에서 확인할 수 있습니다.

### Toy FashionIQ 데이터셋 추가(`cir_datasets.py`)

- `mode`: 기존과 동일하게 `classic` / `relative`
    - classic: DB 이미지 피처 추출용 (img_feature, img_name) 인덱스 구성
    - relative: query prediction용 (ref 이미지 + target 라벨)

```python
class ToyFashionIQDataset(Dataset):
	...
	def __getitem__(self, index) -> dict:
		try:
			if self.mode == 'relative':
				relative_captions = self.triplets[index]['captions']
				reference_name = self.triplets[index]['candidate']
				target_name = self.triplets[index]['target']

				reference_image_path = self.dataset_path / 'images' / f"{reference_name}.png"
				reference_image = self.preprocess(PIL.Image.open(reference_image_path))
				target_image_path = self.dataset_path / 'images' / f"{target_name}.png"
				target_image = self.preprocess(PIL.Image.open(target_image_path))
				
				blip_ref_img_pil = PIL.Image.open(reference_image_path).convert('RGB')
				blip_target_img_pil = PIL.Image.open(target_image_path).convert('RGB')

				return {
					'reference_image': reference_image,
					'blip_ref_img_pil': blip_ref_img_pil,
					'blip_target_img_pil': blip_target_img_pil,
					'reference_name': reference_name,
					'target_image': target_image,
					'target_name': target_name,
					'relative_captions': relative_captions
				}

			elif self.mode == 'classic':
				image_name = self.image_names[index]
				image_path = self.dataset_path / 'images' / f"{image_name}.png"
				image = self.preprocess(PIL.Image.open(image_path))
				return {
					'image': image,
					'image_name': image_name
				}

			else:
				raise ValueError("mode should be in ['relative', 'classic']")
		except Exception as e:
			print(f"Exception: {e}")
```

### LLM 변경: OpenAI API → Phi-3-mini

```python
@torch.no_grad()
def completion(self, prompt: str, max_new_tokens: int = 256) -> str:
	"""
	Text-only inference with Phi-3-mini.
	"""
	self._ensure_loaded()
	assert self.model is not None
	assert self.tokenizer is not None

	inputs = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True)
	input_ids = inputs["input_ids"].to(self.device)
	attention_mask = inputs["attention_mask"].to(self.device)

	generated = self.model.generate(
		input_ids=input_ids,
		attention_mask=attention_mask,
		max_new_tokens=max_new_tokens,
		do_sample=False,
		use_cache=True,
		eos_token_id=self.tokenizer.eos_token_id,
		pad_token_id=self.tokenizer.pad_token_id,
	)

	new_tokens = generated[:, input_ids.shape[1]:]
	text = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
	return text.strip()
```

- `transformers==4.46.3` 기반 Phi3ForCausalLM 사용

### BLIP2 변경: XXL → XL + 라이브러리 조정

- Flan-T5-XXL(11B) 대신 Flan-T5-XL(약 4.3B) 사용
- 호환성 이슈로 라이브러리 분리
    - BLIP2, Phi3: `transformers==4.46.3`
    - CLIP: `openai-clip==1.0.1`
- LAVIS는 이미지 전처리만 수행하나, transformers BLIP 프로세서는 prompt 입력과 함께 처리
    - 데이터셋 전처리를 PIL 반환으로 바꾸고, 추론 시 BLIP 프로세서 적용
