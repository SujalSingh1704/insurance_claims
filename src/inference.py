import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

class ClaimExtractor:
    def __init__(self, adapter_path):
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        self.model = PeftModel.from_pretrained(base_model, adapter_path)

    def extract_claim_data(self, claim_text):
        # The Fix: Multi-Turn Conversation + Fill-in-the-Blank Blueprint
        messages = [
            {
                "role": "user",
                "content": (
                    "Extract the data from the text and populate this EXACT JSON template. "
                    "Return NOTHING but the JSON. The claim_amount must be numbers only.\n\n"
                    "Template:\n"
                    "{\n"
                    '  "plan_id": "",\n'
                    '  "age": 0,\n'
                    '  "claim_amount": 0,\n'
                    '  "condition_keywords": []\n'
                    "}\n\n"
                    "Text: Patient is John, 45 years old. Admitted for a broken leg. Claiming 5000 INR under PLAN_002."
                )
            },
            {
                "role": "assistant",
                "content": (
                    "{\n"
                    '  "plan_id": "PLAN_002",\n'
                    '  "age": 45,\n'
                    '  "claim_amount": 5000,\n'
                    '  "condition_keywords": ["broken leg"]\n'
                    "}"
                )
            },
            {
                "role": "user",
                "content": (
                    "Extract the data from the text and populate this EXACT JSON template. "
                    "Return NOTHING but the JSON. The claim_amount must be numbers only.\n\n"
                    "Template:\n"
                    "{\n"
                    '  "plan_id": "",\n'
                    '  "age": 0,\n'
                    '  "claim_amount": 0,\n'
                    '  "condition_keywords": []\n'
                    "}\n\n"
                    f"Text: {claim_text}"
                )
            }
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # We use greedy decoding (do_sample=False) to ensure strict adherence to the template
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        prompt_length = inputs.input_ids.shape[-1]
        new_tokens = outputs[0][prompt_length:]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # Greedy Regex to capture the full JSON block safely
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
        
        if json_match:
            return json_match.group(0)
        else:
            return output_text