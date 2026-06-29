import pandas as pd
import json
import random
import re
import os

def clean_currency(value):
    if pd.isna(value): return 0.0
    clean_str = re.sub(r'[^\d.]', '', str(value))
    return float(clean_str) if clean_str else 0.0

def convert_csv_to_jsonl(csv_path, output_jsonl_path, ssot_path):
    print("Loading dataset and rules...")
    df = pd.read_csv(csv_path)
    
    with open(ssot_path, 'r', encoding='utf-8') as f:
        rules = json.load(f)
        valid_plan_ids = [plan["plan_id"] for plan in rules]
    
    medical_procedures = [
        "Viral Fever", "Appendectomy", "Covid-19", "Cosmetic surgery", 
        "Cataract surgery", "Maternity/Childbirth", "Cardiac Arrest", 
        "Hazardous sports injury", "Dengue", "Knee Replacement"
    ]
    
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            age = int(row.get('age', 30))
            gender = "Female" if str(row.get('gender')).upper() == 'F' else "Male"
            room = str(row.get('room_category', 'general'))
            los = int(row.get('length_of_stay', 1))
            claim_amount = clean_currency(row.get('total_claim_amount', 0))
            
            plan_id = random.choice(valid_plan_ids)
            diagnosis = random.choice(medical_procedures)
            
            instruction = (
                f"Extract metrics from this medical claim: A {age}-year-old {gender} "
                f"filed a claim under policy {plan_id} for {diagnosis}. They were hospitalized "
                f"for {los} days in a {room} room. The total claimed amount is {claim_amount} INR."
            )
            
            output_data = {
                "plan_id": str(plan_id), "age": age,
                "claim_amount": claim_amount, "condition_keywords": [diagnosis]
            }
            
            training_sample = {"text": f"Instruction: {instruction}\nOutput: {json.dumps(output_data)}"}
            f.write(json.dumps(training_sample) + '\n')
            
    print(f"Success! Instruction dataset saved at: {output_jsonl_path}")

if __name__ == "__main__":
    convert_csv_to_jsonl(
        csv_path="./data/raw/indian_health_insurance_claims_dataset.csv", 
        output_jsonl_path="./data/processed/train_data.jsonl", 
        ssot_path="./data/rules/Structured_Insurance_Rules.json"
    )