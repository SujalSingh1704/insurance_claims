import json

class ClaimsValidationEngine:
    def __init__(self, ssot_path="data/rules/Structured_Insurance_Rules.json"):
        with open(ssot_path, 'r', encoding='utf-8') as f:
            self.rules = json.load(f)
        self.plan_map = {plan['plan_id']: plan for plan in self.rules}

    def evaluate_claim(self, llm_output_json):
        try:
            extracted_data = json.loads(llm_output_json)
        except json.JSONDecodeError:
            return {"status": "REJECT", "score": 0.0, "reason": "Malformed AI extraction."}

        plan_id = extracted_data.get("plan_id")
        age = extracted_data.get("age", 0)
        condition_keywords = extracted_data.get("condition_keywords", [])
        
        if plan_id not in self.plan_map:
            return {"status": "REJECT", "score": 0.0, "reason": f"Plan ID {plan_id} not found in SSOT."}
            
        plan_rules = self.plan_map[plan_id]
        eligibility = plan_rules.get("parameters", {}).get("eligibility", {})
        exclusions = plan_rules.get("parameters", {}).get("exclusions", [])
        
        min_age, max_age = eligibility.get("min_age"), eligibility.get("max_age")
        
        if min_age is not None and age < min_age:
            return {"status": "REJECT", "score": 0.0, "reason": f"Age {age} is below minimum {min_age}."}
        if max_age is not None and age > max_age:
            return {"status": "REJECT", "score": 0.0, "reason": f"Age {age} exceeds maximum {max_age}."}
            
        for keyword in condition_keywords:
            for exclusion in exclusions:
                if keyword.lower() in exclusion.lower():
                    return {"status": "REJECT", "score": 0.0, "reason": f"Triggered policy exclusion: '{exclusion}'."}
                    
        return {"status": "APPROVE", "score": 1.0, "reason": "Claim verified against plan parameters."}