import pandas as pd
import json
import re

# Load the data
df = pd.read_csv('./data/raw/Insurance_Plan_Details_Detailed.csv')

structured_rules = []

for index, row in df.iterrows():
    plan_name = str(row['Provider & Plan Name']).strip()
    eligibility_text = str(row['Eligibility']).strip()
    limits_text = str(row['Limits']).strip()
    exclusions_text = str(row['Exclusions']).strip()
    
    # 1. Parse Eligibility (Extracting Age Ranges)
    min_age, max_age = None, None
    age_match = re.search(r'(?:Age|Between)\s*(\d+)\s*(?:-|to|and)\s*(\d+)', eligibility_text, re.IGNORECASE)
    if age_match:
        min_age = int(age_match.group(1))
        max_age = int(age_match.group(2))
        
    # 2. Parse Exclusions (Splitting into lists)
    # Remove trailing periods and split by commas or the word 'and'
    exclusions_clean = re.sub(r'\.$', '', exclusions_text)
    exclusions_list = [ex.strip().capitalize() for ex in re.split(r',|\band\b', exclusions_clean) if len(ex.strip()) > 2]
    
    # Structure the rule for this plan
    rule = {
        "plan_id": f"PLAN_{index+1:03d}",
        "plan_name": plan_name,
        "parameters": {
            "eligibility": {
                "raw_text": eligibility_text,
                "min_age": min_age,
                "max_age": max_age
            },
            "limits": {
                "raw_text": limits_text
            },
            "exclusions": exclusions_list
        }
    }
    structured_rules.append(rule)

# Save to a JSON file
output_file = './data/rules/Structured_Insurance_Rules.json'
with open(output_file, 'w') as f:
    json.dump(structured_rules, f, indent=4)

print(f"Successfully processed {len(structured_rules)} plans into JSON.")
print("\n--- Example Output (First 3 Plans) ---")
print(json.dumps(structured_rules[:3], indent=2))