import json
import os
import sys

# Suppress some Hugging Face warnings for a cleaner terminal output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import warnings
warnings.filterwarnings("ignore")

from src.inference import ClaimExtractor
from src.engine import ClaimsValidationEngine

def run_demo():
    print("\n" + "="*60)
    print("🏥 AI HEALTH CLAIMS PIPELINE - LIVE DEMO")
    print("="*60)

    # File paths relative to the root directory
    SSOT_PATH = "data/rules/Structured_Insurance_Rules.json"
    ADAPTER_PATH = "models/lora_adapter"
    
    # 1. Pre-Flight Checks
    if not os.path.exists(SSOT_PATH):
        print(f"\n❌ Error: SSOT rules file not found at {SSOT_PATH}")
        print("Please run 'python src/rules_generator.py' first.")
        return
        
    if not os.path.exists(ADAPTER_PATH):
        print(f"\n❌ Error: Fine-tuned model not found at {ADAPTER_PATH}")
        print("Please run 'python src/train.py' to train your local AI first.")
        return

    # 2. Load the Core Components
    print("\n[System] Booting up Local AI (Loading LoRA weights into GPU)...")
    try:
        extractor = ClaimExtractor(adapter_path=ADAPTER_PATH)
    except Exception as e:
        print(f"\n❌ Failed to load the AI model: {e}")
        return

    print("[System] Loading Deterministic Validation Engine...")
    engine = ClaimsValidationEngine(ssot_path=SSOT_PATH)

    # 3. The Unstructured Test Claim
    # We are simulating a doctor's note or an email from a hospital desk.
    test_claim = (
        "Patient name is Sarah Connor, she is 28 years old. Gender: Female. "
        "She was rushed to the hospital and admitted for Dengue fever. "
        "She stayed in a general room for 3 days. "
        "She is filing a claim for 45000 INR under PLAN_001."
    )

    print("\n" + "-"*60)
    print("📥 INCOMING UNSTRUCTURED CLAIM:")
    print(f"'{test_claim}'")
    print("-"*60)

    # 4. AI Extraction Phase (Unstructured -> Structured)
    print("\n🧠 [Phase 1: AI Extraction] Parsing text and extracting metrics...")
    
    # The AI does the heavy lifting of reading English and outputting JSON
    structured_json_str = extractor.extract_claim_data(test_claim)
    
    print("\n📊 AI Output (JSON format):")
    print(structured_json_str)

    # 5. Deterministic Validation Phase (Checking the SSOT)
    print("\n⚖️ [Phase 2: Engine Validation] Cross-referencing against SSOT...")
    decision = engine.evaluate_claim(structured_json_str)

    # 6. Final Output for the User/Boss
    print("\n" + "="*60)
    print("🎯 FINAL SYSTEM DECISION")
    print("="*60)
    
    status_icon = "🟢" if decision['status'] == "APPROVE" else "🔴"
    
    print(f"STATUS: {status_icon} {decision['status']}")
    print(f"SCORE:  {decision['score']}")
    print(f"AUDIT:  {decision['reason']}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_demo() 