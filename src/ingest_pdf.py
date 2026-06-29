import os
import json
import time
import pandas as pd
import pdfplumber
import glob
from google import genai
from google.genai import types

# ==========================================
# CONFIGURATION
# ==========================================
GEMINI_API_KEY = ""

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
os.makedirs(DATA_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Reads a PDF file and extracts all text efficiently."""
    text_pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_pages.append(text)
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return None
        
    return "\n".join(text_pages)

def extract_structured_data_with_ai(filename, raw_text, client):
    """Uses Gemini SDK with auto-retries for Rate Limits and JSON errors."""
    
    prompt = f"""
    You are an expert insurance underwriter and auditor. Your task is to deeply analyze the provided insurance policy document and extract highly detailed, comprehensive information. 
    
    DO NOT summarize briefly. You must be exhaustive. Capture specific sub-limits, exact waiting periods, detailed lists of inclusions/exclusions, and optional covers exactly as they appear in the text.
    
    Format your response as a valid JSON object. For fields that require lists, use bullet points (•) separated by newlines within the JSON string so it formats cleanly into a multi-line CSV cell.
    
    Keys to include in the JSON:
    "Provider & Plan Name": (Use the filename '{filename}' as a hint, but format it cleanly as 'Provider: Plan Name')
    "Eligibility": (Provide exhaustive rules on family composition, proposer requirements, and age. STRICT RULE: If there are age limits, you MUST write them using the exact phrase 'Age [Min] to [Max]'.)
    "Key Benefits": (Provide an exhaustive bulleted list (using '•') of all covered benefits, optional add-ons, specific diseases covered, and accidental covers.)
    "Limits": (Provide a comprehensive bulleted list (using '•') of all financial caps. You MUST include exact percentages and rupee amounts for room rent, ICU, pre/post-hospitalization days, ambulance caps, maternity limits, and specific treatment sub-limits.)
    "Exclusions": (Provide a comprehensive bulleted list (using '•') of all waiting periods (e.g., 30-day, 24-month PED), standard exclusions, permanent exclusions, and unsupported treatments.)
    
    Insurance Policy Text:
    -----------------------
    {raw_text} 
    """
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    # 0.1 prevents the LLM "stuttering" loop that breaks JSON formatting
                    temperature=0.1 
                )
            )
            
            structured_data = json.loads(response.text)
            return structured_data
            
        except json.JSONDecodeError:
            print(f"   ⚠️ Attempt {attempt + 1}/{max_retries}: AI generated invalid JSON (stutter bug). Retrying...")
            time.sleep(3) # Short pause before retrying a formatting error
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                # Exponential backoff: Wait 10s, then 20s, then 30s
                sleep_time = 10 * (attempt + 1)
                print(f"   ⏳ Rate limit hit (429). Sleeping for {sleep_time} seconds before retry {attempt + 1}/{max_retries}...")
                time.sleep(sleep_time)
            else:
                # If it's a completely different error (like 401 Unauthorized), fail immediately
                print(f"❌ Unrecoverable AI Extraction error for {filename}: {e}")
                break 

    # If the loop finishes all retries without returning, it failed
    return _fallback_dict(filename)

def _fallback_dict(filename):
    """Helper to return a default dictionary on total failure."""
    return {
        "Provider & Plan Name": os.path.splitext(filename)[0],
        "Eligibility": "Extraction Failed",
        "Key Benefits": "Extraction Failed",
        "Limits": "Extraction Failed",
        "Exclusions": "Extraction Failed"
    }

def main():
    print("🤖 AI Master Policy Extractor (Powered by Gemini 1.5 Flash 1M Context)")
    
    pdf_files = glob.glob(os.path.join(DATA_DIR, '**', '*.pdf'), recursive=True)
    
    if not pdf_files:
        print(f"\n❌ No PDF files found in: {DATA_DIR}")
        return

    print(f"\n🔍 Found {len(pdf_files)} PDF(s). Starting deep extraction...\n")
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    all_records = []
    
    for i, pdf_path in enumerate(pdf_files):
        filename = os.path.basename(pdf_path)
        print(f"📄 Processing {i+1}/{len(pdf_files)}: {filename}...")
        
        raw_text = extract_text_from_pdf(pdf_path)
        
        if raw_text:
            print(f"   [Read {len(raw_text):,} characters. Analyzing...]")
            record = extract_structured_data_with_ai(filename, raw_text, client)
            
            if record.get("Eligibility") == "Extraction Failed":
                print("   ❌ Exhausted all retries. Falling back to empty record.\n")
            else:
                print("   ✅ Deep extraction successful.")
            
            all_records.append(record)
            
            # THE PACER: Wait 5 seconds between PDFs to respect the 15 req/min free tier limit.
            # We don't need to sleep after the very last file.
            if i < len(pdf_files) - 1:
                print("   [Pacing] Sleeping 5 seconds to respect API rate limits...\n")
                time.sleep(5)
                
        else:
            print(f"❌ Failed to read text from {filename}.\n")

    if all_records:
        print("\n📦 Compiling all records into a single dataset...")
        df = pd.DataFrame(all_records)
        
        expected_columns = ["Provider & Plan Name", "Eligibility", "Key Benefits", "Limits", "Exclusions"]
        if all(col in df.columns for col in expected_columns):
            df = df[expected_columns]
            
        output_filepath = os.path.join(DATA_DIR, 'Insurance_Plan_Details_Detailed.csv')
        df.to_csv(output_filepath, index=False, encoding='utf-8-sig') 
        
        print(f"🎉 Master extraction complete!")
        print(f"📁 Consolidated CSV saved to: {output_filepath}")
    else:
        print("⚠️ No data was successfully extracted.")

if __name__ == "__main__":
    if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        print("⚠️ WARNING: You need to insert your Gemini API key at the top of the script!")
    else:
        main()