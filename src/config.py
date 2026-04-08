"""Central configuration for Hospital Price Transparency Engine."""
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42

# Hospital generation
N_HOSPITALS = 500
STATES = ["IL", "IN", "WI", "MI", "OH", "MN", "IA", "MO", "NY", "CA", "TX", "FL", "PA", "GA", "NC"]
CHICAGO_CENTER = (41.8781, -87.6298)
RADIUS_MILES = 500

# Canonical schema (the 15 standardized fields every hospital SHOULD have)
CANONICAL_COLUMNS = {
    "hospital_name": "Name of the hospital",
    "hospital_npi": "National Provider Identifier",
    "procedure_code": "CPT/HCPCS/DRG code",
    "procedure_description": "Human-readable procedure name",
    "code_type": "CPT, HCPCS, DRG, or MS-DRG",
    "gross_charge": "Hospital's list price (chargemaster)",
    "discounted_cash_price": "Price for self-pay patients",
    "min_negotiated_rate": "Lowest negotiated insurer rate",
    "max_negotiated_rate": "Highest negotiated insurer rate",
    "payer_name": "Insurance company name",
    "plan_name": "Specific insurance plan",
    "plan_type": "PPO, HMO, EPO, POS, etc.",
    "setting": "Inpatient or Outpatient",
    "billing_class": "Professional or Facility",
    "effective_date": "Date pricing became effective",
}

# Common column name variants (what hospitals actually use)
COLUMN_VARIANTS = {
    "gross_charge": [
        "Gross_Charge", "Standard_Price", "CDM_Amount", "List_Price",
        "Chargemaster_Price", "Full_Price", "Hospital_Charge", "Charge_Amount",
        "gross charge", "GROSS_CHG", "GrossCharge", "Std_Charge", "charge",
    ],
    "discounted_cash_price": [
        "Discounted_Cash_Price", "Cash_Price", "Self_Pay_Price", "Cash_Rate",
        "Self_Pay_Rate", "Uninsured_Price", "Cash_Discount_Price", "SelfPay",
        "cash price", "CASH_PRICE", "CashRate", "discount_cash",
    ],
    "min_negotiated_rate": [
        "Min_Negotiated_Rate", "Minimum_Rate", "De-Identified_Min",
        "DeidentifiedMinimum", "min_rate", "MIN_NEGOTIATED", "Min_Neg_Rate",
        "Payer_Min", "negotiated_min", "lowest_rate",
    ],
    "max_negotiated_rate": [
        "Max_Negotiated_Rate", "Maximum_Rate", "De-Identified_Max",
        "DeidentifiedMaximum", "max_rate", "MAX_NEGOTIATED", "Max_Neg_Rate",
        "Payer_Max", "negotiated_max", "highest_rate",
    ],
    "procedure_code": [
        "Procedure_Code", "CPT_Code", "HCPCS_Code", "DRG_Code", "Code",
        "Billing_Code", "Service_Code", "Proc_Code", "CDM_Code",
        "procedure code", "CPT", "HCPCS", "code_value",
    ],
    "procedure_description": [
        "Procedure_Description", "Description", "Service_Description",
        "Item_Description", "Procedure_Name", "Service_Name", "CDM_Description",
        "procedure description", "DESC", "ItemDesc", "svc_desc",
    ],
    "payer_name": [
        "Payer_Name", "Payer", "Insurance_Company", "Insurance_Name",
        "Insurer", "Carrier", "Plan_Carrier", "Third_Party_Payer",
        "payer name", "PAYER", "PayerName", "ins_company",
    ],
    "plan_name": [
        "Plan_Name", "Plan", "Insurance_Plan", "Plan_Description",
        "Benefit_Plan", "Product_Name", "Plan_Type_Name",
        "plan name", "PLAN", "PlanName", "ins_plan",
    ],
    "hospital_name": [
        "Hospital_Name", "Facility_Name", "Hospital", "Provider_Name",
        "Organization_Name", "Facility", "Institution",
        "hospital name", "HOSPITAL", "FacilityName",
    ],
}

# Common payer name variants (for payer normalization)
PAYER_ALIASES = {
    "Blue Cross Blue Shield": [
        "BCBS", "Blue Cross", "BlueCross", "BC/BS", "BCBS_PPO",
        "Blue_Cross_Blue_Shield", "BlueCrossBlueShield", "BCBS PPO",
        "Blue Cross Blue Shield PPO", "BCBS_HMO", "BCBS HMO",
    ],
    "Aetna": ["AETNA", "Aetna_PPO", "Aetna PPO", "Aetna_HMO", "Aetna HMO"],
    "UnitedHealthcare": [
        "United", "UHC", "United_Healthcare", "UnitedHealth",
        "United Healthcare", "UHC_PPO", "UHC PPO", "United_Health_Care",
    ],
    "Cigna": ["CIGNA", "Cigna_PPO", "Cigna PPO", "Cigna_HMO", "Cigna HMO"],
    "Humana": ["HUMANA", "Humana_PPO", "Humana PPO", "Humana_HMO"],
    "Medicare": ["MEDICARE", "Medicare_A", "Medicare_B", "CMS_Medicare"],
    "Medicaid": ["MEDICAID", "State_Medicaid", "Medicaid_FFS", "Medicaid_MCO"],
    "Kaiser Permanente": ["Kaiser", "KAISER", "Kaiser_Permanente"],
    "Anthem": ["ANTHEM", "Anthem_PPO", "Anthem_BCBS", "Anthem Blue Cross"],
    "Molina": ["MOLINA", "Molina_Healthcare", "Molina Healthcare"],
}

# Common procedures for price comparison (CPT codes)
BENCHMARK_PROCEDURES = {
    "73721": ("MRI Knee without contrast", 400, 4800),
    "99213": ("Office visit level 3", 75, 350),
    "99214": ("Office visit level 4", 100, 500),
    "99215": ("Office visit level 5", 150, 700),
    "71046": ("Chest X-ray 2 views", 40, 500),
    "93000": ("Electrocardiogram (ECG)", 25, 250),
    "80053": ("Comprehensive metabolic panel", 15, 200),
    "85025": ("Complete blood count (CBC)", 10, 150),
    "27447": ("Total knee replacement (DRG)", 15000, 120000),
    "59400": ("Vaginal delivery", 5000, 30000),
    "43239": ("Upper GI endoscopy with biopsy", 1000, 8000),
    "29881": ("Knee arthroscopy", 3000, 25000),
    "70553": ("MRI Brain with/without contrast", 500, 6000),
    "74177": ("CT Abdomen/Pelvis with contrast", 300, 4000),
    "90834": ("Psychotherapy 45 minutes", 80, 300),
}

# Fuzzy matching thresholds
FUZZY_MATCH_THRESHOLD = 70  # Levenshtein ratio threshold
TFIDF_SIMILARITY_THRESHOLD = 0.4  # Cosine similarity threshold
AUTO_MAP_CONFIDENCE = 0.75  # Combined confidence for auto-mapping
