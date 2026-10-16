from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_CSV = BASE_DIR / "data" / "raw" / "startup_funding.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CLEANED_CSV = PROCESSED_DIR / "startup_funding_cleaned.csv"
QUALITY_REPORT = PROCESSED_DIR / "data_quality_report.json"
DB_PATH = PROCESSED_DIR / "startup_funding.db"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
CLUSTERS_PATH = PROCESSED_DIR / "cluster_assignments.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Approximate USD -> INR rate used only for display in the dashboard. Held in one
# place so it can be updated without touching prediction logic.
USD_TO_INR = 75
USD_TO_INR_CRORE_DIVISOR = 1e7

# The dataset spans January 2015 - January 2020. The predictor accepts a slightly
# wider window for usability, but anything outside it is rejected outright rather
# than silently extrapolating a model that has never seen that period.
MIN_PREDICTABLE_YEAR = 2015
MAX_PREDICTABLE_YEAR = 2025

# Inner cross-validation scheme for hyperparameter search. TimeSeriesSplit is
# required: the outer split is temporal, so a random KFold inside it would let a
# fold train on 2017 and validate on 2016.
CV_N_SPLITS = 4

CITY_NORMALIZATION = {
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "bangalore urban": "Bengaluru",
    "new delhi": "New Delhi",
    "delhi": "New Delhi",
    "gurgaon": "Gurugram",
    "gurugram": "Gurugram",
    "mumbai": "Mumbai",
    "pune": "Pune",
    "chennai": "Chennai",
    "hyderabad": "Hyderabad",
    "noida": "Noida",
    "ahmedabad": "Ahmedabad",
    "jaipur": "Jaipur",
    "kolkata": "Kolkata",
    "indore": "Indore",
    "kochi": "Kochi",
    "cochin": "Kochi",
    "chandigarh": "Chandigarh",
    "lucknow": "Lucknow",
    "bhopal": "Bhopal",
}

INDUSTRY_NORMALIZATION = {
    "e-commerce": "E-Commerce",
    "ecommerce": "E-Commerce",
    "e - commerce": "E-Commerce",
    "fintech": "FinTech",
    "fin tech": "FinTech",
    "health and wellness": "Health and Wellness",
    "food and beverage": "Food and Beverage",
    "food & beverage": "Food and Beverage",
    "consumer goods": "Consumer Goods",
    "consumer technology": "Consumer Technology",
    "information technology": "Information Technology",
    "technology": "Technology",
    "saas": "SaaS",
    "b2b marketing": "B2B",
    "education": "Education",
    "transportation": "Transportation",
    "logistics": "Logistics",
    "hospitality": "Hospitality",
    "fashion and apparel": "Fashion and Apparel",
    "retail": "Retail",
    "video games": "Gaming",
    "video": "Video",
    "advertising, marketing": "Advertising & Marketing",
    "advertising": "Advertising & Marketing",
    "marketing": "Advertising & Marketing",
    "accounting": "Accounting",
    "aerospace": "Aerospace",
    "iot": "IoT",
    "finance": "Finance",
    "real estate": "Real Estate",
    "agritech": "Agritech",
    "automobile": "Automobile",
    "energy": "Energy",
    "media": "Media",
    "telecom": "Telecom",
    "travel": "Travel",
    "sports": "Sports",
    "pharmaceuticals": "Pharmaceuticals",
    "biotechnology": "Biotechnology",
    "nanotechnology": "Nanotechnology",
    "cleantech": "CleanTech",
    "edtech": "EdTech",
    "healthtech": "HealthTech",
    "legaltech": "LegalTech",
    "spacetech": "SpaceTech",
    "autonomous vehicles": "Autonomous Vehicles",
    "robotics": "Robotics",
    "drones": "Drones",
    "fashion": "Fashion",
    "foodtech": "FoodTech",
    "hr tech": "HR Tech",
    "traveltech": "TravelTech",
    "realty": "Real Estate",
    "e-tail": "E-Tail",
    "e-tailers": "E-Tail",
    "e-tailing": "E-Tail",
    "online marketplace": "Online Marketplace",
    "online education": "Online Education",
    "online lending": "Online Lending",
    "online pharmacy": "Online Pharmacy",
    "online travel": "Online Travel",
    "online food delivery": "Online Food Delivery",
    "online grocery": "Online Grocery",
    "online classifieds": "Online Classifieds",
    "online real estate": "Online Real Estate",
    "online insurance": "Online Insurance",
    "online payments": "Online Payments",
    "online gaming": "Online Gaming",
    "online media": "Online Media",
    "online services": "Online Services",
}

INVESTMENT_TYPE_NORMALIZATION = {
    "private equity round": "Private Equity",
    "private equity": "Private Equity",
    "seed round": "Seed",
    "seed funding": "Seed",
    "seed funding round": "Seed",
    "seed investment": "Seed",
    "pre-series a": "Pre-Series A",
    "pre series a": "Pre-Series A",
    "series a": "Series A",
    "series b": "Series B",
    "series c": "Series C",
    "series d": "Series D",
    "series e": "Series E",
    "series f": "Series F",
    "series g": "Series G",
    "series h": "Series H",
    "series i": "Series I",
    "debt funding": "Debt Funding",
    "venture": "Venture",
    "funding round": "Funding Round",
    "corporate round": "Corporate Round",
    "maiden round": "Maiden Round",
    "single venture": "Single Venture",
    "angel funding": "Angel",
    "angel round": "Angel",
    "pre-seed": "Pre-Seed",
    "pre seed": "Pre-Seed",
    "bridge": "Bridge",
    "debt": "Debt Funding",
    "equity": "Equity",
    "grant": "Grant",
    "crowdfunding": "Crowdfunding",
    "initial coin offering": "ICO",
    "ico": "ICO",
    "secondary market": "Secondary Market",
    "post-ipo equity": "Post-IPO Equity",
    "post-ipo debt": "Post-IPO Debt",
}
