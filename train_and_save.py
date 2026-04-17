import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

def main():
    # 1. Load the raw dataset
    input_file = 'India_Flood_Inventory_Cleaned.csv'
    
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found in this directory!")
        return

    print("--- Step 1: Loading and Decoding Dates ---")
    df = pd.read_csv(input_file)
    
    # Convert dates and extract time features
    df['Start Date'] = pd.to_datetime(df['Start Date'], dayfirst=True, errors='coerce')
    df['Start_Year'] = df['Start Date'].dt.year.fillna(2000).astype(int)
    df['Start_Month'] = df['Start Date'].dt.month.fillna(6).astype(int)

    # 2. Feature Engineering
    print("--- Step 2: Feature Engineering ---")
    SEASON_MAP = {
        1:"winter", 2:"winter", 3:"pre_monsoon", 4:"pre_monsoon", 5:"pre_monsoon", 
        6:"monsoon", 7:"monsoon", 8:"monsoon", 9:"monsoon", 10:"post_monsoon", 
        11:"post_monsoon", 12:"winter"
    }
    df['Season'] = df['Start_Month'].map(SEASON_MAP)

    def duration_to_bin(days):
        if days <= 1: return "single_day"
        if days <= 3: return "2_3days"
        if days <= 7: return "week"
        if days <= 14: return "2weeks"
        return "prolonged"
    df['Duration_Bin'] = df['Duration(Days)'].apply(duration_to_bin)

    # 3. Create Target (Risk Levels) with Jitter Fix
    print("--- Step 3: Generating Risk Levels ---")
    # Risk Score = (Fatalities * 2) + Duration
    df['risk_score'] = (df['Human fatality'].fillna(0) * 2) + df['Duration(Days)'].fillna(1)
    
    # ADD JITTER: This ensures all bin edges are unique for pd.qcut
    df['risk_score_jittered'] = df['risk_score'] + np.random.normal(0, 1e-6, len(df))
    
    df['Risk_Level'] = pd.qcut(
        df['risk_score_jittered'], 
        q=4, 
        labels=["Low", "Medium", "High", "Extreme"]
    )

    # 4. Initialize and fit Encoders
    print("--- Step 4: Encoding Categorical Data ---")
    encoders = {
        'le_cause': LabelEncoder(),
        'le_season': LabelEncoder(),
        'le_state': LabelEncoder(),
        'le_durbin': LabelEncoder(),
        'le_target': LabelEncoder()
    }

    df['cause_enc'] = encoders['le_cause'].fit_transform(df['Main Cause'].astype(str))
    df['season_enc'] = encoders['le_season'].fit_transform(df['Season'].astype(str))
    df['state_enc'] = encoders['le_state'].fit_transform(df['State'].astype(str))
    df['durbin_enc'] = encoders['le_durbin'].fit_transform(df['Duration_Bin'].astype(str))
    y_enc = encoders['le_target'].fit_transform(df['Risk_Level'])

    # 5. Train Model
    print("--- Step 5: Training Random Forest Model ---")
    X = df[['Start_Month', 'Start_Year', 'Duration(Days)', 'cause_enc', 'season_enc', 'state_enc', 'durbin_enc']]
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    # 6. Save Artifacts for Backend
    print("--- Step 6: Saving Artifacts ---")
    for name, le in encoders.items():
        joblib.dump(le, f"{name}.pkl")
    joblib.dump(model, "flood_model.pkl")

    # 7. Create model_meta.json
    meta = {
        "total_records": len(df),
        "year_range": f"{int(df['Start_Year'].min())}-{int(df['Start_Year'].max())}",
        "accuracy": float(accuracy),
        "risk_distribution": df['Risk_Level'].value_counts().to_dict(),
        "top_states": df['State'].value_counts().head(10).to_dict(),
        "state_classes": list(encoders['le_state'].classes_),
        "cause_classes": list(encoders['le_cause'].classes_),
        "classes": ["Low", "Medium", "High", "Extreme"],
        "feature_importances": dict(zip(X.columns, model.feature_importances_.tolist())),
        "model_comparison": {"Random Forest": float(accuracy), "Baseline": 0.50}
    }

    with open("model_meta.json", "w") as f:
        json.dump(meta, f)

    print(f"✅ DONE! Accuracy: {round(accuracy*100, 2)}%")
    print("All files (flood_model.pkl, model_meta.json, le_*.pkl) are ready for main.py")

if __name__ == "__main__":
    main()
