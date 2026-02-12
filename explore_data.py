import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print("=" * 80)
print("FAKE JOB POSTING DETECTION - DATA EXPLORATION")
print("=" * 80)

# Load dataset
df = pd.read_csv('data/fake_job_postings.csv')

print(f"\n✅ Dataset loaded: {len(df):,} job postings\n")

# ============================================================================
# 1. BASIC INFORMATION
# ============================================================================
print("=" * 80)
print("1. DATASET OVERVIEW")
print("=" * 80)

print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn Names and Data Types:")
print(df.dtypes)

print(f"\n\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# 2. TARGET VARIABLE ANALYSIS (FRAUDULENT)
# ============================================================================
print("\n" + "=" * 80)
print("2. TARGET VARIABLE: FRAUDULENT")
print("=" * 80)

fraud_counts = df['fraudulent'].value_counts()
print(f"\nClass Distribution:")
print(f"  Real Jobs (0): {fraud_counts[0]:>6,} ({fraud_counts[0]/len(df)*100:>5.2f}%)")
print(f"  Fake Jobs (1): {fraud_counts[1]:>6,} ({fraud_counts[1]/len(df)*100:>5.2f}%)")

print(f"\n⚠️  Class Imbalance Ratio: 1:{fraud_counts[0]//fraud_counts[1]}")
print(f"    (For every 1 fake job, there are {fraud_counts[0]//fraud_counts[1]} real jobs)")

# ============================================================================
# 3. MISSING VALUES ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("3. MISSING VALUES ANALYSIS")
print("=" * 80)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
}).sort_values('Percentage', ascending=False)

print("\nColumns with Missing Data:")
print(missing_df[missing_df['Missing Count'] > 0])

# ============================================================================
# 4. TEXT FEATURES EXAMINATION
# ============================================================================
print("\n" + "=" * 80)
print("4. TEXT FEATURES ANALYSIS")
print("=" * 80)

text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']

print("\nText Length Statistics:")
for col in text_columns:
    df[f'{col}_length'] = df[col].fillna('').str.len()
    avg_real = df[df['fraudulent']==0][f'{col}_length'].mean()
    avg_fake = df[df['fraudulent']==1][f'{col}_length'].mean()
    
    print(f"\n  {col.upper()}:")
    print(f"    Real jobs: {avg_real:.0f} characters")
    print(f"    Fake jobs: {avg_fake:.0f} characters")
    print(f"    Difference: {abs(avg_real - avg_fake):.0f} characters")

# ============================================================================
# 5. SAMPLE DATA INSPECTION
# ============================================================================
print("\n" + "=" * 80)
print("5. SAMPLE JOB POSTINGS")
print("=" * 80)

print("\n📄 REAL JOB EXAMPLE (fraudulent=0):")
print("-" * 80)
real_job = df[df['fraudulent'] == 0].iloc[0]
print(f"Title: {real_job['title']}")
print(f"Location: {real_job['location']}")
print(f"Company Profile: {str(real_job['company_profile'])[:200]}...")
print(f"Description: {str(real_job['description'])[:200]}...")

print("\n🚨 FAKE JOB EXAMPLE (fraudulent=1):")
print("-" * 80)
fake_job = df[df['fraudulent'] == 1].iloc[0]
print(f"Title: {fake_job['title']}")
print(f"Location: {fake_job['location']}")
print(f"Company Profile: {str(fake_job['company_profile'])[:200]}...")
print(f"Description: {str(fake_job['description'])[:200]}...")

# ============================================================================
# 6. CATEGORICAL FEATURES ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("6. CATEGORICAL FEATURES")
print("=" * 80)

categorical_cols = ['employment_type', 'required_experience', 'required_education', 'industry']

for col in categorical_cols:
    if col in df.columns:
        print(f"\n{col.upper()}:")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Top 3 values:")
        top_values = df[col].value_counts().head(3)
        for val, count in top_values.items():
            print(f"    - {val}: {count:,} ({count/len(df)*100:.1f}%)")

# ============================================================================
# 7. BINARY FEATURES ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("7. BINARY FLAGS (telecommuting, has_company_logo, has_questions)")
print("=" * 80)

binary_cols = ['telecommuting', 'has_company_logo', 'has_questions']

for col in binary_cols:
    real_pct = (df[df['fraudulent']==0][col].sum() / len(df[df['fraudulent']==0]) * 100)
    fake_pct = (df[df['fraudulent']==1][col].sum() / len(df[df['fraudulent']==1]) * 100)
    
    print(f"\n{col.upper()}:")
    print(f"  Real jobs with {col}: {real_pct:.1f}%")
    print(f"  Fake jobs with {col}: {fake_pct:.1f}%")
    print(f"  Difference: {abs(real_pct - fake_pct):.1f}%")

print("\n" + "=" * 80)
print("✅ DATA EXPLORATION COMPLETE!")
print("=" * 80)
print("\nKey Findings:")
print("  1. Severe class imbalance (95% real, 5% fake)")
print("  2. Many text features with missing values")
print("  3. Fake jobs have different text length patterns")
print("  4. Binary features show different distributions")
print("\nNext Step: Run 'python visualize_data.py' to create visualizations")
print("=" * 80)