import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STAGE 3: TEXT PREPROCESSING")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/7] Loading dataset...")
df = pd.read_csv('data/fake_job_postings.csv')
print(f"✅ Loaded {len(df):,} job postings")

# ============================================================================
# STEP 2: COMBINE TEXT FEATURES
# ============================================================================
print("\n[2/7] Combining text features...")
print("   Merging: title + company_profile + description + requirements + benefits")

# Fill missing values with empty string
text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
for col in text_columns:
    df[col] = df[col].fillna('')

# Combine all text into one column
df['combined_text'] = (
    df['title'] + ' ' +
    df['company_profile'] + ' ' +
    df['description'] + ' ' +
    df['requirements'] + ' ' +
    df['benefits']
)

print(f"✅ Combined text created")
print(f"   Average length: {df['combined_text'].str.len().mean():.0f} characters")

# ============================================================================
# STEP 3: TEXT CLEANING FUNCTION
# ============================================================================
print("\n[3/7] Defining text cleaning function...")

def clean_text(text):

    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove very short words
    text = ' '.join([word for word in text.split() if len(word) > 1])
    
    return text

print("✅ Cleaning function ready")

# ============================================================================
# STEP 4: APPLY CLEANING
# ============================================================================
print("\n[4/7] Cleaning text data...")
print("   This may take 1-2 minutes...")

df['cleaned_text'] = df['combined_text'].apply(clean_text)

print("✅ Text cleaning complete")

# Show before/after example
print("\n📝 CLEANING EXAMPLE:")
print("-" * 80)
print("BEFORE:")
print(df['combined_text'].iloc[0][:200])
print("\nAFTER:")
print(df['cleaned_text'].iloc[0][:200])
print("-" * 80)

# ============================================================================
# STEP 5: REMOVE SHORT TEXTS
# ============================================================================
print("\n[5/7] Removing very short texts...")
initial_count = len(df)

# Remove rows where cleaned text is less than 50 characters
df = df[df['cleaned_text'].str.len() >= 50].copy()

removed_count = initial_count - len(df)
print(f"✅ Removed {removed_count:,} rows with text < 50 characters")
print(f"   Remaining: {len(df):,} job postings")

# ============================================================================
# STEP 6: VERIFY CLASS BALANCE AFTER CLEANING
# ============================================================================
print("\n[6/7] Verifying class distribution...")
fraud_counts = df['fraudulent'].value_counts()
print(f"\nAfter preprocessing:")
print(f"  Real Jobs (0): {fraud_counts[0]:>6,} ({fraud_counts[0]/len(df)*100:>5.2f}%)")
print(f"  Fake Jobs (1): {fraud_counts[1]:>6,} ({fraud_counts[1]/len(df)*100:>5.2f}%)")

# ============================================================================
# STEP 7: SAVE PREPROCESSED DATA
# ============================================================================
print("\n[7/7] Saving preprocessed data...")

# Keep only necessary columns
df_final = df[['cleaned_text', 'fraudulent']].copy()

# Save to CSV
df_final.to_csv('data/preprocessed_data.csv', index=False)
print(f"✅ Saved to: data/preprocessed_data.csv")

# Save statistics
stats = {
    'total_samples': len(df_final),
    'real_jobs': int(fraud_counts[0]),
    'fake_jobs': int(fraud_counts[1]),
    'avg_text_length': float(df_final['cleaned_text'].str.len().mean()),
    'max_text_length': int(df_final['cleaned_text'].str.len().max()),
    'min_text_length': int(df_final['cleaned_text'].str.len().min())
}

with open('output/preprocessing_stats.pkl', 'wb') as f:
    pickle.dump(stats, f)

print(f"✅ Saved statistics to: output/preprocessing_stats.pkl")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ PREPROCESSING COMPLETE!")
print("=" * 80)

print("\n📊 FINAL STATISTICS:")
print(f"  Total samples: {stats['total_samples']:,}")
print(f"  Real jobs: {stats['real_jobs']:,}")
print(f"  Fake jobs: {stats['fake_jobs']:,}")
print(f"  Average text length: {stats['avg_text_length']:.0f} characters")
print(f"  Max text length: {stats['max_text_length']:,} characters")
print(f"  Min text length: {stats['min_text_length']:,} characters")

print("\n📁 OUTPUT FILES:")
print("  ✅ data/preprocessed_data.csv - Cleaned dataset")
print("  ✅ output/preprocessing_stats.pkl - Statistics")

print("\n🚀 NEXT STEP: Tokenization & Padding")
print("   Run: python tokenize.py")
print("=" * 80)