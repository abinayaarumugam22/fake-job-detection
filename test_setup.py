import sys
print("=" * 60)
print("SETUP VERIFICATION")
print("=" * 60)
print(f"Python version: {sys.version.split()[0]}")

# Check TensorFlow
try:
    import tensorflow as tf
    print(f"✅ TensorFlow: {tf.__version__}")
except ImportError:
    print("❌ TensorFlow: NOT INSTALLED")

# Check other libraries
libraries = {
    'pandas': 'pd',
    'numpy': 'np', 
    'sklearn': 'scikit-learn',
    'nltk': 'nltk',
    'imblearn': 'imbalanced-learn',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'flask': 'flask'
}

for module, name in libraries.items():
    try:
        lib = __import__(module)
        version = getattr(lib, '__version__', 'installed')
        print(f"✅ {name}: {version}")
    except ImportError:
        print(f"❌ {name}: NOT INSTALLED")

# Check dataset
print("\n" + "=" * 60)
print("DATASET CHECK")
print("=" * 60)

try:
    import pandas as pd
    df = pd.read_csv('data/fake_job_postings.csv')
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   Total job postings: {len(df):,}")
    print(f"   Number of features: {len(df.columns)}")
    
    print(f"\n📊 Class Distribution:")
    counts = df['fraudulent'].value_counts()
    print(f"   Real jobs (0): {counts[0]:,} ({counts[0]/len(df)*100:.1f}%)")
    print(f"   Fake jobs (1): {counts[1]:,} ({counts[1]/len(df)*100:.1f}%)")
    
    print(f"\n📝 Sample columns:")
    for col in df.columns[:8]:
        print(f"   - {col}")
    print(f"   ... and {len(df.columns)-8} more")
    
    print("\n✅ ALL SETUP COMPLETE! Ready to start coding! 🚀")
    
except FileNotFoundError:
    print("❌ Dataset file NOT FOUND!")
    print("   Expected location: data/fake_job_postings.csv")
    print("   Please download from Kaggle and place it there.")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")

print("=" * 60)