import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STAGE 4: TOKENIZATION & PADDING")
print("=" * 80)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
MAX_WORDS = 10000      # Maximum vocabulary size (top 10,000 most common words)
MAX_LEN = 200          # Maximum sequence length (padding/truncating)
TEST_SIZE = 0.2        # 20% for testing, 80% for training
RANDOM_STATE = 42      # For reproducibility

print("\n⚙️  CONFIGURATION:")
print(f"   Max vocabulary size: {MAX_WORDS:,} words")
print(f"   Max sequence length: {MAX_LEN} tokens")
print(f"   Train/Test split: {int((1-TEST_SIZE)*100)}% / {int(TEST_SIZE*100)}%")

# ============================================================================
# STEP 1: LOAD PREPROCESSED DATA
# ============================================================================
print("\n[1/6] Loading preprocessed data...")
df = pd.read_csv('data/preprocessed_data.csv')
print(f"✅ Loaded {len(df):,} preprocessed job postings")

# Separate features and labels
texts = df['cleaned_text'].values
labels = df['fraudulent'].values

print(f"   Text samples: {len(texts):,}")
print(f"   Labels: {len(labels):,}")

# ============================================================================
# STEP 2: CREATE TOKENIZER
# ============================================================================
print("\n[2/6] Creating tokenizer and building vocabulary...")
print("   This will take 30-60 seconds...")

# Initialize tokenizer
tokenizer = Tokenizer(
    num_words=MAX_WORDS,           # Keep only top 10,000 words
    oov_token='<OOV>',             # Out-of-vocabulary token for unknown words
    lower=True,                     # Convert to lowercase (already done, but safe)
    char_level=False                # Word-level tokenization
)

# Build vocabulary from training texts
tokenizer.fit_on_texts(texts)

# Get statistics
vocab_size = len(tokenizer.word_index)
actual_vocab = min(MAX_WORDS, vocab_size)

print(f"✅ Tokenizer created")
print(f"   Total unique words found: {vocab_size:,}")
print(f"   Vocabulary size used: {actual_vocab:,}")
print(f"   Words ignored (beyond {MAX_WORDS:,}): {max(0, vocab_size - MAX_WORDS):,}")

# ============================================================================
# STEP 3: SHOW TOKENIZATION EXAMPLE
# ============================================================================
print("\n" + "=" * 80)
print("📝 TOKENIZATION EXAMPLE")
print("=" * 80)

# Get top 20 most common words
print("\nTop 20 Most Common Words:")
word_freq = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)[:20]
for i, (word, freq) in enumerate(word_freq, 1):
    print(f"   {i:2d}. '{word}' → appears {freq:,} times")

# Example conversion
example_text = texts[0]
example_sequence = tokenizer.texts_to_sequences([example_text])[0]

print(f"\n📄 Example Job Posting (first 100 chars):")
print(f"   {example_text[:100]}...")

print(f"\n🔢 Converted to Token IDs (first 20 tokens):")
print(f"   {example_sequence[:20]}")

print(f"\n📊 Sequence Statistics:")
print(f"   Length: {len(example_sequence)} tokens")

# ============================================================================
# STEP 4: CONVERT ALL TEXTS TO SEQUENCES
# ============================================================================
print("\n" + "=" * 80)
print("[3/6] Converting all texts to sequences...")

sequences = tokenizer.texts_to_sequences(texts)

print(f"✅ Converted {len(sequences):,} texts to sequences")

# Statistics about sequence lengths
seq_lengths = [len(seq) for seq in sequences]
print(f"\n📊 Sequence Length Statistics:")
print(f"   Minimum: {min(seq_lengths)} tokens")
print(f"   Maximum: {max(seq_lengths)} tokens")
print(f"   Average: {np.mean(seq_lengths):.1f} tokens")
print(f"   Median: {np.median(seq_lengths):.1f} tokens")

# Check how many sequences will be truncated
truncated = sum(1 for length in seq_lengths if length > MAX_LEN)
print(f"\n⚠️  Sequences longer than {MAX_LEN}: {truncated:,} ({truncated/len(sequences)*100:.1f}%)")
print(f"   These will be truncated to {MAX_LEN} tokens")

# ============================================================================
# STEP 5: PAD SEQUENCES
# ============================================================================
print("\n[4/6] Padding sequences to uniform length...")

# Pad sequences
X = pad_sequences(
    sequences,
    maxlen=MAX_LEN,
    padding='post',      # Add zeros at the end
    truncating='post'    # Remove from the end if too long
)

print(f"✅ Padding complete")
print(f"   Final shape: {X.shape}")
print(f"   ({X.shape[0]:,} samples × {X.shape[1]} tokens)")

# Show padding example
print(f"\n📝 PADDING EXAMPLE:")
print(f"   Original sequence length: {len(example_sequence)} tokens")
print(f"   After padding: {MAX_LEN} tokens")
print(f"   First 30 tokens: {X[0][:30]}")
print(f"   Last 10 tokens: {X[0][-10:]} (zeros added)")

# ============================================================================
# STEP 6: TRAIN-TEST SPLIT
# ============================================================================
print("\n[5/6] Splitting into train and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels  # Maintain class distribution in both sets
)

print(f"✅ Data split complete")
print(f"\n📊 Training Set:")
print(f"   Samples: {len(X_train):,}")
print(f"   Real jobs: {(y_train == 0).sum():,} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
print(f"   Fake jobs: {(y_train == 1).sum():,} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")

print(f"\n📊 Test Set:")
print(f"   Samples: {len(X_test):,}")
print(f"   Real jobs: {(y_test == 0).sum():,} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")
print(f"   Fake jobs: {(y_test == 1).sum():,} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")

# ============================================================================
# STEP 7: SAVE EVERYTHING
# ============================================================================
print("\n[6/6] Saving tokenizer and processed data...")

# Save tokenizer
with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print(f"✅ Saved tokenizer: models/tokenizer.pkl")

# Save processed data
np.save('models/X_train.npy', X_train)
np.save('models/X_test.npy', X_test)
np.save('models/y_train.npy', y_train)
np.save('models/y_test.npy', y_test)

print(f"✅ Saved training data: models/X_train.npy, models/y_train.npy")
print(f"✅ Saved test data: models/X_test.npy, models/y_test.npy")

# Save configuration
config = {
    'max_words': MAX_WORDS,
    'max_len': MAX_LEN,
    'vocab_size': actual_vocab,
    'total_samples': len(X),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'train_fake': int((y_train == 1).sum()),
    'test_fake': int((y_test == 1).sum())
}

with open('models/config.pkl', 'wb') as f:
    pickle.dump(config, f)
print(f"✅ Saved configuration: models/config.pkl")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ TOKENIZATION & PADDING COMPLETE!")
print("=" * 80)

print("\n📊 SUMMARY:")
print(f"   Vocabulary size: {actual_vocab:,} words")
print(f"   Sequence length: {MAX_LEN} tokens")
print(f"   Training samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")
print(f"   Data shape: ({X.shape[0]:,}, {X.shape[1]})")

print("\n💾 SAVED FILES:")
print("   ✅ models/tokenizer.pkl - Tokenizer object")
print("   ✅ models/X_train.npy - Training features")
print("   ✅ models/X_test.npy - Test features")
print("   ✅ models/y_train.npy - Training labels")
print("   ✅ models/y_test.npy - Test labels")
print("   ✅ models/config.pkl - Configuration")

print("\n🚀 NEXT STEP: Build Deep Learning Model")
print("   Run: python build_model.py")
print("=" * 80)