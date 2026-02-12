import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall  # ← FIXED: Import metric classes
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STAGE 5: BUILD & TRAIN BiLSTM MODEL")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA AND CONFIGURATION
# ============================================================================
print("\n[1/6] Loading preprocessed data...")

# Load training and test data
X_train = np.load('models/X_train.npy')
y_train = np.load('models/y_train.npy')
X_test = np.load('models/X_test.npy')
y_test = np.load('models/y_test.npy')

# Load configuration
with open('models/config.pkl', 'rb') as f:
    config = pickle.load(f)

MAX_WORDS = config['max_words']
MAX_LEN = config['max_len']

print(f"✅ Data loaded successfully")
print(f"   Training samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")
print(f"   Vocabulary size: {MAX_WORDS:,}")
print(f"   Sequence length: {MAX_LEN}")

# ============================================================================
# STEP 2: COMPUTE CLASS WEIGHTS (Handle Imbalance)
# ============================================================================
print("\n[2/6] Computing class weights to handle imbalance...")

# Calculate class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

print(f"✅ Class weights calculated")
print(f"   Real jobs (0): {class_weight_dict[0]:.3f}")
print(f"   Fake jobs (1): {class_weight_dict[1]:.3f}")
print(f"   → Fake jobs weighted {class_weight_dict[1]/class_weight_dict[0]:.1f}x more")
print(f"   This forces model to pay more attention to minority class!")

# ============================================================================
# STEP 3: BUILD BiLSTM MODEL
# ============================================================================
print("\n[3/6] Building BiLSTM model architecture...")

EMBEDDING_DIM = 128    # Size of word embeddings
LSTM_UNITS_1 = 64      # First BiLSTM layer units
LSTM_UNITS_2 = 32      # Second BiLSTM layer units
DENSE_UNITS = 64       # Dense layer units
DROPOUT_RATE_1 = 0.5   # First dropout rate
DROPOUT_RATE_2 = 0.3   # Second dropout rate

model = Sequential([
    # Embedding Layer: Converts token IDs to dense vectors
    Embedding(
        input_dim=MAX_WORDS,           # Vocabulary size
        output_dim=EMBEDDING_DIM,      # Embedding dimension
        input_length=MAX_LEN,          # Sequence length
        name='embedding_layer'
    ),
    
    # First BiLSTM Layer: Reads sequence forward and backward
    Bidirectional(
        LSTM(LSTM_UNITS_1, return_sequences=True),  # return_sequences=True for stacking
        name='bilstm_layer_1'
    ),
    
    # Second BiLSTM Layer: Further processing
    Bidirectional(
        LSTM(LSTM_UNITS_2),  # return_sequences=False (default) for final output
        name='bilstm_layer_2'
    ),
    
    # Dropout: Prevents overfitting
    Dropout(DROPOUT_RATE_1, name='dropout_1'),
    
    # Dense Layer: Pattern recognition
    Dense(DENSE_UNITS, activation='relu', name='dense_layer'),
    
    # Dropout: More regularization
    Dropout(DROPOUT_RATE_2, name='dropout_2'),
    
    # Output Layer: Binary classification (Fake=1, Real=0)
    Dense(1, activation='sigmoid', name='output_layer')
])

print(f"✅ Model architecture created")

# Display model summary
print("\n" + "=" * 80)
print("MODEL ARCHITECTURE SUMMARY")
print("=" * 80)
model.summary()

# ============================================================================
# STEP 4: COMPILE MODEL
# ============================================================================
print("\n[4/6] Compiling model...")

# Optimizer
optimizer = Adam(learning_rate=0.001)

# FIXED: Use metric classes instead of strings
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',      # Binary classification loss
    metrics=[
        'accuracy',                  # Overall accuracy
        Precision(name='precision'), # TP / (TP + FP)
        Recall(name='recall')        # TP / (TP + FN) - Important for fraud detection!
    ]
)

print(f"✅ Model compiled")
print(f"   Optimizer: Adam (lr=0.001)")
print(f"   Loss: Binary Crossentropy")
print(f"   Metrics: Accuracy, Precision, Recall")

# ============================================================================
# STEP 5: SETUP CALLBACKS
# ============================================================================
print("\n[5/6] Setting up training callbacks...")

# Early Stopping: Stop if validation loss doesn't improve
early_stop = EarlyStopping(
    monitor='val_loss',              # Watch validation loss
    patience=3,                      # Wait 3 epochs before stopping
    restore_best_weights=True,       # Restore best model
    verbose=1
)

# Model Checkpoint: Save best model
checkpoint = ModelCheckpoint(
    'models/best_bilstm_model.h5',
    monitor='val_recall',            # Save model with best recall (catch fake jobs!)
    save_best_only=True,
    mode='max',
    verbose=1
)

# Learning Rate Reduction: Reduce LR if stuck
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,                      # Reduce LR by 50%
    patience=2,
    min_lr=0.00001,
    verbose=1
)

print(f"✅ Callbacks configured")
print(f"   - EarlyStopping: patience=3")
print(f"   - ModelCheckpoint: save best model based on recall")
print(f"   - ReduceLROnPlateau: reduce LR if stuck")

# ============================================================================
# STEP 6: TRAIN MODEL
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING BiLSTM MODEL")
print("=" * 80)
print("\n🚀 Starting training... This will take 10-15 minutes")
print("   (Grab a coffee! ☕)\n")

# Training parameters
EPOCHS = 20
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=VALIDATION_SPLIT,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,    # Apply class weights
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE!")
print("=" * 80)

# ============================================================================
# STEP 7: PLOT TRAINING HISTORY
# ============================================================================
print("\n[6/6] Creating training history plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('BiLSTM Training History', fontsize=16, fontweight='bold')

# Plot 1: Loss
axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 0].set_title('Model Loss', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Accuracy
axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0, 1].set_title('Model Accuracy', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Precision
axes[1, 0].plot(history.history['precision'], label='Training Precision', linewidth=2)
axes[1, 0].plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
axes[1, 0].set_title('Model Precision', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Recall
axes[1, 1].plot(history.history['recall'], label='Training Recall', linewidth=2)
axes[1, 1].plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
axes[1, 1].set_title('Model Recall (Most Important!)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/training_history.png', dpi=300, bbox_inches='tight')
print(f"✅ Training plots saved: output/training_history.png")
plt.show()

# ============================================================================
# SAVE FINAL MODEL
# ============================================================================
model.save('models/final_bilstm_model.h5')
print(f"✅ Final model saved: models/final_bilstm_model.h5")

# Save training history
with open('output/training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print(f"✅ Training history saved: output/training_history.pkl")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ MODEL BUILDING & TRAINING COMPLETE!")
print("=" * 80)

print("\n📊 TRAINING SUMMARY:")
print(f"   Total epochs run: {len(history.history['loss'])}")
print(f"   Best validation loss: {min(history.history['val_loss']):.4f}")
print(f"   Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"   Best validation recall: {max(history.history['val_recall']):.4f}")

print("\n💾 SAVED FILES:")
print("   ✅ models/best_bilstm_model.h5 - Best model (highest recall)")
print("   ✅ models/final_bilstm_model.h5 - Final model after all epochs")
print("   ✅ output/training_history.png - Training plots")
print("   ✅ output/training_history.pkl - History data")

print("\n🚀 NEXT STEP: Evaluate Model Performance")
print("   Run: python evaluate_model.py")
print("=" * 80)