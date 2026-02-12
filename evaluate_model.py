import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STAGE 6: MODEL EVALUATION")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD MODEL AND TEST DATA
# ============================================================================
print("\n[1/5] Loading best model and test data...")

# Load the best model (saved during training)
model = load_model('models/best_bilstm_model.h5')
print("✅ Model loaded: models/best_bilstm_model.h5")

# Load test data
X_test = np.load('models/X_test.npy')
y_test = np.load('models/y_test.npy')

print(f"✅ Test data loaded")
print(f"   Test samples: {len(X_test):,}")
print(f"   Real jobs: {(y_test == 0).sum():,} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")
print(f"   Fake jobs: {(y_test == 1).sum():,} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")

# ============================================================================
# STEP 2: MAKE PREDICTIONS
# ============================================================================
print("\n[2/5] Making predictions on test set...")

# Get probability predictions
y_pred_prob = model.predict(X_test, verbose=0)

# Convert probabilities to binary predictions (threshold = 0.5)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print(f"✅ Predictions complete")
print(f"   Predicted as Real: {(y_pred == 0).sum():,}")
print(f"   Predicted as Fake: {(y_pred == 1).sum():,}")

# ============================================================================
# STEP 3: CALCULATE METRICS
# ============================================================================
print("\n[3/5] Calculating performance metrics...")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Calculate metrics manually for clarity
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_prob)

print("\n" + "=" * 80)
print("📊 PERFORMANCE METRICS")
print("=" * 80)

print(f"\n🎯 Overall Performance:")
print(f"   Accuracy:    {accuracy*100:>6.2f}%  (Overall correctness)")
print(f"   ROC-AUC:     {roc_auc:>6.4f}  (Area under curve)")

print(f"\n🚨 Fake Job Detection (Most Important!):")
print(f"   Recall:      {recall*100:>6.2f}%  (% of fakes caught)")
print(f"   Precision:   {precision*100:>6.2f}%  (% of 'fake' predictions correct)")
print(f"   F1-Score:    {f1_score:>6.4f}  (Harmonic mean)")

print(f"\n✅ Real Job Recognition:")
print(f"   Specificity: {specificity*100:>6.2f}%  (% of real jobs correctly identified)")

print(f"\n📈 Confusion Matrix Breakdown:")
print(f"   True Negatives (TN):  {tn:>5,}  (Real jobs correctly identified)")
print(f"   False Positives (FP): {fp:>5,}  (Real jobs wrongly flagged as fake)")
print(f"   False Negatives (FN): {fn:>5,}  (Fake jobs MISSED - bad!)")
print(f"   True Positives (TP):  {tp:>5,}  (Fake jobs caught - good!)")

print(f"\n💡 Interpretation:")
if recall >= 0.80:
    print(f"   ✅ EXCELLENT: Catching {recall*100:.1f}% of fake jobs!")
elif recall >= 0.70:
    print(f"   ✅ GOOD: Catching {recall*100:.1f}% of fake jobs")
else:
    print(f"   ⚠️  NEEDS IMPROVEMENT: Only catching {recall*100:.1f}% of fake jobs")

if fn > 0:
    print(f"   ⚠️  WARNING: {fn} fake jobs slipped through (out of {tp+fn} total fakes)")
    print(f"      → These {fn} people could get scammed!")

# ============================================================================
# STEP 4: DETAILED CLASSIFICATION REPORT
# ============================================================================
print("\n" + "=" * 80)
print("📋 DETAILED CLASSIFICATION REPORT")
print("=" * 80)

target_names = ['Real Jobs (0)', 'Fake Jobs (1)']
report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
print(report)

# ============================================================================
# STEP 5: CREATE VISUALIZATIONS
# ============================================================================
print("\n[4/5] Creating evaluation visualizations...")

fig = plt.figure(figsize=(18, 12))

# ============================================================================
# PLOT 1: Confusion Matrix
# ============================================================================
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
            annot_kws={'size': 16, 'weight': 'bold'})
ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
ax1.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax1.set_xlabel('Predicted', fontsize=12, fontweight='bold')

# Add percentage annotations
total = np.sum(cm)
for i in range(2):
    for j in range(2):
        percentage = cm[i, j] / total * 100
        ax1.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                ha='center', va='center', fontsize=10, color='gray')

# ============================================================================
# PLOT 2: Normalized Confusion Matrix
# ============================================================================
ax2 = plt.subplot(2, 3, 2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', cbar=False,
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
            annot_kws={'size': 14, 'weight': 'bold'})
ax2.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
ax2.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')

# ============================================================================
# PLOT 3: Metrics Bar Chart
# ============================================================================
ax3 = plt.subplot(2, 3, 3)
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [accuracy, precision, recall, f1_score]
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
bars = ax3.bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black')
ax3.set_ylim(0, 1.1)
ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
ax3.set_title('Performance Metrics', fontsize=14, fontweight='bold', pad=20)
ax3.axhline(y=0.8, color='red', linestyle='--', linewidth=1, alpha=0.5, label='80% Target')
ax3.legend()

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height*100:.1f}%', ha='center', va='bottom', fontweight='bold')

# ============================================================================
# PLOT 4: ROC Curve
# ============================================================================
ax4 = plt.subplot(2, 3, 4)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
ax4.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='#2ecc71')
ax4.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier', alpha=0.5)
ax4.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax4.set_ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
ax4.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=20)
ax4.legend()
ax4.grid(True, alpha=0.3)

# ============================================================================
# PLOT 5: Prediction Distribution
# ============================================================================
ax5 = plt.subplot(2, 3, 5)
fake_probs = y_pred_prob[y_test == 1].flatten()
real_probs = y_pred_prob[y_test == 0].flatten()

ax5.hist(real_probs, bins=50, alpha=0.6, label='Real Jobs', color='#2ecc71', edgecolor='black')
ax5.hist(fake_probs, bins=50, alpha=0.6, label='Fake Jobs', color='#e74c3c', edgecolor='black')
ax5.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
ax5.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax5.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold', pad=20)
ax5.legend()
ax5.grid(True, alpha=0.3)

# ============================================================================
# PLOT 6: Error Analysis
# ============================================================================
ax6 = plt.subplot(2, 3, 6)
error_types = ['True\nNegatives\n(Correct)', 'False\nPositives\n(Error)', 
               'False\nNegatives\n(Error)', 'True\nPositives\n(Correct)']
error_counts = [tn, fp, fn, tp]
error_colors = ['#2ecc71', '#f39c12', '#e74c3c', '#2ecc71']
bars = ax6.bar(error_types, error_counts, color=error_colors, alpha=0.7, edgecolor='black')
ax6.set_ylabel('Count', fontsize=12, fontweight='bold')
ax6.set_title('Prediction Breakdown', fontsize=14, fontweight='bold', pad=20)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('output/model_evaluation.png', dpi=300, bbox_inches='tight')
print(f"✅ Evaluation plots saved: output/model_evaluation.png")
plt.show()

# ============================================================================
# SAVE EVALUATION RESULTS
# ============================================================================
print("\n[5/5] Saving evaluation results...")

# Save predictions
results = {
    'y_test': y_test,
    'y_pred': y_pred,
    'y_pred_prob': y_pred_prob.flatten(),
    'confusion_matrix': cm,
    'metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'specificity': float(specificity),
        'roc_auc': float(roc_auc)
    }
}

with open('output/evaluation_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"✅ Results saved: output/evaluation_results.pkl")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ MODEL EVALUATION COMPLETE!")
print("=" * 80)

print(f"\n🎯 FINAL PERFORMANCE SUMMARY:")
print(f"   ✅ Accuracy: {accuracy*100:.2f}%")
print(f"   ✅ Recall (Fake Detection): {recall*100:.2f}%")
print(f"   ✅ Precision: {precision*100:.2f}%")
print(f"   ✅ F1-Score: {f1_score:.4f}")
print(f"   ✅ ROC-AUC: {roc_auc:.4f}")

print(f"\n📊 Out of {tp+fn} fake jobs in test set:")
print(f"   ✅ Caught: {tp} ({tp/(tp+fn)*100:.1f}%)")
print(f"   ❌ Missed: {fn} ({fn/(tp+fn)*100:.1f}%)")

print("\n💾 SAVED FILES:")
print("   ✅ output/model_evaluation.png - Evaluation visualizations")
print("   ✅ output/evaluation_results.pkl - Detailed results")

print("\n🚀 NEXT STEP: Deploy the model with Flask API")
print("   Run: python app.py")
print("=" * 80)