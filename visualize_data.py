import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

print("Creating visualizations...")

# Load data
df = pd.read_csv('data/fake_job_postings.csv')

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))

# ============================================================================
# PLOT 1: Class Distribution
# ============================================================================
ax1 = plt.subplot(3, 3, 1)
fraud_counts = df['fraudulent'].value_counts()
colors = ['#2ecc71', '#e74c3c']
bars = ax1.bar(['Real Jobs', 'Fake Jobs'], fraud_counts.values, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
ax1.set_ylim(0, max(fraud_counts.values) * 1.1)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}\n({height/len(df)*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold')

# ============================================================================
# PLOT 2: Missing Values
# ============================================================================
ax2 = plt.subplot(3, 3, 2)
missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing = missing[missing > 0]
ax2.barh(range(len(missing)), missing.values, color='#3498db', alpha=0.7, edgecolor='black')
ax2.set_yticks(range(len(missing)))
ax2.set_yticklabels(missing.index, fontsize=10)
ax2.set_xlabel('Missing %', fontsize=12, fontweight='bold')
ax2.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
ax2.invert_yaxis()

# ============================================================================
# PLOT 3: Text Length Comparison
# ============================================================================
ax3 = plt.subplot(3, 3, 3)
text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
real_lengths = []
fake_lengths = []

for col in text_cols:
    real_lengths.append(df[df['fraudulent']==0][col].fillna('').str.len().mean())
    fake_lengths.append(df[df['fraudulent']==1][col].fillna('').str.len().mean())

x = range(len(text_cols))
width = 0.35
ax3.bar([i - width/2 for i in x], real_lengths, width, label='Real', color='#2ecc71', alpha=0.7, edgecolor='black')
ax3.bar([i + width/2 for i in x], fake_lengths, width, label='Fake', color='#e74c3c', alpha=0.7, edgecolor='black')
ax3.set_ylabel('Avg Characters', fontsize=12, fontweight='bold')
ax3.set_title('Text Length: Real vs Fake', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels([col.replace('_', '\n') for col in text_cols], fontsize=9)
ax3.legend()

# ============================================================================
# PLOT 4: Telecommuting
# ============================================================================
ax4 = plt.subplot(3, 3, 4)
telecom_data = df.groupby('fraudulent')['telecommuting'].mean() * 100
bars = ax4.bar(['Real', 'Fake'], telecom_data.values, color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
ax4.set_ylabel('% with Telecommuting', fontsize=12, fontweight='bold')
ax4.set_title('Telecommuting Option', fontsize=14, fontweight='bold')
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

# ============================================================================
# PLOT 5: Company Logo
# ============================================================================
ax5 = plt.subplot(3, 3, 5)
logo_data = df.groupby('fraudulent')['has_company_logo'].mean() * 100
bars = ax5.bar(['Real', 'Fake'], logo_data.values, color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
ax5.set_ylabel('% with Company Logo', fontsize=12, fontweight='bold')
ax5.set_title('Company Logo Presence', fontsize=14, fontweight='bold')
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

# ============================================================================
# PLOT 6: Screening Questions
# ============================================================================
ax6 = plt.subplot(3, 3, 6)
questions_data = df.groupby('fraudulent')['has_questions'].mean() * 100
bars = ax6.bar(['Real', 'Fake'], questions_data.values, color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
ax6.set_ylabel('% with Screening Questions', fontsize=12, fontweight='bold')
ax6.set_title('Screening Questions', fontsize=14, fontweight='bold')
for bar in bars:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

# ============================================================================
# PLOT 7: Employment Type Distribution
# ============================================================================
ax7 = plt.subplot(3, 3, 7)
top_employment = df['employment_type'].value_counts().head(5)
ax7.barh(range(len(top_employment)), top_employment.values, color='#9b59b6', alpha=0.7, edgecolor='black')
ax7.set_yticks(range(len(top_employment)))
ax7.set_yticklabels(top_employment.index, fontsize=10)
ax7.set_xlabel('Count', fontsize=12, fontweight='bold')
ax7.set_title('Top Employment Types', fontsize=14, fontweight='bold')
ax7.invert_yaxis()

# ============================================================================
# PLOT 8: Experience Required
# ============================================================================
ax8 = plt.subplot(3, 3, 8)
top_exp = df['required_experience'].value_counts().head(5)
ax8.barh(range(len(top_exp)), top_exp.values, color='#e67e22', alpha=0.7, edgecolor='black')
ax8.set_yticks(range(len(top_exp)))
ax8.set_yticklabels(top_exp.index, fontsize=10)
ax8.set_xlabel('Count', fontsize=12, fontweight='bold')
ax8.set_title('Required Experience Levels', fontsize=14, fontweight='bold')
ax8.invert_yaxis()

# ============================================================================
# PLOT 9: Correlation of Binary Features
# ============================================================================
ax9 = plt.subplot(3, 3, 9)
binary_features = ['fraudulent', 'telecommuting', 'has_company_logo', 'has_questions']
corr_matrix = df[binary_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, ax=ax9, cbar_kws={'shrink': 0.8})
ax9.set_title('Feature Correlations', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('output/data_exploration.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved as 'data_exploration.png'")
print("   Location: C:\\Users\\Vijay\\Desktop\\fake_job_detection\\output\data_exploration.png")
plt.show()