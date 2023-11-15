import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
import sklearn.metrics as sm

dataset_64 = pd.read_csv(r"E:\write\feature\64.csv")
dataset_64_clinical = pd.read_csv(r"E:\write\feature\64+clinical.csv")
dataset_ROI = pd.read_csv(r"E:\write\feature\64 ROI.csv")
dataset_ROI_clinical = pd.read_csv(r"E:\write\feature\64 ROI+clinical.csv")
#X = dataset[['11', '20', '31', '37', '40', '41', '42', '45', '52', '57', 'ageROI', 'stageROI']]  #



X_64 = dataset_64[['0', '3', '12', '16', '19', '28', '32', '34', '37', '43', '47', '52', '55', '58', '62']]#原图
X_64_clinic = dataset_64_clinical[['0', '3', '12', '16', '19', '28', '32', '34', '37', '43', '47', '52', '55', '58', '62', 'age', 'stage']]
X_ROI = dataset_ROI[['11', '20', '31', '37', '40', '41', '42', '45', '52', '57']] #ROI, 'ageROI', 'stageROI'
X_ROI_clinic = dataset_ROI_clinical[['11', '20', '31', '37', '40', '41', '42', '45', '52', '57', 'ageROI', 'stageROI']] #,

#X = dataset[['age', 'stage']]

#X = dataset[['12', '13', '21', '37', '47', '50', '51', 'age', 'stage']]
y_64 = dataset_64[['label']]
y_64_clinic = dataset_64_clinical[['label']]
y_ROI = dataset_ROI[['label']]
y_ROI_clinic = dataset_ROI_clinical[['label']]


# 划分数据集
X_train_64, X_test_64, y_train_64, y_test_64 = train_test_split(X_64, y_64, test_size=0.2, random_state=300)#原图
X_train_64_clinic, X_test_64_clinic, y_train_64_clinic, y_test_64_clinic = train_test_split(X_64_clinic, y_64_clinic, test_size=0.2, random_state=300)
X_train_ROI, X_test_ROI, y_train_ROI, y_test_ROI = train_test_split(X_ROI, y_ROI, test_size=0.2, random_state=300)
X_train_ROI_clinic, X_test_ROI_clinic, y_train_ROI_clinic, y_test_ROI_clinic = train_test_split(X_ROI_clinic, y_ROI_clinic, test_size=0.2, random_state=300)

sc = StandardScaler()
X_train_64_scaled = sc.fit_transform(X_train_64)
X_test_64_scaled = sc.transform(X_test_64)
X_train_64_clinic_scaled = sc.fit_transform(X_train_64_clinic)
X_test_64_clinic_scaled = sc.transform(X_test_64_clinic)
X_train_ROI_scaled = sc.fit_transform(X_train_ROI)
X_test_ROI_scaled = sc.transform(X_test_ROI)
X_train_ROI_clinic_scaled = sc.fit_transform(X_train_ROI_clinic)
X_test_ROI_clinic_scaled = sc.transform(X_test_ROI_clinic)

# 使用XGBoost进行分类预测
model_ROI = XGBClassifier(gamma=0, learning_rate=0.1, max_depth=4, n_estimators=100,  reg_alpha=0.1,
                      reg_lambda=2, subsample=0.7)#ROI
model_ROI_clinic = XGBClassifier(gamma=0, learning_rate=0.05, max_depth=5, n_estimators=300,  reg_alpha=0,
                      reg_lambda=2,  subsample=0.7)#ROI+clinical
model_64_clinic = XGBClassifier(objective='binary:logistic', gamma=0.2, learning_rate=0.01, max_depth=4, n_estimators=100,  reg_alpha=0.2,
                      reg_lambda=2, subsample=0.7)#原图+临床

model_64 = XGBClassifier(gamma=0.1, learning_rate=0.01, max_depth=5, n_estimators=200,  reg_alpha=0,
                      reg_lambda=1, subsample=0.7)#原图

model_64.fit(X_train_64_scaled, y_train_64)
model_64_clinic.fit(X_train_64_clinic_scaled, y_train_64_clinic)
model_ROI.fit(X_train_ROI_scaled, y_train_ROI)
model_ROI_clinic.fit(X_train_ROI_clinic_scaled, y_train_ROI_clinic)

y_test_pred_64 = model_64.predict(X_test_64_scaled)
y_test_pred_64_clinic = model_64_clinic.predict(X_test_64_clinic_scaled)
y_test_pred_ROI = model_ROI.predict(X_test_ROI_scaled)
y_test_pred_ROI_clinic = model_ROI_clinic.predict(X_test_ROI_clinic_scaled)



def bootstrap_auc(y_true, y_scores, n_bootstraps=1000, rng_seed=42):
    n_bootstraps = n_bootstraps
    rng = np.random.RandomState(rng_seed)
    auc_scores = []

    for i in range(n_bootstraps):
        # 有放回地随机抽样
        indices = rng.randint(0, len(y_scores), len(y_scores))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            continue
        auc_score = roc_auc_score(y_true[indices], y_scores[indices])
        auc_scores.append(auc_score)

    sorted_scores = np.array(auc_scores)
    sorted_scores.sort()
    # 计算95%置信区间
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    return confidence_lower, confidence_upper




# 定义函数来绘制ROC曲线
def plot_roc_curve(y_test, y_pred_proba, label=None):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    confidence_lower, confidence_upper = bootstrap_auc(y_test.to_numpy().ravel(), y_pred_proba)
    plt.plot(fpr, tpr, label=f"{label} AUC={auc_score:.2f} (95% CI: {confidence_lower:.2f}-{confidence_upper:.2f})")
    return auc_score

# 获取四种情况下的预测概率
probas_64 = model_64.predict_proba(X_test_64_scaled)[:, 1]
probas_64_clinic = model_64_clinic.predict_proba(X_test_64_clinic_scaled)[:, 1]
probas_ROI = model_ROI.predict_proba(X_test_ROI_scaled)[:, 1]
probas_ROI_clinic = model_ROI_clinic.predict_proba(X_test_ROI_clinic_scaled)[:, 1]

# 绘制ROC曲线
plt.figure(figsize=(10, 8))

plot_roc_curve(y_test_64, probas_64, label="Original")
plot_roc_curve(y_test_64_clinic, probas_64_clinic, label="Original + Clinical")
plot_roc_curve(y_test_ROI, probas_ROI, label="ROI")
plot_roc_curve(y_test_ROI_clinic, probas_ROI_clinic, label="ROI + Clinical")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost-ROC')
plt.legend(loc="lower right")
plt.show()


def print_evaluation_metrics(y_true, y_pred, label=None):
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)

    if label:
        print(f"---- {label} ----")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\n")


# 使用该函数为你的四种情况输出评价指标
print_evaluation_metrics(y_test_64, y_test_pred_64, "Original")
print_evaluation_metrics(y_test_64_clinic, y_test_pred_64_clinic, "Original + Clinical")
print_evaluation_metrics(y_test_ROI, y_test_pred_ROI, "ROI")
print_evaluation_metrics(y_test_ROI_clinic, y_test_pred_ROI_clinic, "ROI + Clinical")

