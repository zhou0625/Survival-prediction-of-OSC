import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

# 定义要调整的参数和其取值范围
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [5, 10, 15, 20, 25, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
#dataset = pd.read_csv(r"C:\Users\86136\Desktop\feature\64 ROI+clinical.csv")
dataset = pd.read_csv(r"E:\yesyes\64.csv")
#X = dataset.drop(['label'], axis=1)
#X = dataset[['age', 'stage']]
#X = dataset[['0', '3', '12', '16', '19', '28', '32', '34', '37', '43', '47', '52', '55', '58', '62', 'age', 'stage']]#原图
#X = dataset[['11', '20', '31', '37', '40', '41', '42', '45', '52', '57', 'ageROI', 'stageROI']]#ROI'ageROI', 'stageROI'
X = dataset[['12', '13', '21', '37', '47', '50', '51']]


y = dataset[['label']]
print(X)
print(y)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 创建随机森林分类器
rf = RandomForestClassifier()

# 创建网格搜索对象
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)

# 拟合数据并找到最优参数
grid_search.fit(X_train, y_train)

# 打印最优参数
print(grid_search.best_params_)
