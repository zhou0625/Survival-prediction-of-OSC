import pandas as pd
from lifelines import CoxPHFitter

# 加载数据集
# data是一个pandas DataFrame，其中包含你的生存时间，事件发生指标和协变量

# 例如：
from matplotlib import pyplot as plt
data = pd.read_csv(r"C:\Users\86136\Desktop\zzzzr\feature select\clinical feature.csv")
# 初始化一个Cox比例风险模型
cph_single = CoxPHFitter()#cph_multi = CoxPHFitter()
# 对单个变量进行Cox回归
cph_single.fit(data, duration_col='survival time(year)', event_col='label', formula="size")
#cph_single.fit(data, duration_col='survival time(year)', event_col='label', formula="size + age + stage + treatment")
# 输出分析结果
cph_single.print_summary()


'''def plot_forest(model, ax, title):
    summary = model.summary
    summary['lower_ci'] = summary['coef'] - 1.96 * summary['se(coef)']
    summary['upper_ci'] = summary['coef'] + 1.96 * summary['se(coef)']

    y = list(range(summary.shape[0], 0, -1))
    ax.errorbar(summary['coef'], y, xerr=(summary['coef'] - summary['lower_ci'], summary['upper_ci'] - summary['coef']),
                fmt='o', label='HR (95% CI)')
    ax.axvline(0, color='grey', linestyle='--')
    ax.set_yticks(y)
    ax.set_yticklabels(summary.index)
    ax.set_title(title)


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

plot_forest(cph_single, axes[0], 'Univariate Cox Regression')
plot_forest(cph_multi, axes[1], 'Multivariate Cox Regression')

plt.tight_layout()
plt.show()'''





