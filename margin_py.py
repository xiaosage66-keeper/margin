# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test



# 1. 

file_path = r'D:\硕士\胰头癌切缘\原始数据\result\切缘术后化疗\Figure3原始表.xlsx'

df = pd.read_excel(file_path)

print("数据读取成功")
print("总病例数:", len(df))

# 2. 
df['duration'] = pd.to_numeric(
    df['生存时间'],
    errors='coerce'
)


df = df.dropna(
    subset=['duration']
)


# 状态：
# 0 = 死亡
# 其他 = 截尾

df['event'] = df['状态'].apply(
    lambda x: 1 if x == 0 else 0
)


df['是否化疗'] = df['是否化疗'].fillna('否')

# 3. 

def define_groups(row):

    margin = ">1mm" if row['切缘'] == 0 else "≤1mm"

    chemo = (
        "+Chemo"
        if row['是否化疗'] == '是'
        else "+NoChemo"
    )

    return f"{margin} {chemo}"


df['Group'] = df.apply(
    define_groups,
    axis=1
)


print("\n分组情况:")
print(df['Group'].value_counts())

# 4. 


results = multivariate_logrank_test(
    df['duration'],
    df['Group'],
    df['event']
)

p_value = results.p_value
print("\nGlobal log-rank P =", p_value)

# 5.
plt.rcParams['font.sans-serif'] = ['Arial']

fig, ax = plt.subplots(
    figsize=(12,9)
)

kmfs = []
order = [
    '>1mm +Chemo',
    '>1mm +NoChemo',
    '≤1mm +Chemo',
    '≤1mm +NoChemo'
]
colors = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728'
]
max_time = int(
    df['duration'].max()
)

time_ticks = np.arange(
    0,
    max_time + 12,
    12
)

mos_info = []

# 6. 
for i,name in enumerate(order):

    mask = (
        df['Group'] == name
    )

    if mask.any():

        kmf = KaplanMeierFitter()

        kmf.fit(
            df.loc[mask,'duration'],
            df.loc[mask,'event'],
            label=name
        )

        kmf.plot_survival_function(
            ax=ax,
            ci_show=False,
            color=colors[i],
            lw=2.5,
            show_censors=True,
            censor_styles={
                'ms':7,
                'marker':'+',
                'mew':1.5
            }
        )

        kmfs.append(kmf)

        # median OS

        mos = kmf.median_survival_time_


        if np.isfinite(mos):
            mos_text = f"{mos:.1f} months"
        else:
            mos_text = "NR"


        mos_info.append(
            f"{name}: {mos_text}"
        )

# 7.
add_at_risk_counts(
    *kmfs,
    ax=ax,
    xticks=time_ticks
)

# 8. 

ax.set_xticks(
    time_ticks
)


ax.set_title(
    'Overall Survival Stratified by Margin and Adjuvant Chemotherapy',
    fontsize=16,
    pad=30
)


ax.set_xlabel(
    'Time (Months)',
    fontsize=12
)


ax.set_ylabel(
    'Survival Probability',
    fontsize=12
)


ax.set_ylim(
    [0,1.05]
)

ax.legend(
    frameon=False,
    loc='upper right',
    fontsize=10
)

# 9. 
if p_value < 0.001:

    p_text = "Log-rank P < 0.001"

else:

    p_text = f"Log-rank P = {p_value:.3f}"



final_text = (
    p_text
    +
    "\n\nMedian Survival:\n"
    +
    "\n".join(
        mos_info
    )
)

ax.text(
    0.05,
    0.18,
    final_text,
    fontsize=11,
    transform=ax.transAxes,
    verticalalignment='bottom',
    horizontalalignment='left',
    linespacing=1.6
)

plt.tight_layout()

save_path = r'D:\硕士\胰头癌切缘\原始数据\result\切缘术后化疗\Figure3_reproduced.png'

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("\n图片保存位置:")
print(save_path)
