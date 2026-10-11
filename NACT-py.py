import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts

# 1.
current_dir = os.path.dirname(os.path.abspath(__file__))
files = [f for f in os.listdir(current_dir) if f.endswith(('.xlsx', '.xls'))]
file_path = os.path.join(current_dir, files[0]) if files else r'C:\Users\knwch\Desktop\margin\result\新辅助-切缘\新辅助化疗与切缘-核对版1.xlsx'

df = pd.read_excel(file_path)

# 2. 
def find_col(keys):
    for c in df.columns:
        if any(k in str(c) for k in keys): return c
    return None

margin_col = find_col(['切缘'])
nac_col = find_col(['化疗', '新辅助'])
time_col = find_col(['OS', '生存时间'])
status_col = find_col(['状态'])

# 3. 
df[nac_col] = df[nac_col].fillna('否').astype(str).str.strip()
df['event'] = df[status_col].apply(lambda x: 1 if x == 0 else 0)

def get_group(row):
    m = '>1mm' if row[margin_col] == 0 else '≤1mm'
    n = 'NAC+' if '是' in str(row[nac_col]) else 'NAC-'
    return f"{m}_{n}"

df['Group'] = df.apply(get_group, axis=1)

# 4. 
def plot_survival_final(t_col, title):
    groups = ['>1mm_NAC+', '>1mm_NAC-', '≤1mm_NAC+', '≤1mm_NAC-']
    actual_groups = [g for g in groups if g in df['Group'].unique()]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    kmfs = []
    mos_list = [] 

    for g in actual_groups:
        g_df = df[df['Group'] == g]
        kmf = KaplanMeierFitter()
        kmf.fit(g_df[t_col], event_observed=g_df['event'], label=g)
        
        kmf.plot_survival_function(ax=ax, ci_show=False, lw=2, 
                                   show_censors=True, 
                                   censor_styles={'ms': 6, 'marker': '+', 'mew': 1.5})
        kmfs.append(kmf)
        
        mos = kmf.median_survival_time_
        mos_val = f"{mos:.1f} m" if np.isfinite(mos) else "NR"
        mos_list.append(f"{g}: {mos_val}")


    max_time = df[t_col].max()
    time_ticks = np.arange(0, max_time + 12, 12) 
    ax.set_xticks(time_ticks) 

    results = multivariate_logrank_test(df[t_col], df['Group'], df['event'])
    p_value = results.p_value
    p_str = f"Log-rank P { '< 0.001' if p_value < 0.001 else f'= {p_value:.3f}'}"


    stats_text = f"{p_str}\n" + "Median Survival:\n  " + "\n  ".join(mos_list)
    ax.text(0.05, 0.18, stats_text, transform=ax.transAxes, 
            verticalalignment='bottom', horizontalalignment='left',
            fontsize=11, family='sans-serif', color='black')

    ax.legend(loc='upper right', frameon=False)


    add_at_risk_counts(*kmfs, ax=ax, xticks=time_ticks)
    
    ax.set_title(title, fontsize=15, pad=20)
    ax.set_xlabel("Time (Months)", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.grid(False) 
    
    plt.tight_layout()
    plt.show()

# 5. 
plot_survival_final(time_col, 'Overall Survival (OS)')