import pandas as pd 
from ip.utils import stats_utils
from ip.experiments.plotting import make_ci_plot

df = pd.read_csv("results/em.csv")
print(df['model'].unique())
print(df['group'].unique())
print(df['evaluation_id'].unique())
print(df['system_prompt_group'].unique())

df['score'] = df['score'].astype(float)

# Construct new groups based on the concatenation of group and system_prompt_group
df['group'] = df['group'].astype(str) + '_' + df['system_prompt_group'].astype(str)
print(df['group'].unique())

# Calculate CI intervals
ci_df = stats_utils.compute_ci_df(df, group_cols=["group", "evaluation_id"], value_col="score")
ci_df.to_csv("results/em_ci.csv", index=False)

# Plot the results
fig, _ = make_ci_plot(ci_df, x_column = 'evaluation_id', legend_nrows=3, ylabel="P(Misaligned Answer)", figsize=(10, 4))
fig.savefig("results/em_ci.pdf", bbox_inches="tight")