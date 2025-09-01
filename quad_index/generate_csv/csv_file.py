import pandas as pd

# 定义列名
columns = [
    'K', 'group', 'database', 'keywords_number', 'keywords_number_avg',
    'original_lines', 'index_lines',
    'invert_index_time(s)', 'quad_index_time(s)', 'tree_min_depth', 'tree_max_depth',
    'voronoi_index_time(s)', 'center_index_time(s)', 'centroid_index_time(s)', 'mixed_index_time(s)', 'encrypt_index_time(s)',
    'original_size(MB)', 'voronoi_index_size(MB)', 'center_index_size(MB)', 'centroid_index_size(MB)', 'mixed_index_size(MB)',
    'KNN_query', 'range_query', 'range_para', 'query_number',
    'use_same_file', 'use_voronoi_file', 'use_center_file', 'use_centroid_file',
    'token_time_avg(s)', 'hash_token_time_avg(s)', 'matched_time_avg(s)', 'decrypt_time_avg(s)', 'refine_time_avg(s)',
    'results_count_avg', 'token_count_avg', 'candidate_count_avg', 'token_size_avg(bytes)', 'candidate_size_avg(bytes)',
    'ratio_avg', 'recall_avg', 'precision_avg',
    'last_modify_time'  # last modification time of the row data
]


"""
生成初始带模版的CSV文件

# 定义K值
k_values = [1, 2, 4, 6, 8, 10]
initial_group = 1  # 初始创建group=1的模板

# 创建空的数据结构（只保留K值，其他列为空）
data = []
for k in k_values:
    row = [k, initial_group] + [None] * (len(columns) - 2)  # K值 + 其他列为空
    data.append(row)
"""

# 创建DataFrame
df = pd.DataFrame(columns=columns)

csv_file_filepath = '../../exp_result/data_statistics0828.csv'

# 保存为CSV文件
df.to_csv(csv_file_filepath, index=False, encoding='utf-8', sep=',')
print(f"CSV模板文件已生成: {csv_file_filepath}")
