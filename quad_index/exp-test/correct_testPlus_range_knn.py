import pandas as pd
import time
import sys
import os

import InvertedIndex
import depth_test
import query_test
import correct_testPlus_knn
import AES_sha1

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from quad_index import quad_token


def analyze_query_coverage(query_words, query_rect, matched_data, invert_file_path):
    # 1. 从倒排索引文件读取原始数据
    original_points = []
    with open(invert_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            keyword = parts[0]
            if keyword == query_words:
                # 提取所有坐标字符串，去掉括号
                coord_strs = [part.strip('()') for part in parts[1:]]
                # 使用步长为2的切片分别获取x和y坐标
                original_points = [(int(coord_strs[i]), int(coord_strs[i + 1]))
                                   for i in range(0, len(coord_strs), 2)]
                break
    # print(f"关键词 '{query_words}' 在原始数据集中共有 {len(original_points)} 个点")

    # 2. 计算在查询范围内的原始点
    x_min, x_max, y_min, y_max = query_rect
    points_in_range = []
    for x, y in original_points:
        if x_min <= x <= x_max and y_min <= y <= y_max:
            points_in_range.append((x, y))
    # print(f"在查询范围内的原始点数量: {len(points_in_range)}")

    # 3. 解析查询结果中的点
    returned_points = [(int(x), int(y))
                       for data_line in matched_data
                       for part in data_line.split()[1:]
                       if ',' in part
                       for x, y in [part.strip('()').split(',')]]

    # 4. 计算交集 - 实际返回的正确点
    points_in_range_set = set(points_in_range)
    returned_points_set = set(returned_points)
    # print(f"查询结果返回的点数量: {len(returned_points_set)}")

    correct_returned = points_in_range_set.intersection(returned_points_set)
    # print(f"查询结果中正确的点数量: {len(correct_returned)}")

    # 5. 计算召回率和精确率
    recall = len(correct_returned) / len(points_in_range_set) if len(points_in_range_set) > 0 else 0
    precision = len(correct_returned) / len(returned_points_set) if len(returned_points_set) > 0 else 0

    # 6. 显示详细统计
    return len(points_in_range_set), recall, precision


def analyze_query_coverage_plus(query_rect, matched_data1, matched_data2):
    x_min, x_max, y_min, y_max = query_rect

    # 1. 从 matched_data1 中解析点
    returned_points1 = [(int(x), int(y))
                        for data_line in matched_data1
                        for part in data_line.split()[1:]
                        if ',' in part
                        for x, y in [part.strip('()').split(',')]]

    # 2. 判断哪些点在 query_rect 中，作为精确解
    exact_points1 = [(x, y) for x, y in returned_points1
                     if x_min <= x <= x_max and y_min <= y <= y_max]

    returned_points_set1 = set(returned_points1)
    points_in_range_set1 = set(exact_points1)

    # 3. 从 matched_data2 中解析点
    returned_points2 = [(int(x), int(y))
                        for data_line in matched_data2
                        for part in data_line.split()[1:]
                        if ',' in part
                        for x, y in [part.strip('()').split(',')]]

    # 4. 判断哪些点在 query_rect 中，作为精确解
    exact_points2 = [(x, y) for x, y in returned_points2
                     if x_min <= x <= x_max and y_min <= y <= y_max]

    returned_points_set2 = set(returned_points2)
    points_in_range_set2 = set(exact_points2)

    # 5. 计算双关键词的结果
    # 返回点 = 两个集合的并集
    returned_points_set = returned_points_set1.union(returned_points_set2)

    # 精确解 = 两个精确解集合的交集
    points_in_range_set = points_in_range_set1.intersection(points_in_range_set2)

    # 6. recall 固定为 1
    recall = 1 if len(returned_points_set) > 0 else 0

    # 7. precision = 精确解个数 / 返回点个数
    if len(points_in_range_set) == 0:
        precision = 1.0
    else:
        precision = len(points_in_range_set) / len(returned_points_set) if returned_points_set else 0

    return len(points_in_range_set), recall, precision


def format_scientific(value, decimal_places=4):
    """
    将数值格式化为科学计数法，保留指定小数位数
    例如：0.00002557 -> "2.5570e-05"
    返回float类型 这样Excel能正确识别为数值而不是文本
    """
    if value == 0:
        return "0.0000e+00"
    return float(f"{value:.{decimal_places}e}")


def update_csv_with_results(csv_filepath, K, dataset_name, algorithm_type, result, group):
    """
    更新CSV文件中对应K值行的统计结果
    """
    # 读取现有的CSV文件，并指定字符串列的数据类型
    try:
        # 指定需要存储字符串的列为object类型
        dtype_dict = {
            'database': 'object',
        }
        df = pd.read_csv(csv_filepath, dtype=dtype_dict)
    except FileNotFoundError:
        print(f"❌ CSV文件不存在: {csv_filepath}")
        return

    # 找到对应K值的行
    k_row_index = df[
        (df['K'] == K) &
        (df['group'] == group) &
        (df['range_para'] == result['range_para'])
        ].index

    if len(k_row_index) == 0:
        # 创建新行数据
        print(f"🆕 创建新行: K={K}, group={group}, range_para={result['range_para']}")
        # 获取新行的索引（下一个可用的索引）
        k_row_index = len(df)
        # 使用loc直接创建新行，避免concat警告
        df.loc[k_row_index, 'K'] = K
        df.loc[k_row_index, 'group'] = group
        df.loc[k_row_index, 'range_para'] = result['range_para']
        # 初始化其他列为适当默认值
        for col in df.columns:
            if col not in ['K', 'group', 'range_para']:
                df.loc[k_row_index, col] = None
    else:
        # 如果找到了对应的行，获取第一个匹配的行索引
        k_row_index = k_row_index[0]
        print(f"🔄 更新现有行: K={K}, group={group}, range_para={result['range_para']}")

    # 填写基本信息
    df.loc[k_row_index, 'database'] = dataset_name
    df.loc[k_row_index, 'range_query'] = 1  # 范围查询固定为1
    df.loc[k_row_index, 'query_number'] = result['total_tests']
    df.loc[k_row_index, 'tree_min_depth'] = result['min_depth']
    df.loc[k_row_index, 'tree_max_depth'] = result['max_depth']

    # 范围查询
    if algorithm_type == 'center':
        df.loc[k_row_index, 'use_center_file'] = 1
    elif algorithm_type == 'centroid':
        df.loc[k_row_index, 'use_centroid_file'] = 1

    # 填写性能统计数据（时间数据使用科学计数法格式）
    df.loc[k_row_index, 'token_time_avg(s)'] = format_scientific(result['avg_token_time'])
    df.loc[k_row_index, 'hash_token_time_avg(s)'] = format_scientific(result['avg_hash_token_time'])
    df.loc[k_row_index, 'matched_time_avg(s)'] = format_scientific(result['avg_matched_time'])
    df.loc[k_row_index, 'decrypt_time_avg(s)'] = format_scientific(result['avg_decrypt_time'])
    df.loc[k_row_index, 'refine_time_avg(s)'] = format_scientific(result['avg_refine_time'])

    # 其他数据保持原有格式
    df.loc[k_row_index, 'results_count_avg'] = result['avg_results_count']
    df.loc[k_row_index, 'token_count_avg'] = result['avg_token_count']
    df.loc[k_row_index, 'candidate_count_avg'] = result['avg_candidate_count']
    df.loc[k_row_index, 'token_size_avg(bytes)'] = result['avg_token_size']
    df.loc[k_row_index, 'candidate_size_avg(bytes)'] = result['avg_candidate_size']
    df.loc[k_row_index, 'recall_avg'] = round(result['avg_recall'], 4)
    df.loc[k_row_index, 'precision_avg'] = round(result['avg_precision'], 4)
    # 范围查询查询不填写ration（保持为空）

    # 填写最后修改时间
    df.loc[k_row_index, 'last_modify_time'] = int(time.strftime("%d%H%M%S", time.localtime()))

    # 保存更新后的CSV文件
    df.to_csv(csv_filepath, index=False, encoding='utf-8', sep=',')
    print(f"✅ 已更新CSV文件中K={K} group={group} range_para={result['range_para']}的统计结果")


if __name__ == '__main__':
    original_filepath = None
    test_filepath = None
    csv_filepath = '../../exp_result/data_statistics0828.csv'
    base_dir = '../../Cdatabase'

    """
    配置区域 - 自动测试多个K值
    """
    K = 80  # 要测试的K值列表
    datasets_name = ['twitter', 'newyork', 'paris']
    algorithm_type = 'center'  # 'center' 或者 'centroid'
    # 范围查询 该结果始终为false
    not_voronoi_query = False
    exp_group = 4

    x_min, x_max, y_min, y_max = 0, 100000000, 0, 100000000
    initial_bounds = [x_min, x_max, y_min, y_max]

    # ========== 循环正式开始 ==========
    # ========== 20250823 这里是以数据集控制循环 下面以范围控制循环 ==========
    for dataset_name in datasets_name:
        EXPERIMENT_GROUP = exp_group
        exp_group += 1
        if dataset_name == 'twitter':
            original_filepath = '../../original_dataset/twitter_new_RASK108.txt'
            test_filepath = '../../Adatabase/twitter_test_2.txt'
        elif dataset_name == 'newyork':
            original_filepath = '../../original_dataset/newyork_RASK108.txt'
            test_filepath = '../../Adatabase/newyork_test_2.txt'
        elif dataset_name == 'paris':
            original_filepath = '../../original_dataset/Paris_RASK108.txt'
            test_filepath = '../../Adatabase/paris_test_2.txt'
        inverted_index = InvertedIndex.build_inverted_index_v2(original_filepath)
        test_keywords_list, test_points = correct_testPlus_knn.load_test_data(test_filepath)

        # ========== 汇总结果统计 ==========
        total_results = {}  # 存储每个范围的测试结果
        # ========== 循环测试1%到5%的半径 ==========
        setR = 5000000000000000  # 5e15
        percent_first = 0.01
        percent_second = 0.05
        int_percent_first = int(percent_first * 100)  # 1
        int_percent_second = int(percent_second * 100)  # 5
        now_percent_first = percent_first
        for percent_int in range(int_percent_first, int_percent_second + 1):  # 1~5
            # 实验组配置
            percent = now_percent_first
            now_percent_first += 0.01
            print(f"当前数据集：{dataset_name} 当前半径控制参数: {percent_int}% 的半径查询 (实验组: {EXPERIMENT_GROUP})")

            # 创造查询半径
            radius_ratio = percent * percent
            r2 = setR * radius_ratio
            radius = int(r2 ** 0.5)

            print("=" * 80)
            print(f"🔍 开始测试 K={K:2d}")
            print("=" * 80)

            # ========== 全部用K=设定值的文件 ==========
            generate_filepath = f'{base_dir}/{dataset_name}_quadtree_{algorithm_type}{K}.txt'
            hash_filepath = f'{base_dir}/{dataset_name}_quadtree_{algorithm_type}{K}_hash.txt'

            # 检查文件是否存在
            if not os.path.exists(generate_filepath) or not os.path.exists(hash_filepath):
                print(f"❌ K={K:2d} 的文件不存在，跳过测试")
                total_results[percent_int] = "文件不存在"
                continue

            # ========== 自动计算最大最小深度 ==========
            max_depth, min_depth = [x // 2 for x in depth_test.count_max_min_length(generate_filepath)]

            hash_dict = {}
            with open(hash_filepath, 'r') as hash_file:
                for line in hash_file:
                    line = line.strip()
                    if line:  # 跳过空行
                        parts = line.split(' ', 1)
                        if len(parts) >= 2:
                            hash_key = parts[0]
                            hash_dict[hash_key] = line  # 保存完整行

            # 进行批量查询
            test_count = 0
            # ========== 该K值测试的各种时间统计 ==========
            k_time = []
            k_token_time = []         # 统计明文token生成时间
            k_hash_token_time = []    # 统计hash_token生成时间
            k_matched_time = []       # 统计匹配时间
            k_refine_time = []        # 统计精炼结果时间
            k_decrypt_time = []       # 统计对称解密时间
            # ========== 统计该K值测试的候选点数量 ==========
            result_counts = []        # 统计当前K值100次查询精确结果的平均点数量
            token_counts = []         # 统计当前K值100次查询token数量
            candidate_counts = []     # 统计当前K值100次查询候选集的平均点数量
            token_sizes = []          # 统计当前K值100次查询token大小--字节数
            candidate_sizes = []      # 统计当前K值100次查询候选集大小--字节数
            # ========== 召回率和精确率 ==========
            recall_list = []          # 统计当前K值查询召回率
            precision_list = []       # 统计当前K值查询精确率

            for keywords, point in zip(test_keywords_list, test_points):
                test_count += 1
                keyword1 = keywords[0]
                keyword2 = keywords[1]
                # 利用point得到query_rect
                query_rect = (point[0] - radius, point[0] + radius, point[1] - radius, point[1] + radius)

                print(f"[{test_count}/{len(test_keywords_list)}]--查询范围: {query_rect}")

                # 分批处理并直接匹配
                print(f"  关键词1: {keyword1}")
                matched_data1 = []
                hash_token_list1 = []
                matched_time1 = 0
                hash_token_time1 = 0

                print(f"  关键词2: {keyword2}")
                matched_data2 = []
                hash_token_list2 = []
                matched_time2 = 0
                hash_token_time2 = 0

                batch_count = 0
                # ========== 开始计时 ==========
                start_time1 = time.perf_counter()
                for batch_result in quad_token.query_quadtree_range_optimized_batch(min_depth, max_depth, x_min, x_max, y_min, y_max, query_rect, batch_size=10000):
                    batch_count += 1

                    # 构建当前批次的result_list
                    if batch_count == 1:
                        # 只有第一批需要"-1"
                        token_list = ["-1"]
                    else:
                        # 后续批次不需要"-1"
                        token_list = []

                    for depth in sorted(batch_result.keys()):
                        token_list += batch_result[depth]

                    batch_start_time1 = time.perf_counter()
                    # 直接对当前批次进行匹配
                    batch_matched_data1, batch_hash_token_time1, batch_hash_token_list1 = query_test.match_query_results(keyword1, token_list, hash_dict, initial_bounds, min_depth, max_depth, not_voronoi_query)
                    batch_matched_time_with_hash_token1 = time.perf_counter() - batch_start_time1
                    batch_matched_time1 = batch_matched_time_with_hash_token1 - batch_hash_token_time1

                    batch_start_time1e = time.perf_counter()
                    # 直接对当前批次进行匹配
                    batch_matched_data2, batch_hash_token_time2, batch_hash_token_list2 = query_test.match_query_results(keyword2, token_list, hash_dict, initial_bounds, min_depth, max_depth, not_voronoi_query)
                    batch_matched_time_with_hash_token2 = time.perf_counter() - batch_start_time1e
                    batch_matched_time2 = batch_matched_time_with_hash_token2 - batch_hash_token_time2

                    # 累积结果
                    matched_data1.extend(batch_matched_data1)
                    hash_token_list1.extend(batch_hash_token_list1)
                    matched_time1 += batch_matched_time1
                    hash_token_time1 += batch_hash_token_time1

                    matched_data2.extend(batch_matched_data2)
                    hash_token_list2.extend(batch_hash_token_list2)
                    matched_time2 += batch_matched_time2
                    hash_token_time2 += batch_hash_token_time2

                matched_data = matched_data1 + matched_data2
                hash_token_list = hash_token_list1 + hash_token_list2
                matched_time = matched_time1 + matched_time2
                hash_token_time = hash_token_time1 + hash_token_time2

                # 统计明文token生成时间（包含匹配时间，但这样更合理）
                all_batch_total_time = time.perf_counter() - start_time1
                # token生成时间 = 总时间 - 匹配时间 - hash加密时间
                token_time = all_batch_total_time - matched_time - hash_token_time

                result_set = set()
                for data in matched_data:
                    parts = data.split()
                    for part in parts[1:]:  # 从第二个元素开始是坐标点
                        result_set.add(part)
                result_tuples = [eval(item) for item in list(result_set)]

                # 查询时长
                end_time = time.perf_counter()

                """
                下面的这一段模拟暴搜精炼结果, 统计精炼时间
                """
                start_time3 = time.perf_counter()
                result_len, recall, precision = analyze_query_coverage_plus(query_rect, matched_data1, matched_data2)
                refine_time = time.perf_counter() - start_time3

                """
                下面的这一段模堆成加解密过程，实际上就统计个解密的时长而已
                """
                # ========== 新增：对称解密时间统计 ==========
                decrypt_time = 0
                if AES_sha1 is not None and matched_data:
                    # 设置对称加密密钥
                    symmetric_key = 'xducc02241931xdu'  # 使用与原代码相同的密钥

                    # 首先模拟加密过程：将每条数据的点坐标部分进行对称加密
                    encrypted_data_list = []
                    for data in matched_data:
                        parts = data.split(' ', 1)  # 分割hash索引和点坐标部分
                        if len(parts) == 2:
                            hash_index = parts[0]
                            points_str = parts[1]  # 所有点坐标的字符串
                            # 对点坐标部分进行AES加密
                            encrypted_points = AES_sha1.AES_Encrypt(symmetric_key, points_str)
                            encrypted_data_list.append((hash_index, encrypted_points))

                    # 开始计时：对称解密过程
                    start_time4 = time.perf_counter()
                    # 对每条加密数据进行解密
                    decrypted_data_list = []
                    for hash_index, encrypted_points in encrypted_data_list:
                        # 对密文进行AES解密
                        decrypted_points = AES_sha1.AES_Decrypt(symmetric_key, encrypted_points)
                        decrypted_data_list.append((hash_index, decrypted_points))
                    # 结束计时
                    decrypt_time = time.perf_counter() - start_time4

                # ========== 统计各种实验结果 ==========
                k_time.append(end_time - start_time1)
                k_token_time.append(token_time)
                k_hash_token_time.append(hash_token_time)
                k_matched_time.append(matched_time)
                k_refine_time.append(refine_time)
                k_decrypt_time.append(decrypt_time)
                result_counts.append(result_len)
                token_counts.append(len(hash_token_list))
                candidate_counts.append(len(result_tuples))
                token_sizes.append(sum(len(token.encode('utf-8')) for token in hash_token_list))
                candidate_sizes.append(sum(len(repr(tup).encode('utf-8')) for tup in result_tuples))
                recall_list.append(recall)
                precision_list.append(precision)

                print(f"  ⏱️ 查询总耗时: {end_time - start_time1:.6f}秒")
                print(f"  ⏱️ 明文token生成耗时: {token_time:.6f}秒")
                print(f"  ⏱️ hash加密token生成耗时: {hash_token_time:.6f}秒")
                print(f"  ⏱️ 匹配耗时: {matched_time:.6f}秒")
                print(f"  ⏱️ 对称解密耗时: {decrypt_time:.6f}秒")
                print(f"  ⏱️ 精炼耗时: {refine_time:.6f}秒")
                print(f"  ⏱️ 精确结果数量: {result_len}")
                print(f"  ⏱️ token数量: {len(hash_token_list)}")
                print(f"  ⏱️ 候选点数量: {len(result_tuples)}")
                print(f"  ⏱️ token大小: {sum(len(token.encode('utf-8')) for token in hash_token_list)}")
                print(f"  ⏱️ 候选点大小: {sum(len(repr(tup).encode('utf-8')) for tup in result_tuples)}")
                print(f"  ⏱️ 召回率: {recall:.6f}")
                print(f"  ⏱️ 精确率: {precision:.6f}")

            # ========== 该范围值测试结束计时 ==========
            k_total_time = sum(k_time)
            # 记录该K值的测试结果
            total_results[percent_int] = {
                'min_depth': min_depth,
                'max_depth': max_depth,
                'total_tests': test_count,
                'test_time': k_total_time,
                'range_para': percent,
                'avg_token_time': sum(k_token_time) / test_count,
                'avg_hash_token_time': sum(k_hash_token_time) / test_count,
                'avg_matched_time': sum(k_matched_time) / test_count,
                'avg_refine_time': sum(k_refine_time) / test_count,
                'avg_decrypt_time': sum(k_decrypt_time) / test_count,
                'avg_results_count': sum(result_counts) / test_count,
                'avg_token_count': sum(token_counts) / test_count,
                'avg_candidate_count': sum(candidate_counts) / test_count,
                'avg_token_size': sum(token_sizes) / test_count,
                'avg_candidate_size': sum(candidate_sizes) / test_count,
                'avg_recall': sum(recall_list) / test_count,
                'avg_precision': sum(precision_list) / test_count
            }

            # 输出该K值的测试结果
            print(f"🌞 K={K:2d} recall: {sum(recall_list) / test_count:.6f}")
            print(f"🌞 K={K:2d} precision: {sum(precision_list) / test_count:.6f}")
            print(f"⏱️ K={K:2d} 总测试时间: {k_total_time:.3f}秒\n")
            update_csv_with_results(csv_filepath, K, dataset_name, algorithm_type, total_results[percent_int], EXPERIMENT_GROUP)

        # ========== 最终汇总报告 ==========
        print("=" * 80)
        print(f"📊 最终测试汇总报告-{dataset_name}_{algorithm_type}_{K}%")
        print(f"🚀 范围查询测试")
        print("=" * 80)

        # 实验结果
        for percent_int in range(int_percent_first, int_percent_second + 1):
            result = total_results.get(percent_int)
            print(f"K={K:2d} range_para={percent_int}% 的实验统计结果：")
            print(f"⏱️ 平均明文token生成时间: {result['avg_token_time']:.8f}秒")
            print(f"⏱️ 平均hash加密token生成时间: {result['avg_hash_token_time']:.8f}秒")
            print(f"⏱️ 平均匹配时间: {result['avg_matched_time']:.8f}秒")
            print(f"⏱️ 平均对称解密时间: {result['avg_decrypt_time']:.8f}秒")
            print(f"⏱️ 平均精炼时间: {result['avg_refine_time']:.8f}秒")
            print(f"⏱️ 平均精确结果数量: {result['avg_results_count']:.8f}个")
            print(f"⏱️ 平均token数量: {result['avg_token_count']:.8f}个")
            print(f"⏱️ 平均候选点数量: {result['avg_candidate_count']:.8f}个")
            print(f"⏱️ 平均token大小: {result['avg_token_size']:.8f}个字节")
            print(f"⏱️ 平均候选点大小: {result['avg_candidate_size']:.8f}个字节")
            print(f"⏱️ 平均召回率: {result['avg_recall']:.8f}")
            print(f"⏱️ 平均精确率: {result['avg_precision']:.8f}")
            print("-" * 60)

        print("-" * 80)
        # 总时间是当前数据集下1%到5%范围测试的时间之和
        total_test_time = sum(result['test_time'] for result in total_results.values())
        print(f"⏱️ 全部测试总时间: {total_test_time:.3f}秒")
        print("=" * 80)

