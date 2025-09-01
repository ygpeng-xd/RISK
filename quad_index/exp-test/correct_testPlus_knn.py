import numpy as np
import pandas as pd
import time
import sys
import os

import InvertedIndex
import depth_test
import query_test
import AES_sha1

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from quad_index import knn
from quad_index import quad_token


def load_test_data(filepath):
    """加载测试数据集（支持多关键词格式）"""
    keywords_list = []  # 存储每行的关键词列表
    points = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue

            # 找到最后一个左括号的位置
            last_paren_index = line.rfind('(')
            if last_paren_index == -1:
                print(f"警告：无法解析行: {line}")
                continue

            # 分离关键词部分和坐标部分
            keywords_part = line[:last_paren_index].strip()
            coord_part = line[last_paren_index:].strip()

            # 解析关键词（用空格分割）
            keywords = keywords_part.split()

            # 解析坐标（去除括号）
            coord_str = coord_part.strip('()')
            x, y = coord_str.split()

            keywords_list.append(keywords)
            points.append([int(x), int(y)])

    return keywords_list, np.array(points)


def format_scientific(value, decimal_places=4):
    """
    将数值格式化为科学计数法，保留指定小数位数
    例如：0.00002557 -> "2.5570e-05"
    返回float类型 这样Excel能正确识别为数值而不是文本
    """
    if value == 0:
        return "0.0000e+00"
    return float(f"{value:.{decimal_places}e}")


def update_csv_with_results(csv_filepath, K, dataset_name, algorithm_type, use_same_file_strategy, same_file_k, result,
                            group):
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
    k_row_index = df[(df['K'] == K) & (df['group'] == group)].index
    if len(k_row_index) == 0:
        # 创建新行数据
        print(f"🆕 创建新行: K={K}, group={group}")
        # 获取新行的索引（下一个可用的索引）
        k_row_index = len(df)
        # 使用loc直接创建新行，避免concat警告
        df.loc[k_row_index, 'K'] = K
        df.loc[k_row_index, 'group'] = group
        # 初始化其他列为适当默认值
        for col in df.columns:
            if col not in ['K', 'group']:
                df.loc[k_row_index, col] = None
    else:
        # 如果找到了对应的行，获取第一个匹配的行索引
        k_row_index = k_row_index[0]
        print(f"🔄 更新现有行: K={K}, group={group}")

    # 填写基本信息
    df.loc[k_row_index, 'database'] = dataset_name
    df.loc[k_row_index, 'KNN_query'] = 1  # KNN查询固定为1
    df.loc[k_row_index, 'query_number'] = result['total_tests']
    df.loc[k_row_index, 'tree_min_depth'] = result['min_depth']
    df.loc[k_row_index, 'tree_max_depth'] = result['max_depth']

    # 根据文件使用策略填写use_*_file字段
    if use_same_file_strategy:
        # 使用各自的文件策略
        if K == 1 and algorithm_type == 'voronoi':
            df.loc[k_row_index, 'use_voronoi_file'] = 1
        elif algorithm_type == 'center':
            df.loc[k_row_index, 'use_center_file'] = 1
        elif algorithm_type == 'centroid':
            df.loc[k_row_index, 'use_centroid_file'] = 1

    else:
        df.loc[k_row_index, 'use_same_file'] = 1
        # use_voronoi_file不填（因为不会使用voronoi文件）
        # 注意：在同一文件策略下，algorithm_type不会是'voronoi'
        # 根据algorithm_type填写对应的文件使用标记
        if algorithm_type == 'center':
            df.loc[k_row_index, 'use_center_file'] = same_file_k
        elif algorithm_type == 'centroid':
            df.loc[k_row_index, 'use_centroid_file'] = same_file_k

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
    df.loc[k_row_index, 'ratio_avg'] = round(result['ratio'], 4)
    # KNN查询不填写recall和precision（保持为空）

    # 填写最后修改时间
    df.loc[k_row_index, 'last_modify_time'] = int(time.strftime("%d%H%M%S", time.localtime()))

    # 保存更新后的CSV文件
    df.to_csv(csv_filepath, index=False, encoding='utf-8', sep=',')
    print(f"✅ 已更新CSV文件中K={K}的统计结果")


if __name__ == '__main__':
    original_filepath = '../../original_dataset/Paris_RASK108.txt'
    test_filepath = '../../Adatabase/paris_test_2.txt'
    csv_filepath = '../../exp_result/data_statistics0828.csv'

    """
    配置区域 - 自动测试多个K值
    """
    K_values = [2, 4, 6, 8, 10]  # 要测试的K值列表
    dataset_name = 'paris'
    algorithm_type = 'center'  # 'center' 或者 'centroid'
    base_dir = '../../Cdatabase'

    # 实验组配置
    EXPERIMENT_GROUP = 3

    # 控制是否用各自的文件
    # use_own_file = True
    use_own_file = False
    same_file_k = 80

    x_min, x_max, y_min, y_max = 0, 100000000, 0, 100000000
    initial_bounds = [x_min, x_max, y_min, y_max]

    inverted_index = InvertedIndex.build_inverted_index_v2(original_filepath)
    test_keywords_list, test_points = load_test_data(test_filepath)

    # ========== 汇总结果统计 ==========
    total_results = {}  # 存储每个K值的测试结果

    # ========== 循环测试每个K值 ==========
    for K in K_values:
        print("=" * 80)
        print(f"🔍 开始测试 K={K:2d}")
        print("=" * 80)

        if use_own_file:
            # ========== 自动生成路径 ==========
            if K == 1:
                algorithm_type = 'voronoi'
                not_voronoi_query = False
            else:
                algorithm_type = 'center'  # 'center' 或者 'centroid'
                not_voronoi_query = True

            generate_filepath = f'{base_dir}/{dataset_name}_quadtree_{algorithm_type}{K}.txt'
            name, ext = os.path.splitext(generate_filepath)
            hash_filepath = f'{name}_hash{ext}'

        else:
            # ========== 全部用K=10的文件 ==========
            generate_filepath = f'{base_dir}/{dataset_name}_quadtree_{algorithm_type}{same_file_k}.txt'
            hash_filepath = f'{base_dir}/{dataset_name}_quadtree_{algorithm_type}{same_file_k}_hash.txt'
            not_voronoi_query = True

        # 检查文件是否存在
        if not os.path.exists(generate_filepath) or not os.path.exists(hash_filepath):
            print(f"❌ K={K:2d} 的文件不存在，跳过测试")
            total_results[K] = "文件不存在"
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
                        hash_key = parts[0]  # c479d5c58198ab4141ea0a071290341566bcb1e6
                        hash_dict[hash_key] = line  # 保存完整行

        # 进行批量查询
        test_count = 0
        # ========== 该K值测试的各种时间统计 ==========
        k_time = []
        k_token_time = []  # 统计明文token生成时间
        k_hash_token_time = []  # 统计hash_token生成时间
        k_matched_time = []  # 统计匹配时间
        k_refine_time = []  # 统计精炼结果时间
        k_decrypt_time = []  # 统计对称解密时间
        # ========== 统计该K值测试的候选点数量 ==========
        result_counts = []  # 统计当前K值100次查询精确结果的平均点数量
        token_counts = []  # 统计当前K值100次查询token数量
        candidate_counts = []  # 统计当前K值100次查询候选集的平均点数量
        token_sizes = []  # 统计当前K值100次查询token大小--字节数
        candidate_sizes = []  # 统计当前K值100次查询候选集大小--字节数
        # ========== 精确率 ==========
        ratio_list = []  # 统计ration

        for keywords, point in zip(test_keywords_list, test_points):
            test_count += 1
            keyword1 = keywords[0]
            keyword2 = keywords[1]
            print(f"[{test_count}/{len(test_keywords_list)}]--查询点: {point}")

            # 获取第一个关键词对应的点
            points1 = np.array(inverted_index[keyword1])
            # 获取实际点数并调整K值
            available_points1 = len(np.unique(points1, axis=0))
            actual_k1 = min(K, available_points1)
            kd_tree1 = knn.build_kdtree(np.unique(points1, axis=0))
            neighbors1 = knn.find_k_nearest_neighbors(kd_tree1, point, actual_k1)
            nearest_points1 = [(int(neighbor[0]), int(neighbor[1])) for neighbor, _ in neighbors1]

            # 获取第二个关键词对应的点
            points2 = np.array(inverted_index[keyword2])
            available_points2 = len(np.unique(points2, axis=0))
            actual_k2 = min(K, available_points2)
            kd_tree2 = knn.build_kdtree(np.unique(points2, axis=0))
            neighbors2 = knn.find_k_nearest_neighbors(kd_tree2, point, actual_k2)
            nearest_points2 = [(int(neighbor[0]), int(neighbor[1])) for neighbor, _ in neighbors2]

            accurate_nearest_points = list(set(nearest_points1).intersection(set(nearest_points2)))

            query_point = tuple(point)

            # ========== 开始计时 ==========
            start_time1 = time.perf_counter()
            point_results = quad_token.query_quadtree(min_depth, max_depth, x_min, x_max, y_min, y_max, query_point)
            # 统计明文token生成时间
            token_time = time.perf_counter() - start_time1

            # 第一个关键词进行匹配
            start_time2 = time.perf_counter()
            # hash_token_time用于统计hash加密token生成时间
            print(f"  关键词1: {keyword1}")
            matched_data1, hash_token_time1, hash_token_list1 = query_test.match_query_results(keyword1, point_results,
                                                                                            hash_dict,
                                                                                            initial_bounds,
                                                                                            min_depth, max_depth,
                                                                                            not_voronoi_query)
            # 统计匹配时间
            matched_time_with_hash_token1 = time.perf_counter() - start_time2
            matched_time1 = matched_time_with_hash_token1 - hash_token_time1

            # 第二个关键词进行匹配
            start_time2e = time.perf_counter()
            print(f"  关键词2: {keyword2}")
            matched_data2, hash_token_time2, hash_token_list2 = query_test.match_query_results(keyword2, point_results,
                                                                                               hash_dict,
                                                                                               initial_bounds,
                                                                                               min_depth, max_depth,
                                                                                               not_voronoi_query)
            matched_time_with_hash_token2 = time.perf_counter() - start_time2e
            matched_time2 = matched_time_with_hash_token2 - hash_token_time2

            # 合并两个关键词的匹配结果
            matched_data = matched_data1 + matched_data2
            hash_token_list = hash_token_list1 + hash_token_list2
            hash_token_time = hash_token_time1 + hash_token_time2
            matched_time = matched_time1 + matched_time2

            start_time3 = time.perf_counter()
            result_set = set()
            for data in matched_data:
                parts = data.split()
                for part in parts[1:]:  # 从第二个元素开始是坐标点
                    result_set.add(part)

            result_tuples = [eval(item) for item in list(result_set)]

            # 查询时长
            end_time = time.perf_counter()

            """
            下面的这一段模拟暴搜精炼结果 实际上就联合上面的start_time3统计个时长而已
            """
            if len(result_tuples) == 0:
                refined_knn = []
            elif len(result_tuples) <= K:
                # 如果结果集数量不足K个，全部返回并按距离排序
                distances = [(pt, ((pt[0] - point[0]) ** 2 + (pt[1] - point[1]) ** 2) ** 0.5) for pt in result_tuples]
                distances.sort(key=lambda x: x[1])
                refined_knn = [pt for pt, dist in distances]
            else:
                # 计算距离并选择最近的K个点
                distances = [(pt, ((pt[0] - point[0]) ** 2 + (pt[1] - point[1]) ** 2) ** 0.5) for pt in result_tuples]
                distances.sort(key=lambda x: x[1])
                refined_knn = [pt for pt, dist in distances[:K]]
            # 统计精炼时间
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

            # ========== 新加ration计算片段 ==========
            if len(accurate_nearest_points) == 0:
                k_ratio = 1.0
                ratio_list.append(k_ratio)
            else:
                # 检查accurate_nearest_points是否为result_tuples的子集
                if set(accurate_nearest_points).issubset(set(result_tuples)):
                    k_ratio = 1.0
                    ratio_list.append(k_ratio)
                else:
                    k_ratio = 0.0
                    ratio_list.append(k_ratio)

            # ========== 统计各种实验结果 ==========
            k_time.append(end_time - start_time1)
            k_token_time.append(token_time)
            k_hash_token_time.append(hash_token_time)
            k_matched_time.append(matched_time)
            k_refine_time.append(refine_time)
            k_decrypt_time.append(decrypt_time)
            result_counts.append(len(accurate_nearest_points))
            token_counts.append(len(hash_token_list))
            candidate_counts.append(len(result_tuples))
            token_sizes.append(sum(len(token.encode('utf-8')) for token in hash_token_list))
            candidate_sizes.append(sum(len(repr(tup).encode('utf-8')) for tup in result_tuples))

            print(f"  ⏱️ 查询总耗时: {end_time - start_time1:.6f}秒")
            print(f"  ⏱️ 明文token生成耗时: {token_time:.6f}秒")
            print(f"  ⏱️ hash加密token生成耗时: {hash_token_time:.6f}秒")
            print(f"  ⏱️ 匹配耗时: {matched_time:.6f}秒")
            print(f"  ⏱️ 对称解密耗时: {decrypt_time:.6f}秒")
            print(f"  ⏱️ 精炼耗时: {refine_time:.6f}秒")
            print(f"  ⏱️ 精确解数量: {len(accurate_nearest_points)}")
            print(f"  ⏱️ token数量: {len(hash_token_list)}")
            print(f"  ⏱️ 候选点数量: {len(result_tuples)}")
            print(f"  ⏱️ token大小: {sum(len(token.encode('utf-8')) for token in hash_token_list)}")
            print(f"  ⏱️ 候选点大小: {sum(len(repr(tup).encode('utf-8')) for tup in result_tuples)}")
            print(f"  ⏱️ k_ratio计算结果: {k_ratio:.6f}\n")

        # ========== 该K值测试结束计时 ==========
        k_total_time = sum(k_time)

        # 记录该K值的测试结果
        total_results[K] = {
            'min_depth': min_depth,
            'max_depth': max_depth,
            'algorithm_type': algorithm_type,
            'total_tests': test_count,
            'test_time': k_total_time,
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
            'ratio': sum(ratio_list) / len(ratio_list)  # 计算该K值下的ration计算结果
        }

        # 输出该K值的测试结果
        print(f"🌞 K={K:2d} ratio: {sum(ratio_list) / len(ratio_list):.6f}")
        print(f"⏱️ K={K:2d} 总测试时间: {k_total_time:.3f}秒\n")

    # ========== 最终汇总报告 ==========
    print("=" * 80)
    print(f"📊 最终测试汇总报告-{algorithm_type}")
    print("=" * 80)

    # 实验结果
    for K in K_values:
        result = total_results.get(K)
        print(f"K={K:2d}的实验统计结果：")
        print(f"⏱️ 平均明文token生成时间: {result['avg_token_time']:.8f}秒")
        print(f"⏱️ 平均hash加密token生成时间: {result['avg_hash_token_time']:.8f}秒")
        print(f"⏱️ 平均匹配时间: {result['avg_matched_time']:.8f}秒")
        print(f"⏱️ 平均对称解密时间: {result['avg_decrypt_time']:.8f}秒")
        print(f"⏱️ 平均精炼时间: {result['avg_refine_time']:.8f}秒")
        print(f"⏱️ 平均精确解数量: {result['avg_results_count']:.8f}个")
        print(f"⏱️ 平均token数量: {result['avg_token_count']:.8f}个")
        print(f"⏱️ 平均候选点数量: {result['avg_candidate_count']:.8f}个")
        print(f"⏱️ 平均token大小: {result['avg_token_size']:.8f}个字节")
        print(f"⏱️ 平均候选点大小: {result['avg_candidate_size']:.8f}个字节")
        print(f"⏱️ ratio计算结果: {result['ratio']:.6f}")
        # ========== 新增：更新CSV文件 ==========
        update_csv_with_results(csv_filepath, K, dataset_name, result['algorithm_type'], use_own_file, same_file_k, result, EXPERIMENT_GROUP)
        print("-" * 60)

    print("-" * 80)
    # 总时间是所有K值测试的时间之和
    total_test_time = sum(result['test_time'] for result in total_results.values())
    print(f"⏱️ 全部测试总时间: {total_test_time:.3f}秒")
    print("=" * 80)
