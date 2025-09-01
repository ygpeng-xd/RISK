import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import AES_sha1
import depth_test

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from quad_index import quad_token
from quad_index import quadtree


def match_query_results(query_words, results, hash_dict, initial_bounds, min_depth, max_depth, is_point_query=True):
    matched_lines = []

    # 生成 {哈希: 原始res} 的映射字典
    start_time = time.perf_counter()
    hash_to_res = {
        AES_sha1.sha1_Encrypt(f"{query_words} {res}"): res
        for res in results
    }
    hash_token_time = time.perf_counter() - start_time

    # 匹配哈希并提取原始 res
    for query_hash in hash_to_res.keys():
        if query_hash in hash_dict:
            matched_lines.append(hash_dict[query_hash])

    if is_point_query and len(matched_lines) == 1:
        target_grid_id = hash_to_res[matched_lines[0].split(' ', 1)[0]]
        print(f"  目标网格ID: {target_grid_id}")

        # "-1"编号直接跳出，不进行范围查询
        if target_grid_id == "-1":
            return matched_lines, hash_token_time, list(hash_to_res.keys())

        query_rect = quad_id_to_bounds(target_grid_id, initial_bounds)

        # 🔥 关键优化：根据target_grid_id动态调整max_depth
        target_depth = len(target_grid_id) // 2  # 目标深度
        dynamic_max_depth = min(max_depth, target_depth + 8)  # 限制范围查询深度

        # 执行范围查询
        range_result = quad_token.query_quadtree_range_optimized(
            min_depth, dynamic_max_depth,
            initial_bounds[0], initial_bounds[1],  # x_min, x_max
            initial_bounds[2], initial_bounds[3],  # y_min, y_max
            query_rect
        )

        result_list = [
            item
            for depth in sorted(range_result.keys())
            for item in range_result[depth]
            if item != target_grid_id
        ]

        query_hashes = {AES_sha1.sha1_Encrypt(f"{query_words} {res}") for res in result_list}
        for query_hash in query_hashes:
            if query_hash in hash_dict:
                matched_lines.append(hash_dict[query_hash])

    return matched_lines, hash_token_time, list(hash_to_res.keys())


def quad_id_to_bounds(quad_id, initial_bounds):
    """
    根据四叉树ID反推对应的区域边界

    Args:
        quad_id (str): 由0和1组成的偶数长度字符串
        initial_bounds (list): 初始边界 [x_min, x_max, y_min, y_max]

    Returns:
        tuple: 最终边界 (x_min, x_max, y_min, y_max)
    """
    x_min, x_max, y_min, y_max = initial_bounds

    # 处理特殊编号"-1"，直接返回原始边界
    if quad_id == "-1":
        return x_min, x_max, y_min, y_max

    # 校验逻辑
    # if (len(quad_id) % 2 != 0) or (not set(quad_id) <= {'0', '1'}):
    #     raise ValueError("ID格式错误：必须为偶数长度且只含0/1")

    for i in range(0, len(quad_id), 2):
        y_bit, x_bit = quad_id[i], quad_id[i + 1]  # 直接索引访问（效率提升）

        # 同步计算中点（减少重复计算）
        x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2

        # 单行边界更新（代码更紧凑）
        y_min, y_max = (y_mid, y_max) if y_bit == '1' else (y_min, y_mid)
        x_min, x_max = (x_mid, x_max) if x_bit == '1' else (x_min, x_mid)

    return x_min, x_max, y_min, y_max


def analyze_query_coverage(query_words, query_rect, matched_data, invert_file_path):
    """
    分析范围查询的覆盖率

    Args:
        query_words: 查询关键词
        query_rect: 查询矩形区域 (x_min, x_max, y_min, y_max)
        matched_data: 查询返回的匹配数据列表
        invert_file_path: 倒排索引文件路径
    """
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
    print("\n=== 查询覆盖率分析 ===")
    print(f"应该返回的点数 (范围内原始点): {len(points_in_range_set)}")
    print(f"实际返回的点数: {len(returned_points_set)}")
    print(f"正确返回的点数: {len(correct_returned)}")
    print(f"召回率 (Recall): {recall:.2%}")
    print(f"精确率 (Precision): {precision:.2%}")


if __name__ == '__main__':
    """
    配置区域 只改K就行
    """
    K = 10
    dataset_name = 'yelp'
    if K == 1:
        algorithm_type = 'voronoi'
        not_voronoi_query = False
    else:
        algorithm_type = 'center'  # 'center' 或者 'centroid'
        not_voronoi_query = True
    base_dir = '../../Adatabase'

    # ========== 自动生成路径 ==========
    generate_filepath = f'{base_dir}/{dataset_name}_quadtree_{algorithm_type}{K}.txt'
    name, ext = os.path.splitext(generate_filepath)
    hash_filepath = f'{name}_hash{ext}'

    # ========== 自动计算最大最小深度 ==========
    max_depth, min_depth = [x // 2 for x in depth_test.count_max_min_length(generate_filepath)]
    print(f"最大深度: {max_depth}, 最小深度: {min_depth}")

    with open(hash_filepath, 'r') as hash_file:
        hash_lines = hash_file.readlines()

    # ========== 边界大小 query的关键词、坐标、范围设置 ==========
    x_min, x_max, y_min, y_max = 0, 100000000, 0, 100000000
    initial_bounds = [x_min, x_max, y_min, y_max]

    query_words = 'Fabricators'
    query_point = (978275, 80079087)  # 查询点 x, y
    query_rect = (83593750, 84375000, 91406250, 92187500)  # 查询区域 x_min, x_max, y_min, y_max

    # ========== 计时开始 ==========
    start = time.perf_counter()

    """
    查询开始 确定查询类型 只改下面的 point或者range
    1-'point' 或 2-'range'
    """
    query_type = 1

    # ========== 自动处理逻辑 ==========
    # ========== token生成 ==========
    # point_results result_list都是列表 里面存的是方格的id
    if query_type == 1:  # query_type == 'point'
        point_results = quad_token.query_quadtree(min_depth, max_depth, x_min, x_max, y_min, y_max, query_point)
        step_time = time.perf_counter() - start

        if K == 1:
            # K=1时是最近邻查询，is_point_query为False
            print(f"最近邻查询token条数: {len(point_results)}条")
            print(f"最近邻查询token生成阶段耗时: {step_time}秒")
            query_results = point_results
            is_point_query = False  # K=1时设为False
            query_name = "最近邻查询"
        else:
            # K>1时是K近邻查询
            print(f"{K}近邻查询token条数: {len(point_results)}条")
            print(f"{K}近邻查询token生成阶段耗时: {step_time}秒")
            query_results = point_results
            is_point_query = True
            query_name = f"K = {K}近邻查询"

    else:  # query_type == 'range'
        range_result = quad_token.query_quadtree_range_optimized(min_depth, max_depth, x_min, x_max,
                                                                 y_min, y_max, query_rect)
        result_list = []
        result_list.append("-1")
        for depth in sorted(range_result.keys()):
            result_list += range_result[depth]
        step_time = time.perf_counter() - start
        print(f"范围查询token条数: {len(result_list)}条")
        print(f"范围查询token生成阶段耗时: {step_time}秒")
        query_results = result_list
        is_point_query = False
        query_name = "范围查询"

    # 统一处理
    matched_data, hash_token_time = match_query_results(query_words, query_results, hash_lines, initial_bounds,
                                                        min_depth, max_depth, is_point_query=is_point_query)

    total_time = time.perf_counter() - start

    print(f"{query_name}匹配到的数据: {len(matched_data)}条")
    for data in matched_data:
        print(data)

    print(f"{query_name}耗时:", total_time)

    # 添加覆盖率分析
    if query_type == 2:
        invert_file_path = f'{base_dir}/{dataset_name}_invert.txt'  # 根据你的路径调整
        analyze_query_coverage(query_words, query_rect, matched_data, invert_file_path)

    """
    以下代码用于绘制查询结果
    """
    import InvertedIndex

    filepath = '../../original_dataset/'
    filename = filepath + 'yelp_RASK108.txt'

    inverted_index = InvertedIndex.build_inverted_index_v2(filename)
    points = np.array(inverted_index[query_words])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(points[:, 0], points[:, 1], s=5)

    results = quadtree.quadtree(ax, points, x_min, x_max, y_min, y_max, max_points=K)

    # 绘制查询点
    ax.scatter(query_point[0], query_point[1], s=10, c='r')
    # 绘制查询区域
    rect = plt.Rectangle((query_rect[0], query_rect[2]),  # 左下角坐标 (x_min, y_min)
                         query_rect[1] - query_rect[0],  # 宽度
                         query_rect[3] - query_rect[2],  # 高度
                         edgecolor='b', facecolor='none', linestyle='--', label='Query Region')

    ax.add_patch(rect)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.show()
