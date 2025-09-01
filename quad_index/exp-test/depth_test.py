"""
统计文件中第二列的最长和最短长度
"""


# 将主函数里的代码封装为一个函数
def count_max_min_length(file_path):
    max_length = 0
    min_length = float('inf')
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue  # 跳过不符合格式的行
            second_col = parts[1]
            # 跳过特殊编号"-1"，不计入长度统计
            if second_col == "-1":
                continue
            length = len(second_col)
            if length > max_length:
                max_length = length
            if length < min_length:
                min_length = length
    return max_length, min_length


def analyze_keyword_depth_range(file_path):
    """
    统计每个关键词的深度差值
    """
    keyword_depths = {}

    # 读取文件，按关键词收集所有网格编号
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue

            keyword = parts[0]
            grid_id = parts[1]

            # 跳过特殊编号"-1"
            if grid_id == "-1":
                continue

            # 计算深度：每2位代表1层深度
            depth = len(grid_id) // 2

            if keyword not in keyword_depths:
                keyword_depths[keyword] = []
            keyword_depths[keyword].append(depth)

    # 计算每个关键词的深度差值
    all_ranges = []
    for keyword, depths in keyword_depths.items():
        if not depths:  # 如果该关键词没有有效的网格编号
            continue

        min_depth = min(depths)
        max_depth = max(depths)
        depth_range = max_depth - min_depth
        all_ranges.append(depth_range)

    if all_ranges:
        max_range = max(all_ranges)
        avg_range = sum(all_ranges) / len(all_ranges)

        print(f"最大深度差值: {max_range}")
        print(f"平均深度差值: {avg_range:.2f}")

        # 差值为0的关键词数量（只有一个深度）
        zero_range_count = sum(1 for r in all_ranges if r == 0)
        print(f"深度差值为0的关键词: {zero_range_count} 个 ({zero_range_count/len(all_ranges)*100:.1f}%)")

    return sum(all_ranges) / len(all_ranges)


if __name__ == '__main__':
    file_path = '../../Adatabase/paris_quadtree_center2.txt'

    max_length, min_length = count_max_min_length(file_path)

    print(f"最长编号长度: {max_length}")
    print(f"最短编号长度: {min_length}")

    avg_range = analyze_keyword_depth_range(file_path)
