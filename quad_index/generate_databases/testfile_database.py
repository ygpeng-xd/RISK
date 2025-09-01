"""
生成用于测试的数据集
格式：keyword (x y)
keyword为原始数据集中随机选择的关键词，(x y)为原始数据集中的一个点的±1偏移
注意：选择时要确保keyword至少有knn_num个点
"""


import re
import random


def generate_test_dataset(input_path, output_path, num_queries=100, knn_num=10):
    # 存储候选数据：{keyword: [(x1,y1), (x2,y2)...]}
    candidate_data = {}

    with open(input_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            # 解析行数据
            parts = re.match(r'(\S+)((?:\s+\(\d+\s+\d+\))+)', line)
            if not parts:
                continue

            keyword = parts.group(1)
            coordinates = re.findall(r'\((\d+)\s+(\d+)\)', parts.group(2))

            if len(coordinates) >= knn_num:
                # 转换为整数元组
                points = [(int(x), int(y)) for x, y in coordinates]
                candidate_data[keyword] = points

    # 检查候选数量是否足够
    if len(candidate_data) < num_queries:
        raise ValueError(f"Only {len(candidate_data)} keywords have ≥5 coordinates, need {num_queries}.")

    # 随机选择关键词
    selected_keywords = random.sample(list(candidate_data.keys()), num_queries)

    # 生成带整数偏移的查询点
    results = []
    for keyword in selected_keywords:
        original_points = candidate_data[keyword]
        # 随机选择一个原始点
        original_x, original_y = random.choice(original_points)

        # 生成±1的偏移（确保不与原坐标相同）
        while True:
            offset_x = random.choice([-1, 0, 1])
            offset_y = random.choice([-1, 0, 1])
            if offset_x != 0 or offset_y != 0:  # 至少有一个方向有偏移
                new_x = original_x + offset_x
                new_y = original_y + offset_y
                # 确保新坐标不与任何原始点重合（可选检查）
                if (new_x, new_y) not in original_points:
                    break

        results.append(f"{keyword} ({new_x} {new_y})")

    # 写入输出文件
    with open(output_path, 'w') as file:
        file.write("\n".join(results))


def process_file(input_file, output_file, num_keywords=1):
    """
    处理已有的查询文件，提取坐标和关键词

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        num_keywords: 每行提取的关键词数量，默认为1
    """
    used_keywords = set()  # 记录已使用的关键词
    results = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 3:
                print(f"警告：第{line_num}行数据不足")
                continue

            # 提取坐标（前两个数字）
            coord1, coord2 = parts[0], parts[1]

            # 提取关键词（从第3个开始）
            keywords = parts[2:]

            if len(keywords) < num_keywords:
                print(f"警告：第{line_num}行关键词不足{num_keywords}个，实际只有{len(keywords)}个")
                num_keywords_actual = len(keywords)
            else:
                num_keywords_actual = num_keywords

            # 选择关键词
            selected_keywords = []
            unused_keywords = [kw for kw in keywords if kw not in used_keywords]

            # 优先选择未使用过的关键词
            if len(unused_keywords) >= num_keywords_actual:
                # 有足够的未使用关键词
                selected_keywords = unused_keywords[:num_keywords_actual]
                for kw in selected_keywords:
                    used_keywords.add(kw)
            else:
                # 未使用的关键词不够，先用完未使用的，再补充已使用的
                selected_keywords.extend(unused_keywords)
                for kw in unused_keywords:
                    used_keywords.add(kw)

                # 需要补充的数量
                need_more = num_keywords_actual - len(unused_keywords)
                used_keywords_in_line = [kw for kw in keywords if kw in used_keywords]
                selected_keywords.extend(used_keywords_in_line[:need_more])

                print(f"注意：第{line_num}行部分关键词重复使用")

            # 格式化输出
            if num_keywords == 1:
                # 单关键词格式：keyword (coord1 coord2)
                result = f"{selected_keywords[0]} ({coord1} {coord2})"
            else:
                # 多关键词格式：keyword1,keyword2,... (coord1 coord2)
                keywords_str = ' '.join(selected_keywords)
                result = f"{keywords_str} ({coord1} {coord2})"

            results.append(result)
            print(f"处理第{line_num}行: {result}")

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(result + '\n')

    print(f"\n处理完成！共处理{len(results)}行数据")
    print(f"使用了{len(used_keywords)}个不同的关键词")
    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":
    # input_file = '../../Adatabase/NY_invert.txt'
    # output_file = '../../Adatabase/NY_test.txt'
    # generate_test_dataset(input_file, output_file)

    # input_file = '../../original_dataset/yelp_query_RASK108.txt'
    # output_file = '../../Adatabase/yelp_test.txt'
    # process_file(input_file, output_file)

    input_file = '../../original_dataset/Paris_query_RASK108.txt'
    output_file = '../../Adatabase/paris_test_2.txt'
    process_file(input_file, output_file, num_keywords=2)
