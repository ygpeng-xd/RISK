import time
import itertools


# 单点查询
def find_quadrant(x, y, x_min, x_max, y_min, y_max):
    """
    根据查询点的坐标 (x, y) 和当前方格的边界，确定它所在的象限编号。
    """
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2

    if x >= x_mid and y >= y_mid:
        return "11"  # 右上
    elif x < x_mid and y >= y_mid:
        return "10"  # 左上
    elif x < x_mid and y < y_mid:
        return "00"  # 左下
    else:
        return "01"  # 右下


# 单点查询--最小划分到最大划分层查询点分别所在的小方格
def query_quadtree(min_depth, max_depth, x_min, x_max, y_min, y_max, query_point):
    """
    优化后的查询算法：逐层递进生成结果，避免重复计算。
    """
    x, y = query_point
    results = []

    # 固定添加 "-1" 编号，用于匹配点数不足的关键词记录
    results.append("-1")

    # 初始化当前层的边界和象限编号
    current_x_min, current_x_max = x_min, x_max
    current_y_min, current_y_max = y_min, y_max
    current_quadrant_id = ""

    if min_depth == 0:
        results.append(current_quadrant_id)

    # 生成从 depth=1 到 max_depth 的所有层
    for depth in range(1, max_depth + 1):
        # 确定当前层的象限
        quadrant = find_quadrant(x, y, current_x_min, current_x_max, current_y_min, current_y_max)
        current_quadrant_id += quadrant

        # 更新边界以进入下一层
        x_mid = (current_x_min + current_x_max) / 2
        y_mid = (current_y_min + current_y_max) / 2
        # print(x_mid, y_mid, quadrant)

        if quadrant == "11":
            current_x_min, current_y_min = x_mid, y_mid
        elif quadrant == "10":
            current_x_max, current_y_min = x_mid, y_mid
        elif quadrant == "00":
            current_x_max, current_y_max = x_mid, y_mid
        elif quadrant == "01":
            current_x_min, current_y_max = x_mid, y_mid

        # 如果当前层数 >= min_depth，记录结果
        if depth >= min_depth:
            results.append(current_quadrant_id)

    return results


# 单点查询--找所在小方格的邻居小方格
# 后经验证 这个方法存在缺陷 找到的邻居小方格不全或者有多余的小方格
def get_neighbor_quadrants(quadrant_id, x_min, x_max, y_min, y_max):
    """
    根据当前小方格的编号和边界，计算其所有邻居小方格的编号。
    """
    neighbors = []

    if len(quadrant_id) == 0:
        neighbors.append('')
        return neighbors

    depth = len(quadrant_id) // 2
    current_x_min, current_x_max = x_min, x_max
    current_y_min, current_y_max = y_min, y_max

    # 计算当前小方格的边界
    for i in range(0, len(quadrant_id), 2):
        x_mid = (current_x_min + current_x_max) / 2
        y_mid = (current_y_min + current_y_max) / 2
        quadrant = quadrant_id[i:i + 2]

        if quadrant == "11":
            current_x_min, current_y_min = x_mid, y_mid
        elif quadrant == "10":
            current_x_max, current_y_min = x_mid, y_mid
        elif quadrant == "00":
            current_x_max, current_y_max = x_mid, y_mid
        elif quadrant == "01":
            current_x_min, current_y_max = x_mid, y_mid

    # 计算邻居小方格的边界
    width = current_x_max - current_x_min
    height = current_y_max - current_y_min

    # 八个可能的邻居方向
    directions = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1)
    ]

    for dx, dy in directions:
        neighbor_x_min = current_x_min + dx * width
        neighbor_x_max = current_x_max + dx * width
        neighbor_y_min = current_y_min + dy * height
        neighbor_y_max = current_y_max + dy * height

        # 检查邻居小方格是否在整体边界内
        if (neighbor_x_min >= x_min and neighbor_x_max <= x_max and
                neighbor_y_min >= y_min and neighbor_y_max <= y_max):
            # 生成邻居小方格的编号
            neighbor_quadrant_id = ""
            temp_x_min, temp_x_max = x_min, x_max
            temp_y_min, temp_y_max = y_min, y_max

            for _ in range(depth):
                x_mid = (temp_x_min + temp_x_max) / 2
                y_mid = (temp_y_min + temp_y_max) / 2

                if neighbor_x_min >= x_mid and neighbor_y_min >= y_mid:
                    neighbor_quadrant_id += "11"
                    temp_x_min, temp_y_min = x_mid, y_mid
                elif neighbor_x_max <= x_mid and neighbor_y_min >= y_mid:
                    neighbor_quadrant_id += "10"
                    temp_x_max, temp_y_min = x_mid, y_mid
                elif neighbor_x_max <= x_mid and neighbor_y_max <= y_mid:
                    neighbor_quadrant_id += "00"
                    temp_x_max, temp_y_max = x_mid, y_mid
                else:
                    neighbor_quadrant_id += "01"
                    temp_x_min, temp_y_max = x_mid, y_mid

            neighbors.append(neighbor_quadrant_id)

    return neighbors


def find_neighbors(results, x_min, x_max, y_min, y_max):
    """
    找到查询点所在小方格的所有邻居小方格。
    """
    neighbors = []

    for quadrant_id in results:
        neighbors.extend(get_neighbor_quadrants(quadrant_id, x_min, x_max, y_min, y_max))

    return neighbors


# 范围查询函数
def is_overlapping(rect1, rect2):
    """
    判断两个矩形是否有重叠。
    rect格式: (x_min, x_max, y_min, y_max)
    """
    x_min1, x_max1, y_min1, y_max1 = rect1
    x_min2, x_max2, y_min2, y_max2 = rect2
    return not (x_max1 < x_min2 or x_min1 > x_max2 or y_max1 < y_min2 or y_min1 > y_max2)


def query_quadtree_range(min_depth, max_depth, x_min, x_max, y_min, y_max, query_rect):
    """
    范围查询的迭代实现：逐层递进，动态生成与查询区域重叠的方格编号。
    """
    results = {}  # 按层存储结果 {depth: [quadrant_ids]}
    queue = []  # 队列元素：(depth, x_min, x_max, y_min, y_max, quadrant_id)

    # 初始化队列（处理 depth=0 的情况）
    initial_overlap = is_overlapping((x_min, x_max, y_min, y_max), query_rect)
    if min_depth <= 0 <= max_depth and initial_overlap:
        results[0] = [""]
    if max_depth >= 1 and initial_overlap:
        queue.append((1, x_min, x_max, y_min, y_max, ""))

    # 逐层处理队列中的方格
    while queue:
        depth, curr_x_min, curr_x_max, curr_y_min, curr_y_max, quad_id = queue.pop(0)

        # 如果当前层超过 max_depth，跳过
        if depth > max_depth:
            continue

        # 划分当前方格为四个子方格
        x_mid = (curr_x_min + curr_x_max) / 2
        y_mid = (curr_y_min + curr_y_max) / 2
        sub_quadrants = [
            ("11", x_mid, curr_x_max, y_mid, curr_y_max),  # 右上
            ("10", curr_x_min, x_mid, y_mid, curr_y_max),  # 左上
            ("00", curr_x_min, x_mid, curr_y_min, y_mid),  # 左下
            ("01", x_mid, curr_x_max, curr_y_min, y_mid)  # 右下
        ]

        for suffix, sub_x_min, sub_x_max, sub_y_min, sub_y_max in sub_quadrants:
            sub_quad_id = quad_id + suffix
            sub_rect = (sub_x_min, sub_x_max, sub_y_min, sub_y_max)

            # 检查子方格是否与查询区域重叠
            if is_overlapping(sub_rect, query_rect):
                # 如果当前层在 min_depth 到 max_depth 之间，记录结果
                if min_depth <= depth <= max_depth:
                    if depth not in results:
                        results[depth] = []
                    results[depth].append(sub_quad_id)
                # 如果未到最大层，将子方格加入队列
                if depth < max_depth:
                    queue.append((depth + 1, sub_x_min, sub_x_max, sub_y_min, sub_y_max, sub_quad_id))

    # 按层整理结果（填充未达到 min_depth 的层）
    formatted_results = {}
    for depth in range(min_depth, max_depth + 1):
        formatted_results[depth] = results.get(depth, [])
    return formatted_results


# 范围查询函数--优化
def is_fully_inside(rect, query_rect):
    x_min, x_max, y_min, y_max = rect
    q_x_min, q_x_max, q_y_min, q_y_max = query_rect
    return (x_min >= q_x_min) and (x_max <= q_x_max) and (y_min >= q_y_min) and (y_max <= q_y_max)


def query_quadtree_range_optimized(min_depth, max_depth, x_min, x_max, y_min, y_max, query_rect):
    results = {}
    queue = []

    initial_rect = (x_min, x_max, y_min, y_max)
    initial_inside = is_fully_inside(initial_rect, query_rect)
    initial_overlap = is_overlapping(initial_rect, query_rect)

    if initial_inside:
        for l in range(0, max_depth + 1):
            if l < min_depth:
                continue
            k = l
            suffixes = itertools.product(['00', '01', '10', '11'], repeat=k)
            quad_ids = [''.join(s) for s in suffixes]
            results[l] = quad_ids
    # 先把深度为1的加入
    elif initial_overlap:
        if min_depth <= 0 <= max_depth:
            results[0] = [""]
        if max_depth >= 1:
            x_mid = (x_min + x_max) / 2
            y_mid = (y_min + y_max) / 2
            sub_quadrants = [
                ("11", x_mid, x_max, y_mid, y_max),
                ("10", x_min, x_mid, y_mid, y_max),
                ("00", x_min, x_mid, y_min, y_mid),
                ("01", x_mid, x_max, y_min, y_mid)
            ]
            for suffix, sub_x_min, sub_x_max, sub_y_min, sub_y_max in sub_quadrants:
                if is_overlapping((sub_x_min, sub_x_max, sub_y_min, sub_y_max), query_rect):
                    queue.append((1, sub_x_min, sub_x_max, sub_y_min, sub_y_max, suffix))

    while queue:
        depth, curr_x_min, curr_x_max, curr_y_min, curr_y_max, quad_id = queue.pop(0)
        current_rect = (curr_x_min, curr_x_max, curr_y_min, curr_y_max)
        # 如果当前小方格完全在查询范围内，则直接将该小方格递归到最大深度并保存结果
        if is_fully_inside(current_rect, query_rect):
            for l in range(depth, max_depth + 1):
                if l < min_depth:
                    continue
                k = l - depth
                suffixes = itertools.product(['00', '01', '10', '11'], repeat=k)
                quad_ids = [quad_id + ''.join(s) for s in suffixes]
                if l not in results:
                    results[l] = []
                results[l].extend(quad_ids)
            continue
        overlap = is_overlapping(current_rect, query_rect)
        if not overlap:
            continue
        if min_depth <= depth <= max_depth:
            if depth not in results:
                results[depth] = []
            results[depth].append(quad_id)
        if depth < max_depth:
            x_mid = (curr_x_min + curr_x_max) / 2
            y_mid = (curr_y_min + curr_y_max) / 2
            sub_quadrants = [("11", x_mid, curr_x_max, y_mid, curr_y_max),
                             ("10", curr_x_min, x_mid, y_mid, curr_y_max),
                             ("00", curr_x_min, x_mid, curr_y_min, y_mid),
                             ("01", x_mid, curr_x_max, curr_y_min, y_mid)]
            for suffix, sub_x_min, sub_x_max, sub_y_min, sub_y_max in sub_quadrants:
                sub_quad_id = quad_id + suffix
                sub_rect = (sub_x_min, sub_x_max, sub_y_min, sub_y_max)
                if is_overlapping(sub_rect, query_rect):
                    queue.append((depth + 1, sub_x_min, sub_x_max, sub_y_min, sub_y_max, sub_quad_id))

    formatted_results = {depth: [] for depth in range(min_depth, max_depth + 1)}
    for depth in range(min_depth, max_depth + 1):
        formatted_results[depth] = results.get(depth, [])
    return formatted_results


def query_quadtree_range_optimized_batch(min_depth, max_depth, x_min, x_max, y_min, y_max, query_rect, batch_size=10000):
    """
    分批返回查询结果的生成器函数
    """
    current_batch = {}
    current_batch_count = 0

    def add_to_batch(depth, quad_ids):
        nonlocal current_batch_count

        for quad_id in quad_ids:
            # 每次添加前都检查depth是否存在
            if depth not in current_batch:
                current_batch[depth] = []

            current_batch[depth].append(quad_id)
            current_batch_count += 1

            # 如果达到批次大小，yield当前批次
            if current_batch_count >= batch_size:
                yield current_batch.copy()
                current_batch.clear()
                current_batch_count = 0

    def generate_suffixes_batch(k, quad_id_prefix, target_depth):
        """分批生成suffixes"""
        if k == 0:
            yield [quad_id_prefix]
            return

        batch = []
        suffixes = itertools.product(['00', '01', '10', '11'], repeat=k)

        for suffix in suffixes:
            full_quad_id = quad_id_prefix + ''.join(suffix)
            batch.append(full_quad_id)

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:  # 处理剩余的
            yield batch

    queue = []
    initial_rect = (x_min, x_max, y_min, y_max)
    initial_inside = is_fully_inside(initial_rect, query_rect)
    initial_overlap = is_overlapping(initial_rect, query_rect)

    if initial_inside:
        for l in range(0, max_depth + 1):
            if l < min_depth:
                continue
            k = l
            # 分批生成和处理suffixes
            for suffix_batch in generate_suffixes_batch(k, "", l):
                yield from add_to_batch(l, suffix_batch)
    elif initial_overlap:
        if min_depth <= 0 <= max_depth:
            yield from add_to_batch(0, [""])

        if max_depth >= 1:
            x_mid = (x_min + x_max) / 2
            y_mid = (y_min + y_max) / 2
            sub_quadrants = [
                ("11", x_mid, x_max, y_mid, y_max),
                ("10", x_min, x_mid, y_mid, y_max),
                ("00", x_min, x_mid, y_min, y_mid),
                ("01", x_mid, x_max, y_min, y_mid)
            ]
            for suffix, sub_x_min, sub_x_max, sub_y_min, sub_y_max in sub_quadrants:
                if is_overlapping((sub_x_min, sub_x_max, sub_y_min, sub_y_max), query_rect):
                    queue.append((1, sub_x_min, sub_x_max, sub_y_min, sub_y_max, suffix))

    while queue:
        depth, curr_x_min, curr_x_max, curr_y_min, curr_y_max, quad_id = queue.pop(0)
        current_rect = (curr_x_min, curr_x_max, curr_y_min, curr_y_max)

        if is_fully_inside(current_rect, query_rect):
            for l in range(depth, max_depth + 1):
                if l < min_depth:
                    continue
                k = l - depth
                # 分批处理完全包含的区域
                for suffix_batch in generate_suffixes_batch(k, quad_id, l):
                    yield from add_to_batch(l, suffix_batch)
            continue

        overlap = is_overlapping(current_rect, query_rect)
        if not overlap:
            continue

        if min_depth <= depth <= max_depth:
            yield from add_to_batch(depth, [quad_id])

        if depth < max_depth:
            x_mid = (curr_x_min + curr_x_max) / 2
            y_mid = (curr_y_min + curr_y_max) / 2
            sub_quadrants = [("11", x_mid, curr_x_max, y_mid, curr_y_max),
                             ("10", curr_x_min, x_mid, y_mid, curr_y_max),
                             ("00", curr_x_min, x_mid, curr_y_min, y_mid),
                             ("01", x_mid, curr_x_max, curr_y_min, y_mid)]
            for suffix, sub_x_min, sub_x_max, sub_y_min, sub_y_max in sub_quadrants:
                sub_quad_id = quad_id + suffix
                sub_rect = (sub_x_min, sub_x_max, sub_y_min, sub_y_max)
                if is_overlapping(sub_rect, query_rect):
                    queue.append((depth + 1, sub_x_min, sub_x_max, sub_y_min, sub_y_max, sub_quad_id))

    # 返回最后的批次
    if current_batch_count > 0:
        yield current_batch


# 比较两个列表是否相等 元素顺序无关
def are_lists_equal(list1, list2):
    return set(list1) == set(list2)


if __name__ == "__main__":
    # 测试参数
    min_depth = 1
    max_depth = 15
    query_point = (4.5, 6.5)
    x_min, x_max, y_min, y_max = 0, 100000000, 0, 100000000

    start = time.time()

    my_results = query_quadtree(min_depth, max_depth, x_min, x_max, y_min, y_max, query_point)

    neighbors = find_neighbors(my_results, x_min, x_max, y_min, y_max)

    print("result:", my_results)
    print("邻居小方格:", neighbors)

    query_rect = (0, 25000000, 75000000, 100000000)  # 查询区域 x_min, x_max, y_min, y_max
    # results_1 = query_quadtree_range(min_depth, max_depth, x_min, x_max, y_min, y_max, query_rect)
    results_optimized = query_quadtree_range_optimized(min_depth, max_depth, x_min, x_max, y_min, y_max, query_rect)

    total_time = time.time() - start

    # 比较结果
    # for depth in sorted(results_1.keys()):
    #     bool_results = are_lists_equal(results_1[depth], results_optimized[depth])
    #     print(f"Depth {depth}: {bool_results}")

    for depth in sorted(results_optimized.keys()):
        print(f"Depth {depth}: {results_optimized[depth]}")
        print(len(results_optimized[depth]))
    print("总耗时:", total_time)
