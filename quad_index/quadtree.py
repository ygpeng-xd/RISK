import matplotlib.pyplot as plt
import numpy as np
from rtree import index
import time


def quadtree(divide_line, points, x_min, x_max, y_min, y_max, max_points=2, results=None, quad_id="", depth=0):
    if results is None:
        results = []

    # 如果点数小于等于 max_points，记录该小方格的信息
    if len(points) <= max_points:
        results.append({
            "id": quad_id,  # 小方格唯一编号
            "bounds": [x_min, y_min, x_max, y_max],
            "points": points.tolist(),
            "depth": depth  # 记录当前划分层数
        })
        return results

    # 计算中点
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2

    divide_line.plot([x_mid, x_mid], [y_min, y_max], 'k-')  # 垂直线
    divide_line.plot([x_min, x_max], [y_mid, y_mid], 'k-')  # 水平线

    # 按象限划分点
    q1 = points[(points[:, 0] >= x_mid) & (points[:, 1] >= y_mid)]  # 右上
    q2 = points[(points[:, 0] < x_mid) & (points[:, 1] >= y_mid)]  # 左上
    q3 = points[(points[:, 0] < x_mid) & (points[:, 1] < y_mid)]  # 左下
    q4 = points[(points[:, 0] >= x_mid) & (points[:, 1] < y_mid)]  # 右下

    # 递归处理每个象限并添加编号，同时增加层数
    quadtree(divide_line, q1, x_mid, x_max, y_mid, y_max, max_points, results, quad_id + "11", depth + 1)
    quadtree(divide_line, q2, x_min, x_mid, y_mid, y_max, max_points, results, quad_id + "10", depth + 1)
    quadtree(divide_line, q3, x_min, x_mid, y_min, y_mid, max_points, results, quad_id + "00", depth + 1)
    quadtree(divide_line, q4, x_mid, x_max, y_min, y_mid, max_points, results, quad_id + "01", depth + 1)

    return results


# 计算邻居函数--朴素方法
def compute_neighbors(results):
    for grid in results:
        grid["neighbors"] = []  # 初始化邻居列表
        for other_grid in results:
            if grid["id"] == other_grid["id"]:  # 跳过自身
                continue

            # 获取两个网格的边界
            bounds1 = grid["bounds"]
            bounds2 = other_grid["bounds"]

            # 判断两个网格是否是邻居
            if (bounds1[0] <= bounds2[2] and bounds1[2] >= bounds2[0] and  # x方向重叠
                bounds1[1] <= bounds2[3] and bounds1[3] >= bounds2[1]):  # y方向重叠
                grid["neighbors"].append(other_grid["id"])  # 添加邻居的编号


# 计算邻居函数--R 树方法
def compute_neighbors_rtree(results):
    # 创建 R 树索引
    idx = index.Index()

    # 将每个网格的边界存入 R 树，ID 作为索引
    for i, grid in enumerate(results):
        x_min, y_min, x_max, y_max = grid["bounds"]
        idx.insert(i, (x_min, y_min, x_max, y_max))

    # 查找每个网格的邻居
    for i, grid in enumerate(results):
        x_min, y_min, x_max, y_max = grid["bounds"]
        grid["neighbors"] = []

        # 查询与当前网格相交的所有网格
        neighbor_ids = list(idx.intersection((x_min, y_min, x_max, y_max)))

        # 添加邻居 ID（排除自身）
        for neighbor_id in neighbor_ids:
            if neighbor_id != i:
                grid["neighbors"].append(results[neighbor_id]["id"])


# 计算网格的中心点
def calculate_grid_center(bounds):
    x_min, y_min, x_max, y_max = bounds
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return np.array([x_center, y_center])


# 计算网格的质心，如果网格内没有点，则返回网格的中心点
def calculate_grid_centroid(bounds, grid_points):
    # 如果网格内没有点，则返回网格的中心点
    if len(grid_points) == 0:
        return calculate_grid_center(bounds)  # 使用中心点

    grid_points = np.array(grid_points)
    # 否则计算质心
    x_centroid = np.mean(grid_points[:, 0])  # 所有点的x坐标的平均值
    y_centroid = np.mean(grid_points[:, 1])  # 所有点的y坐标的平均值

    return np.array([x_centroid, y_centroid])


if __name__ == "__main__":
    # 生成随机点
    points = np.random.rand(1000, 2)

    x_boundary = 1
    y_boundary = 1

    # 创建图像
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(points[:, 0], points[:, 1], s=5)  # 绘制点（加小点）

    start_time = time.time()

    # 调用四叉树划分函数并记录结果
    results = quadtree(ax, points, 0, x_boundary, 0, y_boundary, max_points=5)
    # 计算所有网格的邻居
    # compute_neighbors_rtree(results)

    total_time = time.time() - start_time

    # 打印划分结果
    print(f"总共划分出 {len(results)} 个小方格。\n")
    for result in results:
        center = calculate_grid_center(result["bounds"])
        centroid = calculate_grid_centroid(result["bounds"], result["points"])

        print(f"ID: {result['id']}, Bounds: {result['bounds']}, Depth: {result['depth']}, "
              f"Points: {len(result['points'])}个：{result['points']}, Center: {center}, Centroid: {centroid}")

    print(f"总共耗时 {total_time:.6f} 秒。")

    min_depth = min(result['depth'] for result in results)
    max_depth = max(result['depth'] for result in results)

    print(f"全局最小 depth: {min_depth}")
    print(f"全局最大 depth: {max_depth}")

    ax.set_xlim(0, x_boundary)
    ax.set_ylim(0, y_boundary)
    plt.show()

    # 测试两种方法的效率及结果是否一致
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.scatter(points[:, 0], points[:, 1], s=5)  # 绘制点（加小点）
    # results = quadtree(ax, points, 0, 1, 0, 1, max_points=5)
    # # 比较邻居查找方法
    # # 测试朴素方法
    # start_time = time.time()
    # compute_neighbors(results)
    # naive_time = time.time() - start_time
    #
    # # 保存朴素方法的邻居结果
    # naive_neighbors = {grid["id"]: set(grid["neighbors"]) for grid in results}
    #
    # # 测试 R 树方法
    # start_time = time.time()
    # compute_neighbors_rtree(results)
    # rtree_time = time.time() - start_time
    #
    # # 保存 R 树方法的邻居结果
    # rtree_neighbors = {grid["id"]: set(grid["neighbors"]) for grid in results}
    #
    # # 验证两种方法的正确性
    # correct = True
    # for grid_id in naive_neighbors:
    #     if naive_neighbors[grid_id] != rtree_neighbors[grid_id]:
    #         correct = False
    #         print(f"邻居列表不一致，网格 ID: {grid_id}")
    #         print(f"朴素方法: {naive_neighbors[grid_id]}")
    #         print(f"R 树方法: {rtree_neighbors[grid_id]}")
    #
    # # 打印结果
    # print(f"朴素方法耗时: {naive_time:.6f} 秒")
    # print(f"R 树方法耗时: {rtree_time:.6f} 秒")
    # print(f"两种方法结果是否一致: {correct}")

