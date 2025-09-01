import numpy as np
import time
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box

import InvertedIndex
import voronoi
import self_shapely
from quad_index import knn


def quadtree(points, x_min, x_max, y_min, y_max, max_points=2, results=None, quad_id="", depth=0):
    if results is None:
        results = []

    # 如果点数小于等于 max_points，记录该小方格的信息
    if len(points) <= max_points:
        # 如果没有进行过划分（quad_id为空），设置为-1
        final_quad_id = quad_id if quad_id else "-1"
        results.append({
            "id": final_quad_id,  # 小方格唯一编号
            "bounds": [x_min, y_min, x_max, y_max],
            "points": points.tolist(),
            "depth": depth  # 记录当前划分层数
        })
        return results

    # 计算中点
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2

    # 按象限划分点
    q1 = points[(points[:, 0] >= x_mid) & (points[:, 1] >= y_mid)]  # 右上
    q2 = points[(points[:, 0] < x_mid) & (points[:, 1] >= y_mid)]  # 左上
    q3 = points[(points[:, 0] < x_mid) & (points[:, 1] < y_mid)]  # 左下
    q4 = points[(points[:, 0] >= x_mid) & (points[:, 1] < y_mid)]  # 右下

    # 递归处理每个象限并添加编号，同时增加层数
    quadtree(q1, x_mid, x_max, y_mid, y_max, max_points, results, quad_id + "11", depth + 1)
    quadtree(q2, x_min, x_mid, y_mid, y_max, max_points, results, quad_id + "10", depth + 1)
    quadtree(q3, x_min, x_mid, y_min, y_mid, max_points, results, quad_id + "00", depth + 1)
    quadtree(q4, x_mid, x_max, y_min, y_mid, max_points, results, quad_id + "01", depth + 1)

    return results


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


def center_knn(kd_tree, cell, K):
    center = calculate_grid_center(cell["bounds"])
    # 获取KD树中的点数
    available_points = kd_tree.data.shape[0]
    # 如果可用点数少于K，则使用所有可用点
    actual_k = min(K, available_points)
    k_nearest = knn.find_k_nearest_neighbors(kd_tree, center, actual_k)
    # 将近邻点转换为整型，并返回列表
    nearest_points = [(int(neighbor[0]), int(neighbor[1])) for neighbor, _ in k_nearest]
    return nearest_points


def centroid_knn(kd_tree, cell, K):
    centroid = calculate_grid_centroid(cell["bounds"], cell["points"])
    # 获取KD树中的点数
    available_points = kd_tree.data.shape[0]
    # 如果可用点数少于K，则使用所有可用点
    actual_k = min(K, available_points)
    k_nearest = knn.find_k_nearest_neighbors(kd_tree, centroid, actual_k)
    nearest_points = [(int(neighbor[0]), int(neighbor[1])) for neighbor, _ in k_nearest]
    return nearest_points


def voronoi_creat(points, x_boundary, y_boundary):
    my_vor = Voronoi(points)
    point_lines, point_lines_count = voronoi.points_with_rigdes(my_vor)
    problem_point_index2 = voronoi.update_virtual_end_point_v2(my_vor, point_lines)
    point_lines2 = voronoi.update_virtual_end_point_v12(x_boundary, y_boundary, my_vor, point_lines,
                                                        problem_point_index2)
    point_lines3 = voronoi.update_out_boundary(x_boundary, y_boundary, point_lines2)
    final_point_lines = voronoi.final_update(x_boundary, y_boundary, point_lines3)
    final_point_vertices = voronoi.points_with_vertices(x_boundary, y_boundary, my_vor, final_point_lines)
    return final_point_vertices


def voronoi_add_points(points, final_point_vertices, cell):
    grid = cell["bounds"]
    grid_box = box(grid[0], grid[1], grid[2], grid[3])
    added_points = []
    for point_index in final_point_vertices:
        coords = np.array(final_point_vertices[point_index])
        # 判断该点对应的 voronoi 多边形是否和当前网格有交集
        if min(coords[:, 0]) > grid[2] or max(coords[:, 0]) < grid[0] or \
           min(coords[:, 1]) > grid[3] or max(coords[:, 1]) < grid[1]:
            continue
        new_coords = self_shapely.get_new_coords(coords)
        voronoi_poly = Polygon(new_coords)
        if self_shapely.is_intersect(grid_box, voronoi_poly):
            added_points.append((int(points[point_index][0]), int(points[point_index][1])))
    return added_points


if __name__ == '__main__':
    filepath = '../../original_dataset/'
    filename = None

    x_boundary = 100000000
    y_boundary = 100000000

    datasets_name = ['twitter', 'newyork', 'paris']
    K_values = [80]
    algorithm_type = 'center'  # 'center' 或者 'centroid'
    point_knn_count = 10
    base_dir = '../../Cdatabase'

    for dataset_name in datasets_name:
        if dataset_name == 'twitter':
            filename = filepath + 'twitter_new_RASK108.txt'
        elif dataset_name == 'newyork':
            filename = filepath + 'newyork_RASK108.txt'
        elif dataset_name == 'paris':
            filename = filepath + 'Paris_RASK108.txt'
        inverted_index = InvertedIndex.build_inverted_index_v2(filename)

        for K in K_values:
            output_filename = f'{base_dir}/{dataset_name}_quadtree_{algorithm_type}{K}.txt'
            # output_filename = f'{base_dir}/{dataset_name}_quadtree{K}.txt'
            # output_filename = base_dir + 'newyork_quadtree_voronoi1.txt'

            start_time = time.perf_counter()

            # 保存所有结果的列表
            result_lines = []

            for word in inverted_index:
                points = np.array(inverted_index[word])
                # 使用numpy.unique()函数进行去重操作
                points = np.unique(points, axis=0)

                # 构建kd树
                kd_tree = knn.build_kdtree(points)

                # 构建Voronoi图
                # voronoi_vertices = voronoi_creat(points, x_boundary, y_boundary)

                # 进行四叉树划分
                results = quadtree(points, 0, x_boundary, 0, y_boundary, max_points=K)

                # 处理结果并保存到列表中
                for cell in results:
                    result_points = center_knn(kd_tree, cell, point_knn_count)
                    # result_points = centroid_knn(kd_tree, cell, K)
                    # result_points = voronoi_add_points(points, voronoi_vertices, cell)

                    cell_points = [(int(x), int(y)) for x, y in cell['points']]

                    # 合并两个列表，保持顺序且不重复
                    seen = set()
                    combined_points = []
                    for pt in cell_points + result_points:
                        if pt not in seen:
                            seen.add(pt)
                            combined_points.append(pt)

                    # 格式化输出并添加到结果列表
                    points_str = ' '.join(f"({x},{y})" for x, y in combined_points)
                    result_lines.append(f"{word} {cell['id']} {points_str}\n")
                    # print(f"{word} {cell['id']} {points_str}")

                    # 输出纯净quadtree索引
                    # if len(cell['points']) == 0:
                    #     result_lines.append(f"{word} {cell['id']}\n")
                    # else:
                    #     cell_points = ' '.join(f"({int(x)},{int(y)})" for x, y in cell['points'])
                    #     result_lines.append(f"{word} {cell['id']} {cell_points}\n")

            gen_index_time = time.perf_counter() - start_time

            # 时间统计结束后再写入文件
            with open(output_filename, 'w') as f_out:
                f_out.writelines(result_lines)

            print(f"{dataset_name}_quadtree_{algorithm_type}{K}.txt 非密索引构建完成")
            print(f"非密索引构建时间: {gen_index_time:.4f} 秒")


