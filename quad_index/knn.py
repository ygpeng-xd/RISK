import numpy as np
from sklearn.neighbors import KDTree


def build_kdtree(points):
    return KDTree(points)


def find_k_nearest_neighbors(kdtree, query_point, k):
    # 使用 K-d 树查询 K 近邻
    distances, indices = kdtree.query([query_point], k)

    # 获取查询结果，直接从kdtree.data中获取点集
    k_neighbors = []
    for i in range(k):
        # 直接将 memoryview 转换为 numpy 数组
        nepoint = np.array(kdtree.data[indices[0][i]])
        k_neighbors.append((nepoint, distances[0][i]))

    return k_neighbors


if __name__ == "__main__":
    # 生成一个随机的点集
    points = np.random.rand(1000, 2)  # 生成一个随机的点集
    query_point1 = np.array([0.863, 0.768])  # 第一个查询点
    query_point2 = np.array([0.231, 0.345])  # 第二个查询点

    # 只需要构建一次KDTree
    kdtree = build_kdtree(points)

    # 进行多次查询
    neighbors1 = find_k_nearest_neighbors(kdtree, query_point1, 5)
    neighbors2 = find_k_nearest_neighbors(kdtree, query_point2, 5)

    # 输出查询结果
    print("Query point 1:", query_point1)
    print("K nearest neighbors for query point 1:")
    for neighbor, distance in neighbors1:
        print(f"Point: {neighbor}, Distance: {distance}")

    print("\nQuery point 2:", query_point2)
    print("K nearest neighbors for query point 2:")
    for neighbor, distance in neighbors2:
        print(f"Point: {neighbor}, Distance: {distance}")

