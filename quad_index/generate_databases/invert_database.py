import numpy as np
import time

import InvertedIndex

if __name__ == '__main__':

    filepath = '../../original_dataset/'
    filename = filepath + 'twitter_new_RASK108.txt'

    filepath = '../../Adatabase/'
    output_filename = filepath + 'twitter_invert.txt'

    start_time = time.perf_counter()
    inverted_index = InvertedIndex.build_inverted_index_v2(filename)

    # 处理倒排索引数据，保存到结果列表中
    result_lines = []
    for word in inverted_index:
        points = np.array(inverted_index[word])
        points = np.unique(points, axis=0)
        points_str = ' '.join(f'({x} {y})' for x, y in points)
        result_lines.append(f"{word} {points_str}\n")

    invert_index_time = time.perf_counter() - start_time

    # 时间统计结束后再写入文件
    with open(output_filename, 'w') as f:
        f.writelines(result_lines)

    print('倒排索引文件已生成！')
    print(f"倒排索引构建时间: {invert_index_time:.2f} 秒")




