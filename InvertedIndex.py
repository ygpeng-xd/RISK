from datetime import datetime
import numpy as np


# 构建倒排索引，返回倒排索引（行号和坐标）和词频
def build_inverted_index(filename):
    inverted_index = {}
    word_count = {}
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            words = line.strip().split(' ')
            # strip()方法可以去掉字符串开头和结尾的空格和换行符等字符，split(' ')
            # 方法可以以空格为分隔符将字符串切分成多个子串，并将它们保存到列表中
            coord = (int(words[0]), int(words[1]))
            for word in words[2:]:
                if word not in inverted_index:
                    inverted_index[word] = []
                    # 首先判断word是否已经在倒排索引inverted_index中。
                    # 如果没有出现过，我们就创建一个空列表并将其加入到inverted_index
                    word_count[word] = 0

                inverted_index[word].append((i+1, np.array(coord)))
                # 将当前行号i和坐标元组coord组成一个元组(i, coord)，
                # 并将其添加到word对应的列表中
                word_count[word] += 1
    return inverted_index, word_count


# 第二版本，构建倒排索引，返回倒排索引（坐标） + 原始记录
def build_inverted_index_v2(filename, flag=True):
    inverted_index = {}
    line_dict = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            words = line.strip().split(' ')
            coord = (int(words[0]), int(words[1]))
            if not flag:
                line_dict[coord] = line.strip()
            for word in words[2:]:
                if word not in inverted_index:
                    inverted_index[word] = []
                inverted_index[word].append(coord)
    if flag:
        return inverted_index
    else:
        return inverted_index, line_dict


# 查询指定单词的行号和坐标信息
def search_word_line(word, inverted_index):
    if word not in inverted_index:
        print("单词不存在")
        return
    locations = inverted_index[word]
    # 从倒排索引inverted_index中获取word对应的行号和坐标信息列表locations。
    # 该列表的每个元素是一个二元组，第一个元素是行号，第二个元素是坐标

    seen = set()
    # 为了防止重复输出同一行的信息，需要使用一个集合 seen 来记录已经输出过的行号

    for loc in locations:
        if loc[0] not in seen:
            print(f"行号：{loc[0]}，坐标：{loc[1]}")
            seen.add(loc[0])
    # 在遍历 locations 的过程中，对于每个二元组 loc，先判断它的行号 loc[0] 是否已经在集合 seen 中出现过。
    # 如果已经出现过，说明这个行号的信息已经被输出过了，就直接跳过。
    # 如果是第一次出现，就将行号加入到集合 seen 中，并输出该行的信息，包括行号和坐标


# 查询指定单词的出现次数
def search_word_count(word, word_count):
    if word not in word_count:
        return
    else:
        count = word_count[word]
        print(f"{word}出现次数: {count}")


# 输出单词的行号和坐标信息
def output_inverted_index(inverted_index):
    for word in inverted_index:
        print("keyword: " + word)
        locations = inverted_index[word]
        seen = set()
        # 防止重复输出
        for loc in locations:
            if loc[0] not in seen:
                print(f"行号：{loc[0]}，坐标：{loc[1]}")
                seen.add(loc[0])


if __name__ == "__main__":
    filepath = './original_dataset/'
    filename = filepath + 'twitter_new_RASK108.txt'
    time_start = datetime.now()
    inverted_index, word_count = build_inverted_index(filename)

    # 统计数据集的行数
    with open(filename, 'r') as f:
        total_lines = sum(1 for line in f)
    print(f'数据集总行数：{total_lines}')

    # 统计每个keywords对应点的数量的平均值
    if word_count:
        total_points = sum(word_count.values())
        avg_points_per_keyword = total_points / len(word_count)
        print(f'每个keywords对应点的数量的平均值：{avg_points_per_keyword:.2f}')

    Max = max(word_count.values())
    Min = min(word_count.values())
    print(Max, Min)

    time_end = datetime.now()
    print(f"本次构建倒排索引用时：{str(time_end - time_start)}秒 ")
    # output_inverted_index(inverted_index)

    print('不同keywords数：', len(word_count))
    # 测试查询单词
    while True:
        query = input("请输入要查询的单词：")
        search_word_line(query, inverted_index)
        search_word_count(query, word_count)


