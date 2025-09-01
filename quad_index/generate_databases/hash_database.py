import AES_sha1
import time


# 处理文件并加密前两列
def hash_and_save_file(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            parts = line.strip().split(' ', 2)
            if len(parts) >= 2:
                senator, code, points = parts[0], parts[1], parts[2] if len(parts) > 2 else ''
                encrypted_code = AES_sha1.sha1_Encrypt(f"{senator} {code}")
                outfile.write(f"{encrypted_code} {points}\n")

    print(f"文件处理完成，新文件保存到: {output_path}")


def hash_and_aes_encrypt_file(input_path, output_path):
    """
    对文件进行SHA1哈希（前两列）+ AES加密（坐标部分）处理
    统计纯加密时间（不包括IO时间）
    """
    symmetric_key = 'xducc02241931xdu'

    # 第一步：读取所有文件内容到内存（IO时间不计算）
    lines = []
    with open(input_path, 'r') as infile:
        lines = [line.strip() for line in infile.readlines()]

    print(f"读取完成，共 {len(lines)} 行数据")

    # 第二步：开始计时，进行加密处理
    start_time = time.perf_counter()

    processed_data = []  # 存储处理后的数据

    for line in lines:
        if not line.strip():
            continue

        # 分割每一行：前两列 + 坐标部分
        parts = line.split(' ', 2)
        if len(parts) >= 3:
            senator = parts[0]          # 第一列：如 "Acupuncture"
            code = parts[1]             # 第二列：如 "11"
            points_str = parts[2]       # 坐标部分：如 "(26375486,72367620) (26383203,72224630)"

            # SHA1哈希前两列
            hash_input = f"{senator} {code}"
            encrypted_hash = AES_sha1.sha1_Encrypt(hash_input)

            # AES加密坐标部分
            encrypted_points = AES_sha1.AES_Encrypt(symmetric_key, points_str)

            # 存储处理后的数据
            processed_line = f"{encrypted_hash} {encrypted_points}"
            processed_data.append(processed_line)

    # 结束计时
    end_time = time.perf_counter()
    encryption_time = end_time - start_time

    print(f"纯加密时间: {encryption_time:.4f} 秒")

    # 第三步：写出结果到文件（IO时间不计算在加密时间内）
    print("正在写出结果文件...")
    with open(output_path, 'w') as outfile:
        for processed_line in processed_data:
            outfile.write(processed_line + '\n')

    print(f"文件处理完成，新文件保存到: {output_path}")


if __name__ == '__main__':
    # 文件路径
    # K_values = [80]  # 要测试的K值列表
    # datasets_name = ['twitter', 'newyork', 'paris']
    # algorithm_type = 'center'  # 'center' 或者 'centroid'
    # base_dir = '../../Cdatabase'
    #
    # for dataset_name in datasets_name:
    #     for K in K_values:
    #         input_filepath = f'{base_dir}/{dataset_name}_quadtree_{algorithm_type}{K}.txt'
    #         output_filepath = f'{base_dir}/{dataset_name}_quadtree_{algorithm_type}{K}_hash.txt'
    #         hash_and_save_file(input_filepath, output_filepath)

    base_dir = '../../Cdatabase'
    algorithm_type = 'center'  # 'center' 或者 'centroid'
    datasets_name = ['twitter', 'newyork', 'paris']
    K_values = [80]
    for dataset_name in datasets_name:
        for K in K_values:
            input_filepath = f'{base_dir}/{dataset_name}_quadtree_{algorithm_type}{K}.txt'
            output_filepath = f'{base_dir}/{dataset_name}_quadtree_{algorithm_type}{K}_encrypt.txt'
            print(f"正在加密文件: {input_filepath}")
            hash_and_aes_encrypt_file(input_filepath, output_filepath)


