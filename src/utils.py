import uuid
import hashlib

def generate_uuid():
    """
    生成一个随机的 UUID (Universally Unique Identifier)。
    
    Returns:
        str: 一个 UUID4 字符串，不包含连字符。
    """
    return str(uuid.uuid4()).replace('-', '')

def generate_md5(input_string: str) -> str:
    """
    根据输入的字符串生成其 MD5 哈希值。

    Args:
        input_string (str): 需要生成 MD5 的字符串。

    Returns:
        str: 输入字符串的 MD5 哈希值。
    """
    return hashlib.md5(input_string.encode('utf-8')).hexdigest()

import os

def get_dir_and_file_names(path):
    """
    获取文件或目录的最后一层目录名和文件名列表。

    Args:
        path (str): 本地文件或目录的路径。

    Returns:
        tuple: 包含最后一层目录名 (str) 和文件名列表 (list) 的元组。
    """
    file_names = []
    dir_name = ""

    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                file_names.append(os.path.join(root, file))
        dir_name = os.path.basename(os.path.normpath(path))
    elif os.path.isfile(path):
        file_names.append(os.path.basename(path))
        dir_name = os.path.basename(os.path.dirname(path))
    
    return dir_name, file_names

if __name__ == '__main__':
    # print(generate_uuid())
    # print(generate_md5('abc'))
    # print(generate_md5('abc'))
    # 示例用法
    # 创建一个测试目录和文件
    # os.makedirs('test_dir/subdir', exist_ok=True)
    # with open('test_dir/file1.txt', 'w') as f: f.write('test')
    # with open('test_dir/subdir/file2.txt', 'w') as f: f.write('test')
    # print(get_dir_and_file_names('test_dir'))
    # print(get_dir_and_file_names(docs_path))
    pass
