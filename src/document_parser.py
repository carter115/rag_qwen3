import io
import json
import os
import time
import zipfile

import requests


class ParsedRecordManager:
    """
    管理解析文档记录的类，包括加载、查询和追加功能。
    """

    def __init__(self, output_dir, record_filename):
        """
        初始化ParsedRecordManager。
        Args:
            output_dir (str): 存储解析记录的目录。
            record_filename (str): 存储解析记录的文件名。
        """
        self.output_dir = output_dir
        self.record_file_path = os.path.join(output_dir, record_filename)
        self.records = self._load_records()

    def _load_records(self):
        """
        从JSON文件加载解析记录。
        Returns:
            list: 解析记录列表。
        """
        if os.path.exists(self.record_file_path):
            try:
                with open(self.record_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {self.record_file_path}. Starting with empty records.")
                return []
        return []

    def read_document(self, json_file: str):
        """
        读取指定的 JSON 文件内容。

        Args:
            json_file (str): JSON 文件的路径。

        Returns:
            list: JSON 文件的内容，如果文件不存在或解析失败则返回空字典。
        """
        filepath = os.path.join(self.output_dir, json_file)
        if not os.path.exists(filepath):
            print(f"Error: JSON file not found at {filepath}")
            return {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {filepath}: {e}")
            return {}
        except Exception as e:
            print(f"An unexpected error occurred while reading {filepath}: {e}")
            return {}

    def save_records(self):
        """
        将当前解析记录保存到JSON文件。
        """
        os.makedirs(os.path.dirname(self.record_file_path), exist_ok=True)
        with open(self.record_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=4, ensure_ascii=False)
        print(f"Saved parsed records to {self.record_file_path}")

    def add_record(self, record):
        """
        向记录中追加新的解析结果。
        Args:
            record (dict): 要追加的解析记录，包含'filename'和'original_filename'。
        """
        if self.find_record_idx(record) != 0:
            self.records.append(record)

    def has_record(self, pdf_file: str) -> bool:
        """
        检查给定文件名是否存在于已解析的记录中。
        :param pdf_file: 要检查的文件名。
        :return: 如果文件存在于记录中则返回 True，否则返回 False。
        """
        return any(record.get('original_filename') == pdf_file for record in self.records)

    def find_record_idx(self, record: dict):
        """
        根据原始文件名查找记录。
        :param record: 要查找的记录，包含'filename'键。
        :return: 如果找到记录则返回记录的索引
        """
        for idx, r in enumerate(self.records):
            if r.get('filename') == record.get('filename'):
                return idx
        return None

    def record_status_is_embed(self, record: dict):
        """
        检查给定记录的状态是否为已嵌入。
        :param record: 要检查的记录，应该包含'filename'和'status'键。
        :return: 如果记录的状态为'已嵌入'则返回 True，否则返回 False。
        """
        idx = self.find_record_idx(record)
        return self.records[idx].get('status') == 'embed'

    def record_update_status_embed(self, record: dict):
        """
        更新给定记录的状态为已嵌入。
        :param record: 要更新的记录，应该包含'filename'键。
        """
        idx = self.find_record_idx(record)
        self.records[idx]['status'] = 'embed'
        self.save_records()


class MineruParser:
    """
    Mineru文件解析器，用于通过Mineru API上传文件并获取解析结果。
    """

    def __init__(self, api_key):
        """
        初始化MineruParser。
        Args:
            api_key (str): Mineru API的授权Token。
        """
        self.api_key = api_key
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

    def upload_files_batch(self, file_paths, enable_formula=True, language="ch", enable_table=True):
        """
        批量上传文件到Mineru并获取上传URL。
        Args:
            file_paths (list): 待上传文件的路径列表。
            enable_formula (bool): 是否启用公式识别。
            language (str): 解析语言（如"en", "ch"）。
            enable_table (bool): 是否启用表格识别。
        Returns:
            tuple: (batch_id, file_urls) 如果成功，否则返回(None, None)。
        """
        url = 'https://mineru.net/api/v4/file-urls/batch'
        files_data = []
        for fp in file_paths:
            file_name = os.path.basename(fp)
            files_data.append({"name": file_name, "is_ocr": True, "data_id": file_name})  # data_id 可以是任意唯一标识

        data = {
            "enable_formula": enable_formula,
            "language": language,
            "enable_table": enable_table,
            "files": files_data
        }

        try:
            response = requests.post(url, headers=self.headers, json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"Mineru API:{url} Response: {result}")  # 打印API结果
                if result["code"] == 0:
                    batch_id = result["data"]["batch_id"]
                    urls = result["data"]["file_urls"]
                    print(f'Mineru: Batch ID: {batch_id}, Upload URLs obtained.')
                    # 上传文件到Mineru提供的URL
                    for i, upload_url in enumerate(urls):
                        with open(file_paths[i], 'rb') as f:
                            res_upload = requests.put(upload_url, data=f)
                            if res_upload.status_code == 200:
                                print(f"Mineru: {os.path.basename(file_paths[i])} uploaded successfully.")
                            else:
                                print(
                                    f"Mineru: {os.path.basename(file_paths[i])} upload failed: {res_upload.status_code} {res_upload.text}")
                    return batch_id, urls
                else:
                    print(f'Mineru: Failed to get upload URLs: {result.get("msg", "Unknown error")}')
            else:
                print(f'Mineru: API response not successful. Status: {response.status_code}, Result: {response.text}')
        except Exception as err:
            print(f'Mineru: Error during file upload batch: {err}')
        return None, None

    def get_extract_results_batch(self, batch_id, timeout=300, interval=5, output_dir=""):
        """
        批量获取Mineru文件解析结果。
        Args:
            batch_id (str): 批量任务ID。
            timeout (int): 等待结果的超时时间（秒）。
            interval (int): 轮询间隔时间（秒）。
            output_dir (str): 解析结果保存的目录。
            file_name (str): 文件名，用于构建输出文件名。
        Returns:
            list: 包含已保存的Json文件路径的列表。
        """
        url = f'https://mineru.net/api/v4/extract-results/batch/{batch_id}'
        start_time = time.time()
        last_response = None
        fina_results = []
        while time.time() - start_time < timeout:
            try:
                res = requests.get(url, headers=self.headers)
                last_response = res  # 保存最新的响应
                if res.status_code == 200:
                    result = res.json()
                    print(f"Mineru API({url}) Response: {result}")  # 打印API结果
                    if result["code"] == 0:
                        extract_results = result["data"].get("extract_result", [])
                        print(f"Mineru extract_results: {extract_results}")
                        # 检查是否有文件处于running状态
                        if any(item.get("state") != "done" for item in extract_results):
                            print(f"Mineru: Batch {batch_id} still processing... Waiting {interval} seconds.")
                        else:
                            # 所有文件都已处理完毕（done或failed）
                            print(f"Mineru: Batch {batch_id} results obtained successfully.")
                            for item in extract_results:
                                if item.get("state") == "done" and item.get("full_zip_url"):
                                    full_zip_url = item["full_zip_url"]
                                    jsondata_file = self._process_zip_file(full_zip_url, output_dir=output_dir)
                                    if jsondata_file:
                                        fina_results.append((item['file_name'], jsondata_file))
                            return fina_results
                    else:
                        print(f'Mineru: Failed to get extract results: {result.get("msg", "Unknown error")}')
                else:
                    print(f'Mineru: API response not successful. Status: {res.status_code}, Result: {res.text}')
            except Exception as err:
                print(f'Mineru: Error during getting extract results: {err}')
            time.sleep(interval)
        print(f"Mineru: Timeout waiting for batch {batch_id} results.")
        return fina_results

    def _process_zip_file(self, full_zip_url, output_dir):
        """
        处理从Mineru API下载的ZIP文件，解压并保存Markdown内容。
        Args:
            full_zip_url (str): ZIP文件的下载URL。
        Returns:
            bool: 如果成功处理并保存了Markdown文件，则返回True，否则返回False。
        """
        try:
            zip_response = requests.get(full_zip_url)
            zip_response.raise_for_status()  # 检查HTTP请求是否成功
            print(f"Download ZIP Response for {full_zip_url}: {zip_response.status_code}")  # 打印下载响应状态

            # # 保存ZIP文件到本地
            # zip_file_path = os.path.join(output_dir, os.path.basename(full_zip_url))
            # with open(zip_file_path, 'wb') as f:
            #     f.write(zip_response.content)
            # print(f"ZIP file saved to: {zip_file_path}")
            # 使用BytesIO处理ZIP文件内容
            with zipfile.ZipFile(io.BytesIO(zip_response.content), 'r') as zf:
                for zf_info in zf.infolist():
                    # 查找JSON文件
                    if zf_info.filename != "layout.json" and zf_info.filename.endswith('.json'):
                        with zf.open(zf_info.filename) as json_file:
                            json_content = json_file.read().decode('utf-8')

                            # 构造输出文件路径
                            # output_filename 已经是原始文件的basename，例如 'document.pdf'
                            # 我们需要将其扩展名改为 .json
                            base_name_without_ext = os.path.splitext(os.path.basename(full_zip_url))[0] + '.json'
                            output_json_path = os.path.join(output_dir, base_name_without_ext)

                            # 确保输出目录存在
                            os.makedirs(output_dir, exist_ok=True)

                            # 保存JSON内容到本地文件
                            with open(output_json_path, "w", encoding="utf-8") as f:
                                f.write(json_content)
                            print(f"Saved extracted JSON to {base_name_without_ext}")
                            return base_name_without_ext
            return None
        except requests.exceptions.RequestException as e:
            print(f"Failed to download zip from {full_zip_url}: {e}")
            return None
        except zipfile.BadZipFile:
            print(f"Downloaded file is not a valid zip file from {full_zip_url}")
            return None
        except Exception as e:
            print(f"An error occurred during zip processing: {e}")
            return None
