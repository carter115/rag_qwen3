import os

from .dashscope_processor import DashscopeProcessor
from .document_parser import ParsedRecordManager, MineruParser
from .utils import get_dir_and_file_names
from .vector_processor import VectorProcessor


class Pipeline:
    def __init__(self, mineru_api_key='', parsed_output_dir='', record_filename='parsed_records.json') -> None:
        """
        主业务逻辑
        Args:
            record_manager (ParsedRecordManager, optional): 解析记录管理器实例。如果提供，将使用它来管理记录。
        """
        self.parsed_output_dir = parsed_output_dir
        self.record_manager = ParsedRecordManager(output_dir=self.parsed_output_dir, record_filename=record_filename)
        self.mineru_parser = MineruParser(mineru_api_key)

        dashscope_api_key = "sk-xxx"  # 建议将API Key从代码中移除，使用环境变量或配置文件
        # self.vector = VectorProcessor(dashscope_api_key=dashscope_api_key, record_manager=self.record_manager)
        # self.llm = DashscopeProcessor(dashscope_api_key=dashscope_api_key)
        print("pipeline初始化成功......")


    def parse_documents(self, path):
        """
        处理文档，上传到Mineru并获取解析结果。
        Args:
            path (str): 待处理文件或目录的路径。
            api_key (str): Mineru API的授权Token。
        Returns:
            list: 包含已保存的Markdown文件路径的列表。
        """
        dir_name, files = get_dir_and_file_names(path)
        print(f"待处理目录: {dir_name}, 文件[{len(files)}]: {files}")
        if not files:
            print("No files found to process.")
            return []

        files_to_process = []
        for file_path in files:
            if self.record_manager and self.record_manager.has_record(os.path.basename(file_path)):
                print(f"Skipping {file_path} as it's already parsed.")
            else:
                files_to_process.append(file_path)

        if not files_to_process:
            print("All files already parsed or no new files to process.")
            return self.record_manager.records if self.record_manager else []

        print(f"正在上传文件[{len(files_to_process)}]：{files_to_process}")
        batch_id, _ = self.mineru_parser.upload_files_batch(files_to_process)
        if batch_id:
            # bsname = os.path.basename(file_path)
            # # 提取文件名作为前缀，如果处理的是目录，则加上目录名
            # processed_filename = f"{dir_name}_{generate_md5(bsname)}.json"
            file_results = self.mineru_parser.get_extract_results_batch(batch_id, output_dir=self.parsed_output_dir)
            print(f"get_extract_results_batch: {file_results}")
            if file_results:
                for src_file, json_file in file_results:
                    record = {
                        "filename": json_file,
                        "original_filename": src_file,
                        "collection": dir_name
                    }
                    self.record_manager.add_record(record)
            self.record_manager.save_records()
            return self.record_manager.records if self.record_manager else []
        return self.record_manager.records if self.record_manager else []

    def save_embeding(self):
        return self.vector.save_document_to_milvus()

    def search(self, coll, query, count):
        return self.vector.search_and_rerank(coll, query, top_k=count)


output_dir = './parsed_documents'
api_key = "eyJ0eXBlIjoiSldUIixxxxxxxxxx"
singleton_pipeline = Pipeline(mineru_api_key=api_key, parsed_output_dir=output_dir,
                              record_filename="parsed_records.json")
# print(pipeline.parse_documents(docs_path))
# print(pipeline.read_doc('zhongxin_f4b2f9b4039a69b830a274ea0c3024b9.json'))
