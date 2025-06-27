import os
import time
from http import HTTPStatus
from typing import List, Dict, Union

import dashscope
import torch
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from transformers import AutoTokenizer, AutoModelForCausalLM

from .document_parser import ParsedRecordManager


class VectorProcessor:
    """
    向量处理实现类，包含Embedding、Reranker模型初始化、文档保存到Milvus以及从Milvus进行召回和重排序的功能。
    """

    def __init__(self, milvus_host="localhost", milvus_port="19530",
                 dashscope_api_key="",
                 rerank_model_path='../models/Qwen/Qwen3-Reranker-0___6B',
                 record_manager: ParsedRecordManager = None):
        """
        初始化Embedding和Reranker模型，并连接Milvus。
        Args:
            milvus_host (str): Milvus服务的地址。
            milvus_port (str): Milvus服务的端口。
            mineru_api_key (str, optional): Mineru API 密钥，默认为从环境变量 `MINERU_API_KEY` 获取。
        """
        self.milvus_client = MilvusClient(host=milvus_host, port=milvus_port)
        # self.milvus_client = MilvusClient(uri="./milvus_demo.db")
        self.record_manager = record_manager

        # 初始化Qwen3-Embedding模型
        # self.embedding_model = SentenceTransformer("../../models/Qwen/Qwen3-Embedding-0___6B")
        dashscope.api_key = dashscope_api_key

        # 初始化Qwen3-Reranker模型
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=rerank_model_path,
                                                                padding_side='left')
        self.reranker_model = AutoModelForCausalLM.from_pretrained(rerank_model_path).eval()

        # Reranker配置
        self.token_false_id = self.reranker_tokenizer.encode("no", add_special_tokens=False)[0]
        self.token_true_id = self.reranker_tokenizer.encode("yes", add_special_tokens=False)[0]
        self.max_reranker_length = 8192
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.reranker_tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.reranker_tokenizer.encode(self.suffix, add_special_tokens=False)

        # # 清理（可选）
        # self.milvus_client.drop_collection(collection_name='zhongxin')
        # print(f"Collection 'zhongxin' dropped.")
        # time.sleep(1)

        print(f"向量数据库启动成功")

    def _create_collection(self, collection_name):
        """
        创建Milvus集合，如果集合不存在的话。
        Args:
            collection_name (str): 要创建或加载的Milvus集合的名称。
        """
        if self.milvus_client.has_collection(collection_name=collection_name):
            print(f"Collection '{collection_name}' already exists.")
            # 如果集合已存在，也需要加载到内存
            self.milvus_client.load_collection(collection_name=collection_name)
            print(f"Collection '{collection_name}' loaded into memory.")
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),  # Qwen3-Embedding-0.6B的维度是1024
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256, description="原始文件名"),
            FieldSchema(name="page_number", dtype=DataType.VARCHAR, max_length=128, description="文本所在页码")
        ]
        schema = CollectionSchema(fields, description=f"{collection_name} RAG Collection")
        self.milvus_client.create_collection(collection_name=collection_name, schema=schema)
        print(f"Collection '{collection_name}' created successfully.")

        # 创建索引
        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="FLAT", metric_type="L2")
        self.milvus_client.create_index(collection_name=collection_name, index_params=index_params)
        print(f"Index created for collection '{collection_name}'.")

        # 加载集合到内存
        self.milvus_client.load_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' loaded into memory.")

    def emb_text(self, text, is_query=False):
        """
        使用Qwen3-Embedding模型生成文本嵌入。
        Args:
            text (str): 输入文本。
            is_query (bool): 是否为查询文本。
        Returns:
            list: 文本的嵌入向量。
        """
        resp = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input=text,
            dimension=1024,  # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
            output_type="dense&sparse"
        )
        if resp.status_code == HTTPStatus.OK:
            # 确保返回的是一个扁平的浮点数列表
            # embeddings = resp.output['embeddings']
            # if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], list):
            #     return embeddings
            return resp.output['embeddings'][0]['embedding']

    def save_chunks(self, chunks, file_name, collection_name):
        """
        将文本块保存到Milvus中。
        Args:
            chunks (list): 文本块列表。
            file_name (str): 原始文件名。
            page_number (list): 文本块所在页码。
            collection_name (str): 要使用的Milvus集合名称。
        Returns:
            list: 插入文档的ID列表。
        """
        entities = []
        for chunk_item in chunks:
            chunk_text = chunk_item['text']
            if chunk_text:
                page_number = chunk_item['page_number']
                embedding = self.emb_text(chunk_text)
                # 确保embedding是一个扁平的浮点数列表，如果emb_text返回的是列表的列表，则取第一个元素
                # if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
                #     embedding = embedding[0]
                entities.append({"embedding": embedding, "text": chunk_text, "file_name": file_name,
                                 "page_number": ','.join(map(str, page_number))})

        if entities:
            res = self.milvus_client.insert(collection_name=collection_name, data=entities)
            print(f"Inserted {len(res['ids'])} documents into Milvus collection {collection_name}.")
            return res['ids']
        return []

    def save_document_to_milvus(self, chunk_size: int = 500, chunk_overlap_percent: float = 0.2) -> list:
        """
        将解析后的文档内容分段后保存到Milvus。

        Args:
            chunk_size (int): 每个文本块的最大长度。
            chunk_overlap_percent (float): 文本块之间的重叠百分比（0.0到1.0之间）。
            separators (list): 用于分段的自定义分隔符列表。

        Returns:
            list: 包含每个文件处理结果的字典列表，每个字典包含文件名和插入的ID列表。
        """
        results = []
        # 计算实际重叠大小
        chunk_overlap = int(chunk_size * chunk_overlap_percent)

        for record in self.record_manager.records:
            ori_filename = os.path.basename(record['original_filename'])
            parsed_filename = record['filename']
            coll_name = record['collection']
            if self.record_manager.record_status_is_embed(record):  # 忽略已经embedding 的文档
                print(f"文件 {parsed_filename} 已经向量化，已忽略")
                continue

            # 确保集合存在并加载
            self._create_collection(coll_name)

            result = dict()
            result['name'] = ori_filename
            result['inserted_ids'] = []

            print(f"数据集: {coll_name}, 文件: {parsed_filename}, 原文件: {ori_filename} 正在向量化......")
            # 读取解析后的JSON文件内容
            json_content_list = self.record_manager.read_document(parsed_filename)
            if not json_content_list:
                print(f"读取内容为空")
                continue

            # 提取并合并文本内容和页码
            content_with_pages = self._extract_and_content(json_content_list)
            chunks_with_pages = self._merge_chunks(content_with_pages=content_with_pages, chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)

            inserted_ids = self.save_chunks(chunks_with_pages, ori_filename, coll_name)
            result['inserted_ids'].extend(inserted_ids)

            result['size'] = len(result['inserted_ids'])
            print(f"数据集: {coll_name}, 文件 '{parsed_filename}' 处理完成，共插入 {result['size']} 个文本块。")
            self.record_manager.record_update_status_embed(record)
            results.append(result)
        return results

    def _extract_and_content(self, json_content_list: list) -> List[Dict[str, Union[str, int]]]:
        """
        从解析后的 JSON 内容列表中提取文本和表格数据，并合并成一个包含文本和页码的列表。
        忽略类型为 'image' 的内容。

        Args:
            json_content_list (list): 包含解析后文档内容的字典列表。

        Returns:
            List[Dict[str, Union[str, int]]]: 包含合并后的文本内容和对应页码的字典列表。
                                              每个字典包含 'text' 和 'page_number' 键。
        """
        combined_content_with_pages = []
        for item in json_content_list:
            item_type = item.get('type')
            page_number = item.get('page_idx') + 1  # 假设每个item都有page_idx
            if item_type == 'text':
                text_content = item.get('text')
                if text_content:
                    combined_content_with_pages.append({"text": text_content, "page_number": page_number})
            elif item_type == 'table':
                table_body = item.get('table_body')
                if table_body:
                    combined_content_with_pages.append({"text": table_body, "page_number": page_number})
            # type=image 忽略
        return combined_content_with_pages

    def _merge_chunks(self, content_with_pages: List[Dict[str, Union[str, int]]], chunk_size: int,
                      chunk_overlap: float) -> List[Dict[str, any]]:
        """
        根据指定大小和重叠率分段文本，并保留页码信息。

        Args:
            content_with_pages (List[Dict[str, Union[str, int]]]): 包含文本内容和页码的字典列表。
            chunk_size (int): 每个文本块的最大长度。
            chunk_overlap (float): 文本块之间的重叠比例（0~1）。

        Returns:
            List[Dict[str, Union[str, int]]]: 分段后的文本块列表，每个字典包含 'text' 和 'page_number' 键。

        实现逻辑：
        1. 第一次合并：从头开始，将文本内容逐个添加到 `current_chunk_text` 中，直到其长度超过 `chunk_size`。
           将此 `current_chunk_text` 及其对应的页码保存为一个文本块。
        2. 后续合并：对于剩余的每个文本项，将其作为新块的起始。然后，从该文本项之前的内容中，向前查找并添加重叠内容。
           重叠内容的长度由 `chunk_overlap_size` 决定。将重叠内容与当前文本项合并，形成新的文本块。
           这样确保了每个块都包含前一个块的重叠部分，以便在后续处理中保持上下文。
        
        """

        all_chunks_with_pages = []
        current_chunk_text = ""
        current_chunk_pages = set()
        last_chunk_end_idx = -1

        for i, item in enumerate(content_with_pages):
            text_to_add = item['text']
            page_number_to_add = item['page_number']

            # 如果当前块加上新文本不超过chunk_size，则继续添加
            if len(current_chunk_text) + len(text_to_add) <= chunk_size:
                current_chunk_text += text_to_add
                current_chunk_pages.add(page_number_to_add)
            else:
                # 当前块已满，保存当前块
                if current_chunk_text:
                    all_chunks_with_pages.append({
                        'text': current_chunk_text,
                        'page_number': sorted(list(current_chunk_pages))
                    })

                # 重置当前块，并处理重叠部分
                current_chunk_text = ""
                current_chunk_pages = set()

                # 从上一个块的末尾开始，向前寻找重叠内容
                overlap_start_idx = i - 1
                temp_overlap_text = ""
                temp_overlap_pages = set()

                while overlap_start_idx >= 0 and len(temp_overlap_text) < chunk_overlap:
                    prev_item = content_with_pages[overlap_start_idx]
                    temp_overlap_text = prev_item['text'] + temp_overlap_text
                    temp_overlap_pages.add(prev_item['page_number'])
                    overlap_start_idx -= 1

                # 将重叠内容添加到新块的开头
                current_chunk_text = temp_overlap_text + text_to_add
                current_chunk_pages.update(temp_overlap_pages)
                current_chunk_pages.add(page_number_to_add)

        # 添加最后一个块（如果存在）
        if current_chunk_text:
            all_chunks_with_pages.append({
                'text': current_chunk_text,
                'page_number': sorted(list(current_chunk_pages))
            })

        return all_chunks_with_pages

    def format_retrieval_results(self, query, retrieval_results):
        """
        将检查结果格式化。

        Args:
            query (str): 用户提问
            retrieval_results (list): 字符串列表。
        Returns:
            str: 格式化后的输入字符串。
        """
        user_question = f"用户的问题：{query}"
        context_parts = [user_question]
        for result in retrieval_results:
            page_number = result['page_number']
            text = result['text']
            context_parts.append(f'Text retrieved from page {page_number}: \n"""\n{text}\n"""')

        document = "\n\n---\n\n".join(context_parts)
        return document

    def _format_rerank_instruction(self, query, document, instruction=None):
        """
        将查询、文档和任务指令格式化为重排序模型的标准输入格式。
        Args:
            query (str): 查询字符串。
            document (str): 文档字符串。
            instruction (str): 任务指令。
        Returns:
            str: 格式化后的输入字符串。
        """
        if instruction:
            return f"{self.prefix}Query: {query}\nDocument: {document}\nInstruct: {instruction}{self.suffix}"
        else:
            return f"{self.prefix}Query: {query}\nDocument: {document}{self.suffix}"

    def rerank(self, query, documents, top_k=5, batch_size=32):
        """
        优化后的重排序方法，支持分批次处理大量文档。

        Args:
            query (str): 查询字符串。
            documents (list): 待排序文档列表。
            top_k (int): 返回top结果数量。
            batch_size (int): 每批次处理文档数。

        Returns:
            list: (score, document)元组列表，按分数降序排列。
        """
        if not documents:
            return []

        # 分批次处理文档
        all_scores = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]

            # 生成格式化输入
            formatted_inputs = [
                self._format_rerank_instruction(query, doc)
                for doc in batch_docs
            ]

            # 编码处理
            inputs = self.reranker_tokenizer(
                formatted_inputs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_reranker_length
            ).to(self.reranker_model.device)

            with torch.no_grad():
                outputs = self.reranker_model(**inputs)
                batch_logits = outputs.logits[:, -1, :]  # 获取最后一个token的logits

                # 准确获取yes/no的logits
                yes_logits = batch_logits[:, self.token_true_id]
                no_logits = batch_logits[:, self.token_false_id]

                # 计算相关性分数
                batch_scores = torch.nn.functional.log_softmax(
                    torch.stack([no_logits, yes_logits], dim=1), dim=1
                )[:, 1].exp().tolist()

            all_scores.extend(batch_scores)

        # 组合文档与分数并按分数排序
        scored_docs = sorted(
            zip(all_scores, documents),
            key=lambda x: x[0],
            reverse=True
        )

        return scored_docs[:top_k]

    def search_and_rerank(self, collection_name, query, limit=10, top_k=3):
        """
        在Milvus中搜索相关文档，然后使用Reranker模型对结果进行重排序。
        Args:
            query (str): 用户查询。
            limit (int): 从Milvus检索的文档数量。
            top_k (int): 重排序后返回的顶部K个文档数量。
            collection_name (str, optional): 要搜索的Milvus集合名称。如果为None，则使用实例的默认集合名称。
        Returns:
            list: (score, document)元组列表，按分数降序排列。
        """
        t0 = time.time()
        # 1. 在Milvus中搜索
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        query_embedding = self.emb_text(query)
        res = self.milvus_client.search(
            collection_name=collection_name,
            data=[query_embedding],
            limit=limit,
            search_params=search_params,
            output_fields=["text", "file_name", "page_number"]
        )

        # 提取搜索到的文档及其元数据
        retrieved_docs = []
        for hit in res[0]:
            retrieved_docs.append({
                'text': hit.entity.get('text'),
                'file_name': hit.entity.get('file_name'),
                'page_number': hit.entity.get('page_number')
            })
        t1 = time.time()
        print(f"检索到的文档: {len(retrieved_docs)}个，耗时: {t1 - t0:.2f} 秒, 结果:{retrieved_docs}")

        # 提取文档文本用于重排序
        t2 = time.time()
        documents_for_rerank = [doc['text'] for doc in retrieved_docs]

        if not documents_for_rerank:
            return []

        # 2. 对搜索结果进行重排序
        reranked_results = self.rerank(query, documents_for_rerank, top_k=top_k)

        # 3. 组合重排序结果与原始文档的元数据
        final_results = []
        for score, doc_text in reranked_results:
            # 找到对应的原始文档，这里假设文本是唯一的或者第一个匹配的即可
            original_doc_info = next((doc for doc in retrieved_docs if doc['text'] == doc_text), None)
            if original_doc_info:
                final_results.append({
                    'score': score,
                    'text': original_doc_info['text'],
                    'file_name': original_doc_info['file_name'],
                    'page_number': original_doc_info['page_number']
                })
        t3 = time.time()
        print(f"重排序已完成，耗时: {t3 - t2:.2f} 秒,  结果：{final_results}")
        return final_results


if __name__ == '__main__':
    pass
