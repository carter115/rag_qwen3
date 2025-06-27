from src.pipeline import singleton_pipeline


if __name__ == '__main__':
    docs_path = "。/documents/zhongxin"
    singleton_pipeline.parse_documents(docs_path)
    singleton_pipeline.save_embeding()