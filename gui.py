import time

import streamlit as st


if 'search_docs_btn' not in st.session_state:
    st.session_state.search_docs_btn = False

def click_button():
    st.session_state.search_docs_btn = True


# 主函数，用于构建Streamlit RAG系统界面。
st.set_page_config(page_title="RAG Challenge 2 - RTX 5080 Powered", layout="wide")
st.markdown("<h1 style='text-align: center;'>🚀 RAG Challenge 2 - RTX 5080 Powered</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>基于获奖RAG系统，由RTX 5080 GPU加速</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>⚡️ 支持多个数据集问答 | ⚡️ 向量检索 + 本地模型重排序 + Qwen3</p>",
            unsafe_allow_html=True)

st.markdown("--- ")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("查询设置")

    st.markdown("##### ❓选择数据集")
    choice_dataset = st.selectbox(" ", ["zhongxin", "Collection2", "Collection3"], label_visibility="collapsed")

    col_a, col_b = st.columns([0.7, 1.3])
    with col_a:
        enable_local_rerank = st.checkbox("本地模型重排序", value=True, label_visibility="visible")
    with col_b:
        st.markdown("##### 重排数量")
        # st.write("返回的相关文档数量")
        doc_count = st.slider(" ", 1, 20, 2, label_visibility="collapsed")

    st.markdown("--- ")
    st.markdown("##### ❓ 输入问题")
    user_query = st.text_area(" ", height=100,
                              placeholder="中芯国际在晶圆制造行业中的地位如何？其服务范围和全球布局是怎样的？",
                              label_visibility="collapsed")

    st.markdown("--- ")

    st.button("🔍 搜索", use_container_width=True, on_click=click_button)


with col2:
    from src.pipeline import singleton_pipeline
    st.subheader("检索结果")


    if st.session_state.search_docs_btn:
        t0 = time.time()
        st.success(f"数据集: {choice_dataset}, 问题: {user_query} 正在检索......")
        rag_results = singleton_pipeline.search(choice_dataset, user_query, doc_count)
        t1 = time.time()
        if rag_results:
            st.success(f"找到 {len(rag_results)} 个相关文档片段，耗时: {t1 - t0:.2f} 秒")
            for idx, item in enumerate(rag_results):
                st.markdown(f"**结果 {idx + 1}**")
                st.write(f"相似度: {item['score']}")
                st.write(f"文件: {item['file_name']}")
                st.write(f"页码: {item['page_number']}")
                st.markdown(f"内容: {item['text']}")
                st.markdown('---')
            rag_prompt = singleton_pipeline.vector.format_retrieval_results(user_query, rag_results)
            reply = singleton_pipeline.llm.send_message(rag_prompt)
            st.markdown("**最终结果：**")
            st.markdown(reply)


