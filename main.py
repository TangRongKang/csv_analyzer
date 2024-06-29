import streamlit as st
import pandas as pd
from utils import dataframe_agent

#定义根据要求生成图标的函数
def create_chart(input_data,chart_type):
    df.data = pd.DataFrame(input_data["data"],columns=input_data["columns"])
    df_data.set_index(input_data["columns"][0], inplace=True)
    if  chart_type == "bar":
        st.bar_chart(df_data)
    elif chart_type == "line":
        st.line_chart(df_data)
    elif chart_type == "scatter":
        st.scatter_chart(df_data)

st.title ("💡 CSV数据分析智能工具")

with st.sidebar:
    openai_api_key = st.text_input("请输入OpenAI API密钥：", type="password")
    st.markdown("[获取OpenAI API key](https://platform.openai.com/account/api-keys)")
data = st.file_uploader("上传你的数据文件（CSV格式）：", type ="csv" )
#数据读取（借用pandas库）并且用st的表格组件展示
if data:
    st.session_state["df"] = pd.read_csv(data)
    with st.expander("原始数据"):
        #放置数据展示占用过多的页面，选择折叠展开组件
        st.dataframe(st.session_state["df"])
query = st.text_area("请输入你关于以上表格的问题，或数据提取请求，或可视化要求（支持散点图、折线图、条形图）：")
button = st.button("生成回答")

#执行条件判断
if button and not openai_api_key:
    st.info("请输入您的OpenAI API密钥")
if button and "df" not in st.session_state:
    st.info("请先上传您的数据文件")
if button and openai_api_key and "df" in st.session_state:
    with st.spinner("AI正在思考中，请稍等..."):
        response_dict  = dataframe_agent(openai_api_key,st.session_state["df"],query)
        if "answer" in response_dict:
            st.write(response_dict["answer"])
        if "table" in response_dict:
            st.table(pd.DataFrame(response_dict["table"]["data"],
                                  columns=response_dict["table"]["columns"]))
        if "bar" in response_dict:
            create_chart(response_dict["bar"], "bar")
        if "line" in response_dict:
            create_chart(response_dict["line"], "line")
        if "scatter" in response_dict:
            create_chart(response_dict["scatter"], "scatter")
