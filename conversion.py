import time
import threading
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.callbacks import get_openai_callback

# 配置 OpenAI API 密钥和 Base URL 以连接 vLLM 服务
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"

chat = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    model_name="DeepSeek-R1-Distill-Qwen-7B",
    max_tokens=1024
)


memory = ConversationBufferMemory()

# 创建对话链
conversation = ConversationChain(
    llm=chat,
    memory=memory,
    verbose=False
)

# token 阈值
token_threshold = 800
# 滑动窗口
window_size = 10

# 超时标记
timeout = False


def check_timeout():
    global timeout
    time.sleep(60)
    if not timeout:
        print("长时间没有输入，对话结束。")
        timeout = True


while not timeout:
    timer = threading.Timer(60, check_timeout)
    timer.start()
    try:
        user_input = input("输入对话： ")
        if user_input.lower() == 'exit':
            timeout = True
            break
        with get_openai_callback() as cb:
            _ = chat([HumanMessage(content=memory.buffer)])
            current_tokens = cb.total_tokens

        # 超过阈值，使用滑动窗口保留最近的消息
        while current_tokens > token_threshold:
            messages = memory.chat_memory.messages
            if len(messages) <= window_size:
                break
            del messages[0]
            with get_openai_callback() as cb:
                _ = chat([HumanMessage(content=memory.buffer)])
                current_tokens = cb.total_tokens
        model_reply = conversation.predict(input=user_input)
        print("模型: ", model_reply)

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        timer.cancel()