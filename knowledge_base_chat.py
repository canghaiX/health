"""
date: 2025/3/14
知识库问答接口
"""

from fastapi import APIRouter, Request,Body,Response
from config import get_logger

from sse_starlette.sse import EventSourceResponse

from ollama import AsyncClient
from typing import Optional
import json
from routers.knowledge_inter import retrieval
from openai import OpenAI
import aiohttp
import requests

from openai import AsyncOpenAI
client = AsyncOpenAI(
    api_key="sk-7947b081778614c7fe1cda",
    base_url="https://compatible-mode/v1"
)
# 设置 OpenAI 的 API key 和 API base 来使用 vLLM 的 API server.
openai_api_key = "EMPTY"  # 如果不需要 API key，可以留空或设置为 "EMPTY"
openai_api_base = "http://localhost:8000/v1"

#api_key="sk-762b0529add947b081778614c7fe1cda", 
#base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
#url = "http://localhost:8000/v1/chat/completions"
#url = 'http://localhost:7001/generate_text/'
model_path = '/home/ysdx2025model/qwen/Qwen/Qwen2___5-14B-Instruct'
router = APIRouter()

logger = get_logger(__name__)

# @router.options("/knowledge_base_chat/")
# async def options_knowledge_base_chat(request: Request):
#     response = Response(status_code=204)
#     response.headers["Allow"] = "POST"
#     return response


@router.post("/knowledge_base_chat/")
async def knowledge_base_chat(
                   kbId: Optional[str] = Body(None,description='知识库编号，不给时默认从所有知识库里检索'),
                   query: str = Body(...,desription='用户输入',examples=["你是谁"]),
                   history:list = Body([],description="历史对话记录"),
                   stream:bool=Body(True,description="是否流式输出"),
                   modelName:str = Body('qwq-plus',description='模型名称'),
                   temperature:float = Body(0.1,description="LLM 采样温度"),
                   limit:int = Body(3,description="查询最相关的limit个结果")
                   ):
    
    # 知识库问答接口
    
    async def event_generator(query,history,stream,modelName,temperature,kbId,limit):
        logger.info(f"{kbId}zhishikununmber")
        # 检索逻辑
        source = retrieval(kbId=kbId,query=query,limit=limit) 
        #logger.info(source)
                            
        result = []
        if not source:
            logger.info('检索结果为空')
        else:
            #logger.info(source['files'])
            result = source['files']
                                
            input = f"请依据检索结果，精准回答用户问题，若结果无关，可直接基于知识作答。'用户问题:{query}\n''检索结果:{result}\n '。"
            logger.info(f"问题:{input}")
            messages = [
                {
                    'role': 'system',
                    'content': '你现在的身份是医疗助手大模型。'
                },
                {
                'role': 'user',
                'content': f'{input}'
                },
                ]
            # 将history中的字典数据插入到messages列表的第二个位置
            messages[1:1] = history
            logger.info(f"完整的对话消息: {messages}")
            
            if stream:
                
                async for part in await client.chat.completions.create(
                model=modelName,
                messages=messages,
                stream=True,
                temperature=temperature,
                max_tokens=4096,
                stream_options={"include_usage":True}
                ):
                    
                    logger.info(part)
                    #delta_content = part.choices[0].delta.content
                    if(part.choices):
                        delta_content = part.choices[0].delta.content
                        logger.info(delta_content)
                        logger.info({ "data": json.dumps({"answer": delta_content}, ensure_ascii=False) })
                        yield { "data": json.dumps({"answer": delta_content}, ensure_ascii=False) }
                
                yield { 
                    "data": json.dumps({"files": source.get("files")},ensure_ascii=False)
                    }
                
            else:
                
                response = await client.chat.completions.create(
                    model=modelName,
                    messages=messages,
                    stream=False,
                    temperature=temperature
                )
                full_content = response.choices[0].message.content
                logger.info(full_content)
                yield {
                "data": json.dumps(
                    {
                        "answer": full_content,
                        "files": source.get("files")
                    },ensure_ascii=False)
                }
                
            
            """
            # 构造请求数据
            data = {
                    "model": modelName,
                    "messages": messages,
                    "stream": stream
                }
                
            try:
                response = requests.post(url=url,json=data,stream=stream)
                logger.info(f'------{response}')
                response.raise_for_status()
                if stream:
                    for line in response.iter_lines(decode_unicode=True):
                
                        if line:
                            # 处理 "data: " 前缀
                            if line.startswith("data: "):
                                line = line[len("data: "):]
                            if line == "[DONE]":
                                break
                            
                            try:
                                chunk = json.loads(line)
                                # 提取 token 字段
                                token = chunk["choices"][0]["delta"].get("content", "")
                                if token :
                                    logger.info(f"{token}")
                                    yield {
                                        "data": json.dumps({"answer": token}, ensure_ascii=False)
                                            }
                            except json.JSONDecodeError as e:
                                logger.error(f"解析 JSON 数据时出错: {line}")
                                yield {"code": 500, "message": f"解析 JSON 数据时出错: {str(e)}"}
                            # 发送检索到的文件信息
                            yield {
                                "data": json.dumps({"files": source.get("files")}, ensure_ascii=False)
                                }
                else:
                    answer = response.json()["choices"][0]["message"]["content"]
                    logger.info(f"{answer}")
                    yield {
                        "data": json.dumps({
                        "answer": answer,
                        "files": source.get("files")
                        }, ensure_ascii=False)
                    }
            
            except Exception as e:
                logger.error(f"调用模型接口时发生错误: {e}")
                yield {"code": 500, "message": f"调用模型接口时发生错误: {str(e)}"}
            """
            """                
            # 构造请求数据
            data = {
                    "model_path": model_path,
                    "messages": messages,
                    "stream": stream
                }
                
            try:
                response = requests.post(url, json=data)
                if stream:
                    for line in response.iter_lines():
                        line = line.decode('utf-8').strip()
                        if line:
                            try:
                                result = json.loads(line)
                                # 提取 answer 字段
                                answer = json.loads(result["data"]).get("answer")
                                if answer:
                                    yield {
                                        "data": json.dumps({"answer": answer}, ensure_ascii=False)
                                            }
                            except json.JSONDecodeError as e:
                                logger.error(f"解析 JSON 数据时出错: {line}")
                                yield {"code": 500, "message": f"解析 JSON 数据时出错: {str(e)}"}
                            # 发送检索到的文件信息
                            yield {
                                "data": json.dumps({"files": source.get("files")}, ensure_ascii=False)
                                }
                        else:
                            response_data = response.json()
                            answer = json.loads(response_data["data"]).get("answer")
                            yield {
                                "data": json.dumps({
                                "answer": answer,
                                "files": source.get("files")
                                }, ensure_ascii=False)
                            }
            except Exception as e:
                logger.error(f"调用模型接口时发生错误: {e}")
                yield {"code": 500, "message": f"调用模型接口时发生错误: {str(e)}"}
        """    
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if stream:
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if line:
                                try:
                                    result = json.loads(line)
                                    # 提取 answer 字段
                                    answer = json.loads(result["data"]).get("answer")
                                    if answer:
                                        yield {
                                            "data": json.dumps({"answer": answer}, ensure_ascii=False)
                                        }
                                except json.JSONDecodeError:
                                    logger.error(f"解析 JSON 数据时出错: {line}")
                        # 发送检索到的文件信息
                        yield {
                            "data": json.dumps({"files": source.get("files")}, ensure_ascii=False)
                        }
                    else:
                        response_data = await response.json()
                        answer = json.loads(response_data["data"]).get("answer")
                        yield {
                            "data": json.dumps({
                                "answer": answer,
                                "files": source.get("files")
                            }, ensure_ascii=False)
                        }
        except Exception as e:
            logger.error(f"调用模型接口时发生错误: {e}")
            yield {
                "data": json.dumps({
                    "answer": f"发生错误: {str(e)}",
                    "files": []
                }, ensure_ascii=False)
            }
        """
        """
        if stream:
            #client = AsyncClient()
            client = OpenAI(api_key=openai_api_key,base_url=openai_api_base)
            
            #async for part in await client.chat(model=modelName, messages=messages, stream=stream):
                #token = part['message']['content']
                #logger.info(token)
            
            # 创建 Completion 请求
            async for part in await client.completions.create(model=modelName,prompt=messages,stream=stream):
                token = part.choices[0].text
                logger.info(token)
                
                yield { 
                       "data": json.dumps({"answer": token},ensure_ascii=False)
                       }
            yield { 
                    "data": json.dumps({"files": source.get("files")},ensure_ascii=False)
                    }
            
        else:
            #client = AsyncClient()
            client = OpenAI(api_key=openai_api_key,base_url=openai_api_base)
            #response = await client.chat(model=modelName, messages=messages)
            #logger.info(response['message']['content'])
            response = await client.completions.create(model=modelName,prompt=messages)
            logger.info(response.choices[0].text)
            yield {
                "data": json.dumps(
                    {
                        "answer": response.choices[0].text,
                        "files": source.get("files")
                    },ensure_ascii=False)
                }
            """
            

    return EventSourceResponse(event_generator(query=query,history=history,stream=stream,modelName=modelName,kbId=kbId,temperature=temperature,limit=limit))
