"""
AstrBot 万象画卷插件 - ModelScope (魔搭) 异步任务出图实现
"""
import asyncio
import json
import time
from typing import Any, Dict, List, Optional
import aiohttp
from astrbot.api import logger

from .base import BaseProvider, build_image_generations_endpoint, build_image_edits_endpoint, build_tasks_endpoint

class ModelScopeProvider(BaseProvider):
    async def generate_image(self, prompt: str, **kwargs: Any) -> str:
        current_key = self.get_current_key()
        if not current_key:
            raise ValueError("节点未配置 API Key！")

        base_url = self.config.base_url
        ref_images = self.get_reference_images(**kwargs)
        
        # 剥离内置参数
        internal_keys = {"user_refs", "user_ref", "persona_refs", "persona_ref"}
        api_kwargs = {k: v for k, v in kwargs.items() if k not in internal_keys}

        headers = self._prepare_headers(current_key)
        headers["Content-Type"] = "application/json"
        headers["X-ModelScope-Async-Mode"] = "true"
        headers["X-ModelScope-Task-Type"] = "image_generation"

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "n": 1
        }
        payload.update(api_kwargs)

        if ref_images:
            url = build_image_edits_endpoint(base_url)
            for idx, ref_image in enumerate(ref_images[:3], start=1):
                try:
                    image_value = self.encode_local_image_to_base64(ref_image)
                    if not image_value:
                        image_value = ref_image
                    payload["image" if idx == 1 else f"image{idx}"] = image_value
                except Exception as e:
                    logger.warning(f"读取参考图失败: {e}")
        else:
            url = build_image_generations_endpoint(base_url)

        logger.info(f"📤 [ModelScope 通道] 提交任务至: {url}")

        async with self.session.post(url, json=payload, headers=headers, timeout=30) as response:
            if response.status >= 400:
                error_text = await response.text()
                raise RuntimeError(f"提交异步任务失败 (HTTP {response.status}): {error_text}")
            result = await response.json()

        task_id = result.get("id") or result.get("task_id")
        if not task_id and isinstance(result.get("data"), dict):
            task_id = result["data"].get("task_id") or result["data"].get("id")
        elif not task_id and isinstance(result.get("data"), list) and result["data"]:
            # 有些接口把 ID 放在 data 数组的第一个对象里
            if isinstance(result["data"][0], dict):
                task_id = result["data"][0].get("task_id") or result["data"][0].get("id")

        if not task_id:
            try:
                return await self._parse_immediate_result(result)
            except ValueError:
                raise ValueError(f"API 未返回任务 ID 或图片数据。原始返回: {result}")

        logger.info(f"✅ 任务提交成功，获得 Task ID: {task_id}，即将进入轮询。")
        return await self._poll_task_result(task_id, base_url, headers)

    async def _parse_immediate_result(self, result: Dict[str, Any]) -> str:
        if "data" in result and len(result["data"]) > 0:
            data_item = result["data"][0]
            if isinstance(data_item, dict):
                if "url" in data_item: return data_item["url"]
                if "b64_json" in data_item: return "data:image/png;base64," + data_item["b64_json"]

        if "images" in result and isinstance(result["images"], list) and result["images"]:
            item = result["images"][0]
            if isinstance(item, str): return item
            if isinstance(item, dict):
                if "url" in item: return item["url"]
                if "b64_json" in item: return "data:image/png;base64," + item["b64_json"]

        raise ValueError("未找到图片数据")

    async def _poll_task_result(self, task_id: str, base_url: str, headers: Dict[str, str]) -> str:
        endpoint = build_tasks_endpoint(base_url)
        poll_url = f"{endpoint}/{task_id}"

        max_wait = int(self.config.timeout)
        start_time = time.time()
        attempt = 0

        while (time.time() - start_time) < max_wait:
            attempt += 1
            await asyncio.sleep(5)
            try:
                async with self.session.get(poll_url, headers=headers, timeout=15) as response:
                    if response.status >= 400:
                        logger.warning(f"⚠️ 轮询请求失败 (HTTP {response.status})，继续重试...")
                        continue
                    data = await response.json()

                status = str(data.get("status", data.get("task_status", ""))).upper()
                logger.info(f"⏳ [图像轮询] Task ID: {task_id}, 状态: {status} (耗时 {int(time.time() - start_time)}s)")

                if status in {"SUCCESS", "SUCCEEDED", "COMPLETED", "SUCCEED"}:
                    image_url = self._extract_image_url(data)
                    if image_url:
                        return image_url
                    logger.warning(f"⚠️ 任务已完成但未找到图片，等待下一次轮询...")
                    continue

                if status in {"FAIL", "FAILED", "FAILURE"}:
                    logger.error(f"❌ [图像轮询] 任务失败，完整返回: {json.dumps(data, ensure_ascii=False)}")
                    error_msg = data.get("error", data.get("message", data.get("task_status_msg", data.get("msg", "未知失败原因"))))
                    if isinstance(error_msg, dict):
                        error_msg = error_msg.get("message", str(error_msg))
                    raise RuntimeError(f"平台反馈生成失败：{error_msg}")

            except (RuntimeError, ValueError):
                raise
            except Exception as exc:
                logger.warning(f"⚠️ 轮询过程异常: {exc}")

        raise RuntimeError(f"图像生成轮询超时，已等待 {max_wait} 秒。")

    def _extract_image_url(self, data: Dict[str, Any]) -> str:
        if "output_images" in data and isinstance(data["output_images"], list) and data["output_images"]:
            return data["output_images"][0]

        if "data" in data and isinstance(data["data"], list) and data["data"]:
            item = data["data"][0]
            if isinstance(item, dict):
                return item.get("url", item.get("b64_json", ""))
        
        if "images" in data and isinstance(data["images"], list) and data["images"]:
            item = data["images"][0]
            if isinstance(item, str): return item
            if isinstance(item, dict):
                return item.get("url", item.get("b64_json", ""))
        
        return data.get("url", data.get("image_url", ""))
