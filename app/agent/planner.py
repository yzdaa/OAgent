"""
任务规划器
根据用户指令自动拆解为多个子任务步骤
支持 LLM 智能规划和规则降级策略
"""
from typing import List, Dict, Any, Optional
import json
import logging
from langchain_openai import ChatOpenAI
from config.settings import settings
from .schemas import PlanStep

logger = logging.getLogger(__name__)


class TaskPlanner:
    """任务规划器 - 将复杂任务拆解为可执行的步骤"""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """创建规划器；llm 为空时仅使用规则规划。"""
        self.llm = llm  # 规划模型；为空时只使用规则兜底。

    def plan(self, user_request: str, available_tools: List[str]) -> List[Dict[str, Any]]:
        """生成统一的 tool_name/tool_arguments 计划，失败时使用规则兜底。"""
        """
        规划任务步骤：优先使用 LLM，失败时回退到规则
        """
        # 计划协议固定为 tool_name + tool_arguments，Executor 只依赖这两个字段。
        # 1. 尝试 LLM 规划
        if self.llm:
            llm_plan = self._llm_based_plan(user_request, available_tools)
            if llm_plan:
                return llm_plan
        # 2. 规则降级
        return self._rule_based_plan(user_request)

    def _llm_based_plan(self, user_request: str, available_tools: List[str]) -> Optional[List[Dict[str, Any]]]:
        """请求 LLM 生成计划，并校验 JSON 结构和工具名称。"""
        """基于LLM进行规划，失败返回 None"""
        tools_str = ", ".join(available_tools)
        prompt = f"""你是一个任务规划专家。用户请求需要使用以下工具来完成任务：
可用工具：{tools_str}

用户请求：{user_request}

请分析请求，将任务拆解为具体的步骤。每个步骤必须使用上述工具之一，输出JSON格式的步骤列表。
示例格式：
[
    {{"tool_name": "data_analyzer", "tool_arguments": {{"action": "read_data", "file_path": "data/sales_data.xlsx"}}}},
    {{"tool_name": "data_analyzer", "tool_arguments": {{"action": "analyze"}}}},
    {{"tool_name": "data_analyzer", "tool_arguments": {{"action": "generate_report", "auto_generate_doc": true}}}},
]

确保输出有效的JSON数组，不要包含任何额外文字。"""

        try:
            model_response = self.llm.invoke(prompt).content
            start = model_response.find('[')
            end = model_response.rfind(']') + 1
            if start != -1 and end > start:
                # model_response 是 LLM 原始文本；plan_json 才是待解析的 JSON 片段。
                plan_json = model_response[start:end]
                raw_steps = json.loads(plan_json)
                # 先用 Pydantic 统一字段，再检查工具名，拒绝模型生成的虚假工具或动作。
                plan_steps = [PlanStep.model_validate(step) for step in raw_steps]
                invalid_tools = [step.tool_name for step in plan_steps
                                 if step.tool_name not in available_tools]
                if invalid_tools:
                    raise ValueError(f"计划包含未注册工具: {invalid_tools}")
                logger.info(f"LLM规划生成 {len(plan_steps)} 个步骤")
                return [step.model_dump() for step in plan_steps]
        except Exception as e:
            logger.error(f"LLM规划失败: {e}")
        return None

    def _extract_file_path(self, user_request: str) -> str:
        """从用户请求中提取表格路径；未找到时返回默认数据路径。"""
        """从用户输入中提取文件路径（供规则使用）"""
        import re
        patterns = [
            r'文件路径\s*[：:]\s*([^\s]+)',
            r'path\s*[：:]\s*([^\s]+)',
            r'文件\s*[：:]\s*([^\s]+)',
            r'(?:分析|查看|处理|使用|打开|读取|路径)?[：:\s]*(\S+\.(xlsx|xls|csv))',
            r'([^\s]+\.(xlsx|xls|csv))',
        ]
        for pattern in patterns:
            match = re.search(pattern, user_request, re.IGNORECASE)
            if match:
                return match.group(1)
        return "data/sales_data.xlsx"

    def _rule_based_plan(self, user_request: str) -> List[Dict[str, Any]]:
        """根据关键词生成确定性计划，作为 LLM 规划失败时的兜底。"""
        """基于规则的规划（兜底）"""
        user_request_lower = user_request.lower()
        file_path = self._extract_file_path(user_request)

        if "分析" in user_request and ("excel" in user_request_lower or "数据" in user_request_lower or "销售" in user_request_lower):
            return [
                {"tool_name": "data_analyzer", "tool_arguments": {"action": "read_data", "file_path": file_path}},
                {"tool_name": "data_analyzer", "tool_arguments": {"action": "clean_data"}},
                {"tool_name": "data_analyzer", "tool_arguments": {"action": "analyze"}},
                {"tool_name": "data_analyzer", "tool_arguments": {"action": "visualize"}},
                {"tool_name": "data_analyzer", "tool_arguments": {"action": "generate_report", "auto_generate_doc": True}}
            ]
        elif "报告" in user_request and "生成" in user_request:
            return [
                {"tool_name": "data_analyzer", "tool_arguments": {"action": "read_data", "file_path": file_path}},
                {"tool_name": "data_analyzer", "tool_arguments": {"action": "analyze"}},
                {"tool_name": "document_generator", "tool_arguments": {"doc_type": "report", "auto_generate": True, "title": "数据分析报告"}}
            ]
        elif "会议纪要" in user_request or "会议记录" in user_request:
            return [
                {"tool_name": "document_generator", "tool_arguments": {
                    "doc_type": "meeting_notes", "title": "会议纪要",
                    "auto_generate": True, "context": user_request
                }}
            ]
        elif "工作总结" in user_request or "周报" in user_request or "月报" in user_request:
            return [
                {"tool_name": "document_generator", "tool_arguments": {
                    "doc_type": "work_summary", "title": "工作总结",
                    "auto_generate": True, "context": user_request
                }}
            ]
        elif "审批" in user_request or "申请" in user_request:
            system = "OA"
            approval_type = "leave"
            if "报销" in user_request: approval_type = "expense"
            elif "合同" in user_request: approval_type = "contract"
            elif "采购" in user_request: approval_type = "purchase"
            elif "出差" in user_request: approval_type = "travel"
            elif "加班" in user_request: approval_type = "overtime"
            return [{"tool_name": "api_tool", "tool_arguments": {
                "action": "process_approval", "system": system,
                "approval_type": approval_type, "use_mock": True
            }}]
        elif "oa" in user_request_lower or "crm" in user_request_lower:
            system = "OA" if "oa" in user_request_lower else "CRM"
            if "状态" in user_request or "查看" in user_request:
                return [{"tool_name": "api_tool", "tool_arguments": {"action": "get_system_status", "system": system}}]
            elif "审批" in user_request or "申请" in user_request:
                return [{"tool_name": "api_tool", "tool_arguments": {
                    "action": "process_approval", "system": system,
                    "approval_type": "leave", "use_mock": True
                }}]
        elif "知识" in user_request or "制度" in user_request or "政策" in user_request or "产品" in user_request:
            return [{"tool_name": "knowledge_qa", "tool_arguments": {"question": user_request}}]
        else:
            # 默认使用知识问答
            return [{"tool_name": "knowledge_qa", "tool_arguments": {"question": user_request}}]
