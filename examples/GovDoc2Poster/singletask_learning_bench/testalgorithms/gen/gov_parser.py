<<<<<<< HEAD
# Copyright 2024 New Government Agent Project
# New Government Document Parser - Based on PyPDF2 and LLM

import os
import json
import re
import logging
from typing import Dict, Any

# Import LLM related libraries
from openai import OpenAI

class NewGovernmentDocumentParser:
    """
    New Government Document Parser
    
    Lightweight government document parser based on PyPDF2 and LLM
    Responsible for parsing PDF format government reports, extracting text content
    and using LLM for intelligent analysis and structured processing
    """
    
    def __init__(self, llm_base_url: str, api_key: str, model="qwen-max"):
        """
        Initialize parser
        
        Args:
            llm_base_url: LLM API base URL
            api_key: API key
            model: LLM model
        """
        self.llm_base_url = llm_base_url
        self.api_key = api_key
        self.model = model
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=llm_base_url,
            api_key=api_key
        )
        
        # Government document type keywords
        self.rule_type_keywords = {
            "guiding_policy": ["意见", "规划", "决议", "方案", "纲要", "战略", "政策", "指导"],
            "implementation": ["办法", "规定", "细则", "措施", "实施方案", "操作规程", "执行", "实施"],
            "service_guide": ["指南", "步骤", "流程", "服务", "办事指南", "操作手册", "服务", "便民"],
            "notices": ["公告", "通知", "声明", "发布", "通告", "公示", "告知", "通知"],
            "faq": ["常见问题", "咨询", "问答", "解答", "问题", "疑问", "FAQ", "帮助"],
            "legal_documents": ["法律", "条例", "规章", "条款", "法规", "法令", "条文", "法律"]
        }
        
        # Government document structure patterns
        self.structure_patterns = {
            "title_pattern": r"^[一二三四五六七八九十\d]+[、\.]\s*(.+)$",
            "section_pattern": r"^第[一二三四五六七八九十\d]+[章节条]\s*(.+)$",
            "item_pattern": r"^[（\(][一二三四五六七八九十\d]+[）\)]\s*(.+)$",
            "subitem_pattern": r"^[（\(][一二三四五六七八九十\d]+[）\)]\s*(.+)$"
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def parse_document(self, text_content: str) -> Dict[str, Any]:
        """
        Parse government document
        
        Args:
            text_content: Parsed text data
            
        Returns:
            Dict: Parse result, containing text, structured content and metadata
        """
    
        if not text_content.strip():
            self.logger.warning("PDF text extraction is empty")
            return None


        # Use LLM for intelligent analysis
        structured_content = self._analyze_with_llm(text_content)

        # Intelligently classify document type
        rule_type = self._classify_document_type(text_content)

        # Extract key information
        key_info = self._extract_key_information(text_content, structured_content)

        # Generate summary
        summary = self._generate_summary(text_content, rule_type)

        result = {
            'rule_type': rule_type,
            'markdown_content': text_content,
            'structured_content': structured_content,
            'key_info': key_info,
            'summary': summary,
        }

        self.logger.info(f"Document parsing completed, type: {rule_type}")
        return result

    
    def _analyze_with_llm(self, text_content: str) -> Dict[str, Any]:
        """Use LLM to analyze document structure"""
        analyze_times = 0
        try:
            prompt = f"""
            请分析以下政府文档的结构和内容，提取关键信息：
            
            文档内容：
            {text_content}  # 完整内容
            
            请以JSON格式返回以下信息：
            {{
                "title": "文档标题",
                "sections": [
                    {{"level": 1, "text": "章节标题", "content": "章节内容摘要"}},
                    {{"level": 2, "text": "子章节标题", "content": "子章节内容摘要"}}
                ],
                "key_points": ["关键点1", "关键点2"],
                "contact_info": {{
                    "phones": ["电话号码"],
                    "emails": ["邮箱地址"],
                    "addresses": ["地址信息"]
                }},
                "deadlines": ["截止日期1", "截止日期2"],
                "requirements": ["要求1", "要求2"],
                "main_topic": "主要主题",
                "purpose": "文档目的",
                "target_audience": "目标受众",
                "responsible_department": "负责部门"
            }}
            
            请确保返回有效的JSON格式。不要出现其他无关的文字！
            """
            while analyze_times < 4:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.1
                )
                analyze_times += 1
                # 解析LLM响应
                try:
                    llm_result = json.loads(response.choices[0].message.content)
                    return llm_result
                except json.JSONDecodeError as e:
                    self.logger.warning(f"LLM response JSON parsing failed: {str(e)}, retry {analyze_times}/3 times")
                    
            return None
        
        except Exception as e:
            self.logger.warning(f"LLM analysis failed: {str(e)}")
    
    
    def _classify_document_type(self, content: str) -> str:
        """Intelligently classify document type"""
        content_lower = content.lower()
        type_scores = {}
        
        for rule_type, keywords in self.rule_type_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in content_lower:
                    score += 1
            type_scores[rule_type] = score
        
        # Return highest scoring type
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 0:
                return best_type
        
        return "notices"
    
    def _extract_key_information(self, content: str, structured_content: Dict) -> Dict[str, Any]:
        """Extract key information"""
        key_info = {
            'main_topic': structured_content.get('main_topic', ''),
            'purpose': structured_content.get('purpose', ''),
            'target_audience': structured_content.get('target_audience', ''),
            'implementation_period': '',
            'responsible_department': structured_content.get('responsible_department', ''),
            'legal_basis': []
        }
        
        # If LLM didn't extract key information, supplement with regex
        if not key_info['main_topic']:
            title_match = re.search(r'^(.+?)(?:\n|$)', content)
            if title_match:
                key_info['main_topic'] = title_match.group(1).strip()
        
        if not key_info['purpose']:
            purpose_patterns = [
                r'为[了]?([^，。\n]{10,50})',
                r'目的[是]?([^，。\n]{10,50})',
                r'旨在([^，。\n]{10,50})'
            ]
            
            for pattern in purpose_patterns:
                match = re.search(pattern, content)
                if match:
                    key_info['purpose'] = match.group(1).strip()
                    break
        
        return key_info
    
    def _generate_summary(self, content: str, rule_type: str) -> str:
        """Generate document summary"""
        try:
            prompt = f"""
            请为以下{rule_type}类型的政府文档生成一个简洁的摘要（100-200字）：
            
            内容：
            {content}
            
            摘要要求：
            1. 突出文档的主要内容和目的
            2. 包含关键的时间、地点、人物信息
            3. 语言简洁明了，符合政府文档风格
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            self.logger.warning(f"Failed to generate summary: {str(e)}")
            return content
 
=======
version https://git-lfs.github.com/spec/v1
oid sha256:f0d984270bf7bf6e8621b44a7cde70da01896fa8f001684613d7bb153faa7b09
size 8870
>>>>>>> 9676c3e (ya toh aar ya toh par)
