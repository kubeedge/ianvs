# Copyright 2024 New Government Agent Project
# 新版政府文档解析器 - 基于PyPDF2和LLM

import os
import json
import re
import logging
from typing import Dict, Any

# 导入文档解析库
import PyPDF2

# 导入LLM相关库
from openai import OpenAI

class NewGovernmentDocumentParser:
    """
    新版政府文档解析器
    
    基于PyPDF2和LLM的轻量级政府文档解析器
    负责解析PDF格式的政府报告，提取文本内容
    并使用LLM进行智能分析和结构化处理
    """
    
    def __init__(self, llm_base_url: str, api_key: str, model="qwen-max"):
        """
        初始化解析器
        
        Args:
            llm_base_url: LLM API基础URL
            api_key: API密钥
            model: LLM模型
        """
        self.llm_base_url = llm_base_url
        self.api_key = api_key
        self.model = model
        # 初始化OpenAI客户端
        self.client = OpenAI(
            base_url=llm_base_url,
            api_key=api_key
        )
        
        # 政府文档类型关键词
        self.rule_type_keywords = {
            "guiding_policy": ["意见", "规划", "决议", "方案", "纲要", "战略", "政策", "指导"],
            "implementation": ["办法", "规定", "细则", "措施", "实施方案", "操作规程", "执行", "实施"],
            "service_guide": ["指南", "步骤", "流程", "服务", "办事指南", "操作手册", "服务", "便民"],
            "notices": ["公告", "通知", "声明", "发布", "通告", "公示", "告知", "通知"],
            "faq": ["常见问题", "咨询", "问答", "解答", "问题", "疑问", "FAQ", "帮助"],
            "legal_documents": ["法律", "条例", "规章", "条款", "法规", "法令", "条文", "法律"]
        }
        
        # 政府文档结构模式
        self.structure_patterns = {
            "title_pattern": r"^[一二三四五六七八九十\d]+[、\.]\s*(.+)$",
            "section_pattern": r"^第[一二三四五六七八九十\d]+[章节条]\s*(.+)$",
            "item_pattern": r"^[（\(][一二三四五六七八九十\d]+[）\)]\s*(.+)$",
            "subitem_pattern": r"^[（\(][一二三四五六七八九十\d]+[）\)]\s*(.+)$"
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def parse_document(self, file_path: str) -> Dict[str, Any]:
        """
        解析政府文档
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            Dict: 解析结果，包含文本、结构化内容和元数据
        """
        try:
            # 清理路径：移除空字符和不可见字符
            actual_file_path = ''.join(char for char in str(file_path) if ord(char) >= 32).strip()
        
            
            # 检查文件是否存在
            if not os.path.exists(actual_file_path):
                self.logger.warning(f"文件不存在: {actual_file_path}")
                # 尝试使用example.pdf作为默认文件
                example_path = "/home/linux/Desktop/government_agent/ianvs/examples/new_government_agent/resources/datasets/test/notices/example.pdf"
                if os.path.exists(example_path):
                    actual_file_path = example_path
                    self.logger.info(f"使用默认文件: {actual_file_path}")
                else:
                    return self._create_error_result(str(file_path), f"文件不存在: {actual_file_path}")
            
            # 使用PyPDF2提取文本
            text_content = self._extract_text_with_pypdf2(actual_file_path)
            
            if not text_content.strip():
                self.logger.warning("PDF文本提取为空")
                return self._create_error_result(file_path, "PDF文本提取为空")
            
            # 使用LLM进行智能分析
            structured_content = self._analyze_with_llm(text_content)
            
            # 智能分类文档类型
            rule_type = self._classify_document_type(text_content)
            
            # 提取关键信息
            key_info = self._extract_key_information(text_content, structured_content)
            
            # 生成摘要
            summary = self._generate_summary(text_content, rule_type)
            
            result = {
                'file_path': file_path,
                'rule_type': rule_type,
                'markdown_content': text_content,
                'structured_content': structured_content,
                'key_info': key_info,
                'summary': summary,
                'metadata': {
                    'file_size': os.path.getsize(file_path),
                    'parsing_method': 'pypdf2_llm',
                    'extraction_time': None
                }
            }
            
            self.logger.info(f"文档解析完成，类型: {rule_type}")
            return result
            
        except Exception as e:
            self.logger.error(f"解析文档失败: {str(e)}")
            return self._create_error_result(file_path, str(e))
    
    def _extract_text_with_pypdf2(self, file_path: str) -> str:
        """使用PyPDF2提取PDF文本"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += f"=== 第{page_num + 1}页 ===\n"
                            text_content += page_text + "\n\n"
                    except Exception as e:
                        self.logger.warning(f"提取第{page_num + 1}页失败: {str(e)}")
                        continue
                
                return text_content.strip()
                
        except Exception as e:
            self.logger.error(f"PyPDF2文本提取失败: {str(e)}")
            return ""
    
    def _analyze_with_llm(self, text_content: str) -> Dict[str, Any]:
        """使用LLM分析文档结构"""
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
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1
            )
            
            # 解析LLM响应
            try:
                llm_result = json.loads(response.choices[0].message.content)
                return llm_result
            except json.JSONDecodeError as e:
                self.logger.warning(f"LLM响应JSON解析失败: {str(e)}")
        
        except Exception as e:
            self.logger.warning(f"LLM分析失败: {str(e)}")
    
    
    def _classify_document_type(self, content: str) -> str:
        """智能分类文档类型"""
        content_lower = content.lower()
        type_scores = {}
        
        for rule_type, keywords in self.rule_type_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in content_lower:
                    score += 1
            type_scores[rule_type] = score
        
        # 返回得分最高的类型
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 0:
                return best_type
        
        return "notices"
    
    def _extract_key_information(self, content: str, structured_content: Dict) -> Dict[str, Any]:
        """提取关键信息"""
        key_info = {
            'main_topic': structured_content.get('main_topic', ''),
            'purpose': structured_content.get('purpose', ''),
            'target_audience': structured_content.get('target_audience', ''),
            'implementation_period': '',
            'responsible_department': structured_content.get('responsible_department', ''),
            'legal_basis': []
        }
        
        # 如果LLM没有提取到关键信息，使用正则表达式补充
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
        """生成文档摘要"""
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
            self.logger.warning(f"生成摘要失败: {str(e)}")
            return content
    
    def _create_error_result(self, file_path: str, error_msg: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'file_path': file_path,
            'error': error_msg,
            'rule_type': 'unknown',
            'markdown_content': '',
            'structured_content': {},
            'key_info': {},
            'summary': ''
        }
