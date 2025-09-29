# Copyright 2024 New Government Agent Project
# 新版政府海报VLM评估器

import os
import logging
from typing import Dict, Any, Optional
import time

# 导入LLM相关库
from openai import OpenAI

class NewGovernmentPosterEvaluator:
    """
    新版政府海报VLM评估器
    
    只使用VLM进行海报质量评估，提供专业的视觉分析和建议
    """
    
    def __init__(self, llm_base_url: str, api_key: str, model="qwen-vl-max"):
        """
        初始化VLM评估器
        
        Args:
            llm_base_url: LLM API基础URL
            api_key: API密钥
        """
        self.llm_base_url = llm_base_url
        self.api_key = api_key
        self.model = model
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            base_url=llm_base_url,
            api_key=api_key
        )
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def evaluate_poster(self, parsed_data: Dict[str, Any], poster_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用VLM综合评估海报质量
        
        Args:
            parsed_data: 解析后的文档数据
            poster_result: 海报生成结果
            
        Returns:
            Dict: VLM评估结果
        """
        try:
            self.logger.info("开始VLM综合评估海报质量...")
            
            start_time = time.time()
            
            # 只使用VLM进行综合评估
            vlm_evaluation = self._evaluate_with_vlm_comprehensive(parsed_data, poster_result)
            
            # VLM评分（0-10分制）
            vlm_visual_score = vlm_evaluation.get('score', 0)


            
            evaluation_time = time.time() - start_time
            
            result = {
                'success': True,
                'score': vlm_visual_score,
                'evaluation_time': evaluation_time,
                'vlm_evaluation': vlm_evaluation,
                'improvement_suggestions': vlm_evaluation.get('suggestions', []),
                'metadata': {
                    'evaluator_version': '3.0',
                    'evaluation_timestamp': time.time(),
                    'evaluation_method': 'vlm_only'
                }
            }
            
            self.logger.info(f"VLM评估完成，总体得分: {vlm_visual_score:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"VLM评估失败: {str(e)}")
            return {
                'success': False,
                'score': 0.0,
                'error': str(e),
                'improvement_suggestions': [],
            }
    
    def _evaluate_with_vlm_comprehensive(self, parsed_data: Dict[str, Any], poster_result: Dict[str, Any]) -> Dict[str, Any]:
        """使用VLM进行综合评估（包含评分和建议）"""
        try:
            # 获取海报图片路径
            poster_path = poster_result.get('poster_path')
            if not poster_path or not os.path.exists(poster_path):
                self.logger.warning("海报图片不存在，无法进行VLM评估")
                return {
                    'score': 0,
                    'suggestions': ['海报图片不存在，无法进行评估'],
                    'detailed_scores': {},
                    'analysis': '无法分析'
                }
            
            # 读取图片并转换为base64
            import base64
            with open(poster_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # 构建VLM综合评估提示
            prompt = f"""
            请作为专业的政府海报设计评估专家，对以下海报进行全面的视觉质量评估（满分10分）：
            
            评估维度：
            1. 整体布局合理性（2分）- 评估海报整体布局是否合理，区域划分是否清晰
            2. 色彩搭配协调性（2分）- 评估颜色搭配是否协调，是否符合政府文档风格
            3. 字体大小和可读性（2分）- 评估文字是否清晰可读，字体大小是否合适
            4. 信息层次清晰度（2分）- 评估信息组织是否清晰，层次是否分明
            5. 政府文档规范性（2分）- 评估是否符合政府文档的正式性和规范性要求
            
            请从以上5个维度对海报进行评分，并给出总分（0-10分）。
            
            同时请提供3-5条需要优化的方面和具体的改进建议
            
            格式要求：
            评分:
            - 整体布局: X.X分
            - 色彩搭配: X.X分  
            - 字体可读性: X.X分
            - 信息层次: X.X分
            - 政府规范性: X.X分
            
            改进建议:
            - [具体建议1]
            - [具体建议2]
            
            请给出客观、专业、详细的评估。
            """
            
            # 调用VLM API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            # 解析VLM响应
            response_text = response.choices[0].message.content
            evaluation_result = self._parse_vlm_comprehensive_response(response_text)
            
            return evaluation_result
            
        except Exception as e:
            self.logger.warning(f"VLM综合评估失败: {str(e)}")
            return {
                'score': 0,
                'suggestions': [f'VLM评估失败: {str(e)}'],
                'detailed_scores': {},
                'analysis': '评估失败'
            }
    
    def _parse_vlm_comprehensive_response(self, response_text: str) -> Dict[str, Any]:
        """解析VLM综合评估响应"""
        try:
            import re
    
            # 提取详细评分
            detailed_scores = {}
            score_patterns = {
                '整体布局': r'整体布局[：:]\s*(\d+\.?\d*)分?',
                '色彩搭配': r'色彩搭配[：:]\s*(\d+\.?\d*)分?',
                '字体可读性': r'字体可读性[：:]\s*(\d+\.?\d*)分?',
                '信息层次': r'信息层次[：:]\s*(\d+\.?\d*)分?',
                '政府规范性': r'政府规范性[：:]\s*(\d+\.?\d*)分?'
            }
            total_score = 0
            for key, pattern in score_patterns.items():
                match = re.search(pattern, response_text)
                if match:
                    score = float(match.group(1))
                    detailed_scores[key] = min(2.0, max(0.0, score))
                    total_score += score
                else:
                    detailed_scores[key] = 0  # 默认分数
            
            # 提取改进建议
            suggestions = []
            suggestions_section = re.search(r'改进建议[：:]?\s*(.*?)(?=$)', response_text, re.DOTALL)
            if suggestions_section:
                suggestions_text = suggestions_section.group(1)
                suggestions = [line.strip() for line in suggestions_text.split('\n') if line.strip() and line.strip().startswith('-')]
                suggestions = [s[1:].strip() for s in suggestions]  # 去掉开头的'-'
            
            return {
                'score': total_score,
                'detailed_scores': detailed_scores,
                'suggestions': suggestions,
                'analysis': response_text
            }
            
        except Exception as e:
            self.logger.warning(f"解析VLM响应失败: {str(e)}")
            return {
                'score': 0,
                'detailed_scores': {},
                'suggestions': ['解析评估结果失败'],
                'analysis': response_text
            }
    
