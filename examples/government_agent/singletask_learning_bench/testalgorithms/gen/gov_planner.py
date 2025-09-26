# Copyright 2024 New Government Agent Project
# 新版政府海报规划器

import json
import logging
from typing import Dict, Any, Optional

# 导入LLM相关库
from openai import OpenAI

class NewGovernmentPosterPlanner:
    """
    新版政府海报规划器
    
    负责根据解析后的政府文档内容，规划海报的布局和内容结构
    使用LLM进行A4纸区域规划，确保不省略内容
    """
    
    def __init__(self, llm_base_url: str, api_key: str, model="qwen-max"):
        """
        初始化规划器
        
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
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def plan_poster_layout(self, parsed_data: Dict[str, Any], quality_feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        规划海报布局（LLM驱动A4纸区域规划，不省略内容）
        
        Args:
            parsed_data: 解析后的文档数据
            
        Returns:
            Dict: 布局规划结果
        """
        try:
            rule_type = parsed_data.get('rule_type', 'notices')
            self.logger.info(f"开始规划海报布局，文档类型: {rule_type}")
            
            # 使用LLM进行A4纸区域规划
            layout_plan = self._llm_plan_a4_layout(parsed_data, quality_feedback)
            
            if layout_plan:
                self.logger.info("LLM海报布局规划完成")
                return layout_plan
            else:
                self.logger.warning("LLM规划失败，请重试")
                return None
            
        except Exception as e:
            self.logger.error(f"规划海报布局失败: {str(e)}")
            return None
    
    def _llm_plan_a4_layout(self, parsed_data: Dict[str, Any], quality_feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """使用LLM规划横向A4纸三列布局（不省略内容）"""
        try:
            # 横向A4纸尺寸（像素，300 DPI）
            a4_width = 3508  # 11.69英寸 * 300 DPI
            a4_height = 2480  # 8.27英寸 * 300 DPI
            
            # 准备内容信息
            content_info = self._prepare_content_for_llm(parsed_data)
            
            # 构建质量反馈信息
            feedback_info = ""
            if quality_feedback:
                feedback_info = f"""
质量反馈信息：
- 当前评分: {quality_feedback.get('current_score', 0):.3f}
- 目标评分: {quality_feedback.get('target_score', 0.8):.3f}
- 评估建议: {', '.join(quality_feedback.get('evaluation_suggestions', []))}
- 质量问题: {', '.join(quality_feedback.get('quality_issues', []))}
- 优化轮次: {quality_feedback.get('iteration', 1)}

请根据以上反馈信息优化布局设计，重点关注：
1. 解决已识别的质量问题
2. 采纳评估建议
3. 提升整体质量评分
"""
            
            prompt = f"""
请为以下政府文档设计横向A4纸灵活三列布局海报，要求：
1. 横向A4纸尺寸：{a4_width}x{a4_height}像素
2. 根据文档实际内容数量决定区域数量，不要强行分块
3. 内容块可以是任意大小：全宽、半宽、1/3宽等
4. 内容块可以跨列，也可以只占半列
5. 从上到下、从左到右自然衔接，充分利用空间
6. 筛选重组信息，但每一板块生成的内容要完整流畅
7. 在content字段中填入实际的具体内容，不要使用占位符
8. 标题块只需要填写name字段，content字段为空

文档信息：
{content_info}

{feedback_info}

请分析文档内容，根据实际需要创建合适数量的内容区域（通常3-6个区域即可），返回JSON格式的布局规划：

{{
    "canvas_width": {a4_width},
    "canvas_height": {a4_height},
    "regions": [
        {{
            "id": "title_region",
            "name": "请输入文档标题",
            "content": "",
            "x": 0,
            "y": 0,
            "width": 3508,
            "height": 300,
            "font_size": 84,
            "font_color": [255, 255, 255],
            "background_color": [47, 85, 151],
            "text_align": "center",
            "priority": 1
        }},
        {{
            "id": "content_block_1",
            "name": "根据内容命名",
            "content": "请根据文档内容填入具体信息",
            "x": 0,
            "y": 300,
            "width": 1169,
            "height": 600,
            "font_size": 36,
            "font_color": [0, 0, 0],
            "background_color": [248, 248, 255],
            "text_align": "left",
            "priority": 2
        }},
        {{
            "id": "content_block_2",
            "name": "根据内容命名",
            "content": "请根据文档内容填入具体信息",
            "x": 1169,
            "y": 300,
            "width": 1169,
            "height": 600,
            "font_size": 36,
            "font_color": [0, 0, 0],
            "background_color": [255, 248, 248],
            "text_align": "left",
            "priority": 3
        }},
        {{
            "id": "content_block_3",
            "name": "根据内容命名",
            "content": "请根据文档内容填入具体信息",
            "x": 2338,
            "y": 300,
            "width": 1170,
            "height": 600,
            "font_size": 36,
            "font_color": [0, 0, 0],
            "background_color": [248, 255, 248],
            "text_align": "left",
            "priority": 4
        }},
        {{
            "id": "content_block_4",
            "name": "根据内容命名",
            "content": "请根据文档内容填入具体信息",
            "x": 0,
            "y": 900,
            "width": 1754,
            "height": 800,
            "font_size": 32,
            "font_color": [0, 0, 0],
            "background_color": [255, 255, 248],
            "text_align": "left",
            "priority": 5
        }},
        {{
            "id": "content_block_5",
            "name": "根据内容命名",
            "content": "请根据文档内容填入具体信息",
            "x": 1754,
            "y": 900,
            "width": 1754,
            "height": 800,
            "font_size": 32,
            "font_color": [0, 0, 0],
            "background_color": [248, 255, 255],
            "text_align": "left",
            "priority": 6
        }},
        {{
            "id": "content_block_6",
            "name": "根据内容命名",
            "content": "请根据文档内容填入具体信息",
            "x": 0,
            "y": 1700,
            "width": 3508,
            "height": 780,
            "font_size": 30,
            "font_color": [0, 0, 0],
            "background_color": [255, 248, 255],
            "text_align": "left",
            "priority": 7
        }}
    ]
}}

重要要求：
- 根据文档实际内容决定区域数量，不要强行分块
- 确保所有区域覆盖整个横向A4纸，无空白
- 内容块可以跨列、半列、全宽等
- 从上到下、从左到右自然衔接
- 字体大小要适合区域大小（参考标题84px，大块36px，中块32px，小块30px，需要根据区域大小合理调整）
- 颜色搭配要符合政府文档风格，使用不同背景色区分内容块
- 在content字段中填入实际的具体内容，不要使用"具体内容"、"此处省略"等占位符，标题块content字段为空
- 筛选重组信息，但每一板块生成的内容要完整流畅
- 根据文档内容合理分配和组合信息到各个内容块中，摘要和重要的信息要放在前面
- 区域名称要反映实际内容，如"基本信息"、"办理流程"、"联系信息"等

{quality_feedback}
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8000,
                temperature=0.3
            )
            
            # 解析LLM响应
            try:
                response_text = response.choices[0].message.content.strip()
                
                # 尝试提取JSON部分
                if '```json' in response_text:
                    json_start = response_text.find('```json') + 7
                    json_end = response_text.find('```', json_start)
                    if json_end > json_start:
                        response_text = response_text[json_start:json_end].strip()
                elif '{' in response_text and '}' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    response_text = response_text[json_start:json_end]
                
                layout_data = json.loads(response_text)
                
                # 验证和调整布局
                validated_layout = self._validate_and_adjust_layout(layout_data, a4_width, a4_height)
                
                return validated_layout
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"LLM响应JSON解析失败: {str(e)}")
                self.logger.warning(f"响应内容: {response.choices[0].message.content}")
                return None
                
        except Exception as e:
            self.logger.error(f"LLM规划A4布局失败: {str(e)}")
            return None
    
    def _prepare_content_for_llm(self, parsed_data: Dict[str, Any]) -> str:
        """为LLM准备内容信息（不省略内容）"""
        try:
            content_parts = []
            
            # 基本信息
            content_parts.append(f"文档类型: {parsed_data.get('rule_type', 'unknown')}")
            content_parts.append(f"文档标题: {parsed_data.get('structured_content', {}).get('title', '无标题')}")
            
            # 主要内容（不截断）
            markdown_content = parsed_data.get('markdown_content', '')
            if markdown_content:
                content_parts.append(f"文档内容: {markdown_content}")
            
            # 关键信息
            key_info = parsed_data.get('key_info', {})
            if key_info:
                content_parts.append(f"主要主题: {key_info.get('main_topic', '')}")
                content_parts.append(f"目的: {key_info.get('purpose', '')}")
                content_parts.append(f"目标受众: {key_info.get('target_audience', '')}")
                content_parts.append(f"负责部门: {key_info.get('responsible_department', '')}")
            
            # 结构化内容
            structured = parsed_data.get('structured_content', {})
            if structured.get('key_points'):
                content_parts.append(f"关键点: {', '.join(structured['key_points'])}")
            
            if structured.get('contact_info'):
                contact = structured['contact_info']
                if contact.get('phones'):
                    content_parts.append(f"联系电话: {', '.join(contact['phones'])}")
                if contact.get('emails'):
                    content_parts.append(f"电子邮箱: {', '.join(contact['emails'])}")
                if contact.get('addresses'):
                    content_parts.append(f"联系地址: {', '.join(contact['addresses'])}")
            
            if structured.get('deadlines'):
                content_parts.append(f"截止时间: {', '.join(structured['deadlines'])}")
            
            if structured.get('requirements'):
                content_parts.append(f"要求: {', '.join(structured['requirements'])}")
            
            # 摘要
            if parsed_data.get('summary'):
                content_parts.append(f"摘要: {parsed_data['summary']}")
            
            return '\n'.join(content_parts)
            
        except Exception as e:
            self.logger.warning(f"准备LLM内容失败: {str(e)}")
            return "内容准备失败"
    
    def _validate_and_adjust_layout(self, layout_data: Dict[str, Any], canvas_width: int, canvas_height: int) -> Dict[str, Any]:
        """验证和调整布局数据"""
        try:
            regions = layout_data.get('regions', [])
            
            if not regions:
                return None
            
            # 确保所有区域覆盖整个画布
            total_area = 0
            for region in regions:
                width = region.get('width', 0)
                height = region.get('height', 0)
                total_area += width * height
            
            canvas_area = canvas_width * canvas_height
            
            # 如果覆盖不完整，调整区域大小
            if total_area < canvas_area * 0.9:  # 至少覆盖90%
                scale_factor = (canvas_area * 0.95) / total_area
                for region in regions:
                    region['width'] = int(region.get('width', 0) * scale_factor)
                    region['height'] = int(region.get('height', 0) * scale_factor)
            
            # 确保坐标和尺寸在画布范围内
            for region in regions:
                region['x'] = max(0, min(region.get('x', 0), canvas_width - region.get('width', 0)))
                region['y'] = max(0, min(region.get('y', 0), canvas_height - region.get('height', 0)))
                region['width'] = max(100, min(region.get('width', 100), canvas_width - region.get('x', 0)))
                region['height'] = max(50, min(region.get('height', 50), canvas_height - region.get('y', 0)))
            
            # 添加元数据
            layout_data['metadata'] = {
                'canvas_width': canvas_width,
                'canvas_height': canvas_height,
                'total_regions': len(regions),
                'layout_type': 'a4_llm_planned'
            }
            
            return layout_data
            
        except Exception as e:
            self.logger.warning(f"验证调整布局失败: {str(e)}")
            return None
    