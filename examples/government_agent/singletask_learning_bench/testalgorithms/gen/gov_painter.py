# Copyright 2024 New Government Agent Project
# 新版政府海报绘制器

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
# 注释掉本地图像处理依赖，使用API调用
# import PIL.Image
# import PIL.ImageDraw
# import PIL.ImageFont
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

class NewGovernmentPosterPainter:
    """
    新版政府海报绘制器
    
    负责根据布局规划生成PowerPoint格式的海报
    应用政府风格主题，支持多种输出格式
    提供高质量的可视化效果
    """
    
    def __init__(self, width: int = 48, height: int = 36):
        """
        初始化绘制器
        
        Args:
            width: 海报宽度（英寸）
            height: 海报高度（英寸）
        """
        self.width = width
        self.height = height
        
        # 多种主题配色方案
        self.color_schemes = {
            'government_blue': {
                'primary': (47, 85, 151),      # 政府蓝
                'secondary': (255, 255, 255),  # 白色
                'accent': (220, 220, 220),     # 浅灰
                'text': (0, 0, 0),             # 黑色
                'highlight': (255, 193, 7),    # 金色
                'success': (40, 167, 69),      # 绿色
                'warning': (255, 193, 7),      # 黄色
                'danger': (220, 53, 69)        # 红色
            },
            'modern_red': {
                'primary': (178, 34, 34),      # 深红色
                'secondary': (255, 255, 255),  # 白色
                'accent': (240, 240, 240),     # 浅灰
                'text': (0, 0, 0),             # 黑色
                'highlight': (255, 215, 0),    # 金色
                'success': (34, 139, 34),      # 森林绿
                'warning': (255, 140, 0),      # 橙色
                'danger': (220, 20, 60)        # 深红色
            },
            'elegant_green': {
                'primary': (0, 100, 0),        # 深绿色
                'secondary': (255, 255, 255),  # 白色
                'accent': (245, 245, 245),     # 浅灰
                'text': (0, 0, 0),             # 黑色
                'highlight': (255, 215, 0),    # 金色
                'success': (0, 128, 0),        # 绿色
                'warning': (255, 165, 0),      # 橙色
                'danger': (220, 20, 60)        # 深红色
            },
            'professional_gray': {
                'primary': (64, 64, 64),       # 深灰色
                'secondary': (255, 255, 255),  # 白色
                'accent': (230, 230, 230),     # 浅灰
                'text': (0, 0, 0),             # 黑色
                'highlight': (70, 130, 180),   # 钢蓝色
                'success': (46, 125, 50),      # 深绿色
                'warning': (255, 152, 0),      # 橙色
                'danger': (211, 47, 47)        # 红色
            },
            'warm_orange': {
                'primary': (255, 140, 0),      # 橙色
                'secondary': (255, 255, 255),  # 白色
                'accent': (255, 248, 220),     # 浅橙色
                'text': (0, 0, 0),             # 黑色
                'highlight': (255, 215, 0),    # 金色
                'success': (34, 139, 34),      # 森林绿
                'warning': (255, 69, 0),       # 红橙色
                'danger': (220, 20, 60)        # 深红色
            }
        }
        
        # 字体配置（增大字体）
        self.font_config = {
            'title': {'name': 'Microsoft YaHei', 'size': 72, 'bold': True},
            'heading': {'name': 'Microsoft YaHei', 'size': 48, 'bold': True},
            'body': {'name': 'Microsoft YaHei', 'size': 36, 'bold': False},
            'caption': {'name': 'Microsoft YaHei', 'size': 28, 'bold': False}
        }
        
        # 布局配置
        self.layout_config = {
            'margin': 0.05,  # 边距比例
            'padding': 0.02,  # 内边距比例
            'line_spacing': 1.2,
            'section_spacing': 0.03
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _get_default_theme(self) -> Dict[str, Any]:
        """获取默认政府主题"""
        return {
            'name': 'government_blue',
            'secondary_color': (255, 255, 255),  # 白色背景
            'panel_visible': True,
            'panel_theme': {
                'color': (47, 85, 151),  # 政府蓝
                'thickness': 3
            },
            'logo_placement': None,
            'watermark': True
        }
    
    def _get_theme_by_name(self, theme_name: str) -> Dict[str, Any]:
        """根据主题名称获取主题配置"""
        if theme_name not in self.color_schemes:
            self.logger.warning(f"主题 '{theme_name}' 不存在，使用默认主题")
            theme_name = 'government_blue'
        
        return {
            'name': theme_name,
            'colors': self.color_schemes[theme_name],
            'fonts': self.font_config,
            'layout': self.layout_config,
            'background_color': self.color_schemes[theme_name]['secondary'],
            'logo_placement': None,
            'watermark': True
        }
    
    def generate_poster(self, parsed_data: Dict[str, Any], layout_plan: Dict[str, Any], theme: Optional[Dict[str, Any]] = None, theme_name: str = 'government_blue') -> Dict[str, Any]:
        """
        生成海报（直接生成图片，不省略内容）
        
        Args:
            parsed_data: 解析后的文档数据
            layout_plan: 布局规划
            theme: 主题配置（可选，如果未提供则使用默认主题）
            theme_name: 主题名称，可选值：government_blue, modern_red, elegant_green, professional_gray, warm_orange
            
        Returns:
            Dict: 生成结果
        """
        try:
            self.logger.info("开始生成海报...")
            
            # 获取画布尺寸
            canvas_width = layout_plan.get('canvas_width', 2480)
            canvas_height = layout_plan.get('canvas_height', 3508)
            
            # 创建图片画布
            from PIL import Image, ImageDraw, ImageFont
            
            # 应用主题
            if theme is None:
                theme = self._get_theme_by_name(theme_name)
            
            # 创建背景（使用主题颜色）
            background_color = theme.get('background_color', (255, 255, 255))
            img = Image.new('RGB', (canvas_width, canvas_height), background_color)
            draw = ImageDraw.Draw(img)
            
            # 绘制各个区域
            regions = layout_plan.get('regions', [])
            
            for i, region in enumerate(regions):
                try:
                    self._draw_region(draw, region, canvas_width, canvas_height)
                except Exception as e:
                    self.logger.error(f"区域 {i+1} 绘制失败: {str(e)}")
                    continue
            
            # 保存图片
            output_path = self._save_poster_image(img, parsed_data)
            
            result = {
                'poster_path': output_path,
                'image_paths': [output_path],  # 直接返回图片路径
                'dimensions': {
                    'width': canvas_width,
                    'height': canvas_height
                },
                'regions_count': len(regions),
                'generation_time': time.time(),
                'success': True
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"生成海报失败: {str(e)}")
            return {
                'error': str(e),
                'poster_path': None,
                'image_paths': [],
                'success': False
            }
    
    def _draw_region(self, draw, region: Dict[str, Any], canvas_width: int, canvas_height: int):
        """绘制单个区域（不省略内容）"""
        try:
            # 获取区域属性
            x = region.get('x', 0)
            y = region.get('y', 0)
            width = region.get('width', 100)
            height = region.get('height', 50)
            content = region.get('content', '')
            font_size = region.get('font_size', 36)  # 增大默认字体
            font_color = tuple(region.get('font_color', [0, 0, 0]))
            background_color = tuple(region.get('background_color', [255, 255, 255]))
            text_align = region.get('text_align', 'left')
            
            # 绘制背景
            draw.rectangle([x, y, x + width, y + height], fill=background_color)
            
            # 绘制边框（使用主题颜色）
            border_color = (47, 85, 151)  # 默认政府蓝
            # 可以根据区域类型选择不同的边框颜色
            if 'title' in region.get('name', '').lower():
                border_color = (47, 85, 151)  # 标题区域使用政府蓝
            elif 'content' in region.get('name', '').lower():
                border_color = (70, 130, 180)  # 内容区域使用钢蓝色
            else:
                border_color = (105, 105, 105)  # 其他区域使用深灰色
            
            draw.rectangle([x, y, x + width, y + height], outline=border_color, width=3)
            
            # 如果没有内容，使用区域名称作为内容
            if not content:
                content = region.get('name', '未知区域')
            
            # 添加区域标题（使用更大字体）
            region_name = region.get('name', '未知区域')
            title_font_size = int(font_size * 1.5)  # 标题字体比内容大50%
            
            if region_name and region_name != content:
                # 在内容前添加区域标题
                content = f"【{region_name}】\n{content}"
            
            # 加载字体
            from PIL import ImageFont
            
            # 尝试加载中文字体
            font_paths = [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc", 
                "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "arial.ttf",
                "/System/Library/Fonts/Arial.ttf"
            ]
            
            font = None
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except Exception as e:
                    continue
            
            if font is None:
                self.logger.warning("无法加载任何字体，使用默认字体")
                font = ImageFont.load_default()
            
            # 分离区域标题和内容
            if content.startswith('【') and '】' in content:
                # 提取标题和内容
                title_end = content.find('】')
                title_text = content[:title_end + 1]
                body_text = content[title_end + 1:].strip()
                
                # 加载标题字体（更大）
                title_font = None
                for font_path in font_paths:
                    try:
                        title_font = ImageFont.truetype(font_path, title_font_size)
                        break
                    except:
                        continue
                if title_font is None:
                    title_font = ImageFont.load_default()
                
                # 绘制标题
                title_x, title_y = self._calculate_text_position(
                    x, y, width, height, title_text, title_font, text_align
                )
                draw.text((title_x, title_y), title_text, font=title_font, fill=font_color)
                
                # 绘制内容（如果有）
                if body_text:
                    # 计算内容起始位置（标题下方）
                    content_y = title_y + title_font_size + 10
                    
                    # 处理内容换行
                    wrapped_body = self._wrap_text_for_region(body_text, width, font, font_size)
                    
                    # 绘制内容
                    draw.multiline_text(
                        (title_x, content_y),
                        wrapped_body,
                        font=font,
                        fill=font_color,
                        align=text_align
                    )
                    
                    self.logger.info(f"绘制区域 {region.get('name', 'unknown')}: 标题字体={title_font_size}, 内容字体={font_size}")
                    self.logger.info(f"标题: {title_text}")
                else:
                    self.logger.info(f"绘制区域 {region.get('name', 'unknown')}: 仅标题，字体={title_font_size}")
            else:
                # 没有标题格式，按原方式绘制
                wrapped_text = self._wrap_text_for_region(content, width, font, font_size)
                
                # 计算文本位置
                text_x, text_y = self._calculate_text_position(
                    x, y, width, height, wrapped_text, font, text_align
                )
                
                # 绘制文本
                draw.multiline_text(
                    (text_x, text_y),
                    wrapped_text,
                    font=font,
                    fill=font_color,
                    align=text_align
                )
                
                self.logger.info(f"绘制区域 {region.get('name', 'unknown')}: 字体大小={font_size}, 内容长度={len(content)}")
            
        except Exception as e:
            self.logger.warning(f"绘制区域失败: {str(e)}")
    
    def _wrap_text_for_region(self, text: str, region_width: int, font, font_size: int) -> str:
        """为区域包装文本（简化版，确保内容完整）"""
        try:
            if not text:
                return ""
            
            # 简化换行逻辑，基于区域宽度和字体大小估算
            # 中文字符大约占字体大小的宽度
            chars_per_line = max(20, region_width // (font_size // 2))
            
            # 如果文本不长，直接返回
            if len(text) <= chars_per_line:
                return text
            
            # 简单的按字符换行，保持内容完整
            lines = []
            for i in range(0, len(text), chars_per_line):
                line = text[i:i + chars_per_line]
                lines.append(line)
            
            return '\n'.join(lines)
            
        except Exception as e:
            self.logger.warning(f"文本换行失败: {str(e)}")
            return text
    
    def _calculate_text_position(self, x: int, y: int, width: int, height: int, 
                               text: str, font, align: str) -> tuple:
        """计算文本位置（简化版）"""
        try:
            # 计算X坐标
            if align == 'center':
                text_x = x + width // 4  # 简单居中
            elif align == 'right':
                text_x = x + width - width // 4
            else:  # left
                text_x = x + 20  # 左边距
            
            # Y坐标从区域顶部开始，留出一些边距
            text_y = y + 30
            
            return text_x, text_y
            
        except Exception as e:
            self.logger.warning(f"计算文本位置失败: {str(e)}")
            return x + 20, y + 30
    
    def _save_poster_image(self, img, parsed_data: Dict[str, Any]) -> str:
        """保存海报图片"""
        try:
            # 创建输出目录
            output_dir = "/home/linux/Desktop/government_agent/ianvs/workspace/new_government_agent_output/posters"
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = int(time.time())
            rule_type = parsed_data.get('rule_type', 'unknown')
            file_name = f"government_poster_{rule_type}_{timestamp}.png"
            output_path = os.path.join(output_dir, file_name)
            
            # 保存图片
            img.save(output_path, 'PNG', quality=95)
            self.logger.info(f"海报图片已保存到: {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"保存海报图片失败: {str(e)}")
            return ""

   