# Copyright 2024 New Government Agent Project
# New Government Poster Painter

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
# Comment out local image processing dependencies, use API calls
# import PIL.Image
# import PIL.ImageDraw
# import PIL.ImageFont
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

class NewGovernmentPosterPainter:
    """
    New Government Poster Painter
    
    Responsible for generating PowerPoint format posters based on layout planning
    Applies government style theme, supports multiple output formats
    Provides high-quality visual effects
    """
    
    def __init__(self, width: int = 48, height: int = 36):
        """
        Initialize painter
        
        Args:
            width: Poster width (inches)
            height: Poster height (inches)
        """
        self.width = width
        self.height = height
        
        # Multiple theme color schemes
        self.color_schemes = {
            'government_blue': {
                'primary': (47, 85, 151),      # Government blue
                'secondary': (255, 255, 255),  # White
                'accent': (220, 220, 220),     # Light gray
                'text': (0, 0, 0),             # Black
                'highlight': (255, 193, 7),    # Gold
                'success': (40, 167, 69),      # Green
                'warning': (255, 193, 7),      # Yellow
                'danger': (220, 53, 69)        # Red
            },
            'modern_red': {
                'primary': (178, 34, 34),      # Deep red
                'secondary': (255, 255, 255),  # White
                'accent': (240, 240, 240),     # Light gray
                'text': (0, 0, 0),             # Black
                'highlight': (255, 215, 0),    # Gold
                'success': (34, 139, 34),      # Forest green
                'warning': (255, 140, 0),      # Orange
                'danger': (220, 20, 60)        # Deep red
            },
            'elegant_green': {
                'primary': (0, 100, 0),        # Deep green
                'secondary': (255, 255, 255),  # White
                'accent': (245, 245, 245),     # Light gray
                'text': (0, 0, 0),             # Black
                'highlight': (255, 215, 0),    # Gold
                'success': (0, 128, 0),        # Green
                'warning': (255, 165, 0),      # Orange
                'danger': (220, 20, 60)        # Deep red
            },
            'professional_gray': {
                'primary': (64, 64, 64),       # Dark gray
                'secondary': (255, 255, 255),  # White
                'accent': (230, 230, 230),     # Light gray
                'text': (0, 0, 0),             # Black
                'highlight': (70, 130, 180),   # Steel blue
                'success': (46, 125, 50),      # Deep green
                'warning': (255, 152, 0),      # Orange
                'danger': (211, 47, 47)        # Red
            },
            'warm_orange': {
                'primary': (255, 140, 0),      # Orange
                'secondary': (255, 255, 255),  # White
                'accent': (255, 248, 220),     # Light orange
                'text': (0, 0, 0),             # Black
                'highlight': (255, 215, 0),    # Gold
                'success': (34, 139, 34),      # Forest green
                'warning': (255, 69, 0),       # Red orange
                'danger': (220, 20, 60)        # Deep red
            }
        }
        
        # Font configuration (enlarged font)
        self.font_config = {
            'title': {'name': 'Microsoft YaHei', 'size': 72, 'bold': True},
            'heading': {'name': 'Microsoft YaHei', 'size': 48, 'bold': True},
            'body': {'name': 'Microsoft YaHei', 'size': 36, 'bold': False},
            'caption': {'name': 'Microsoft YaHei', 'size': 28, 'bold': False}
        }
        
        # Layout configuration
        self.layout_config = {
            'margin': 0.05,  # Margin ratio
            'padding': 0.02,  # Padding ratio
            'line_spacing': 1.2,
            'section_spacing': 0.03
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _get_default_theme(self) -> Dict[str, Any]:
        """Get default government theme"""
        return {
            'name': 'government_blue',
            'secondary_color': (255, 255, 255),  # White background
            'panel_visible': True,
            'panel_theme': {
                'color': (47, 85, 151),  # Government blue
                'thickness': 3
            },
            'logo_placement': None,
            'watermark': True
        }
    
    def _get_theme_by_name(self, theme_name: str) -> Dict[str, Any]:
        """Get theme configuration by theme name"""
        if theme_name not in self.color_schemes:
            self.logger.warning(f"Theme '{theme_name}' does not exist, using default theme")
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
        Generate poster (direct image generation, no content omitted)
        
        Args:
            parsed_data: Parsed document data
            layout_plan: Layout plan
            theme: Theme configuration (optional, uses default theme if not provided)
            theme_name: Theme name, options: government_blue, modern_red, elegant_green, professional_gray, warm_orange
            
        Returns:
            Dict: Generation result
        """
        try:

            
            # Get canvas dimensions
            canvas_width = layout_plan.get('canvas_width', 2480)
            canvas_height = layout_plan.get('canvas_height', 3508)
            
            # Create image canvas
            from PIL import Image, ImageDraw, ImageFont
            
            # Apply theme
            if theme is None:
                theme = self._get_theme_by_name(theme_name)
            
            # Create background (using theme color)
            background_color = theme.get('background_color', (255, 255, 255))
            img = Image.new('RGB', (canvas_width, canvas_height), background_color)
            draw = ImageDraw.Draw(img)
            
            # Draw each region
            regions = layout_plan.get('regions', [])
            
            for i, region in enumerate(regions):
                try:
                    self._draw_region(draw, region, canvas_width, canvas_height)
                except Exception as e:
                    self.logger.error(f"Region {i+1} drawing failed: {str(e)}")
                    continue
            
            # Save image
            output_path = self._save_poster_image(img, parsed_data)
            
            result = {
                'poster_path': output_path,
                'image_paths': [output_path],  # Directly return image path
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
            self.logger.error(f"Failed to generate poster: {str(e)}")
            return {
                'error': str(e),
                'poster_path': None,
                'image_paths': [],
                'success': False
            }
    
    def _draw_region(self, draw, region: Dict[str, Any], canvas_width: int, canvas_height: int):
        """Draw single region (no content omitted)"""
        try:
            # Get region attributes
            x = region.get('x', 0)
            y = region.get('y', 0)
            width = region.get('width', 100)
            height = region.get('height', 50)
            content = region.get('content', '')
            font_size = region.get('font_size', 36)  # Enlarge default font
            font_color = tuple(region.get('font_color', [0, 0, 0]))
            background_color = tuple(region.get('background_color', [255, 255, 255]))
            text_align = region.get('text_align', 'left')
            
            # Draw background
            draw.rectangle([x, y, x + width, y + height], fill=background_color)
            
            # Draw border (using theme color)
            border_color = (47, 85, 151)  # Default government blue
            # Can select different border colors based on region type
            if 'title' in region.get('name', '').lower():
                border_color = (47, 85, 151)  # Title region uses government blue
            elif 'content' in region.get('name', '').lower():
                border_color = (70, 130, 180)  # Content region uses steel blue
            else:
                border_color = (105, 105, 105)  # Other regions use dark gray
            
            draw.rectangle([x, y, x + width, y + height], outline=border_color, width=3)
            
            # If no content, use region name as content
            if not content:
                content = region.get('name', 'Unknown region')
            
            # Add region title (using larger font)
            region_name = region.get('name', 'Unknown region')
            title_font_size = int(font_size * 1.5)  # Title font 50% larger than content
            
            if region_name and region_name != content:
                # Add region title before content
                content = f"【{region_name}】\n{content}"
            
            # Load font
            from PIL import ImageFont
            
            # Try to load Chinese font
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
                self.logger.warning("Unable to load any font, using default font")
                font = ImageFont.load_default()
            
            # Separate region title and content
            if content.startswith('【') and '】' in content:
                # Extract title and content
                title_end = content.find('】')
                title_text = content[:title_end + 1]
                body_text = content[title_end + 1:].strip()
                
                # Load title font (larger)
                title_font = None
                for font_path in font_paths:
                    try:
                        title_font = ImageFont.truetype(font_path, title_font_size)
                        break
                    except Exception as e:
                        continue
                if title_font is None:
                    title_font = ImageFont.load_default()
                
                # Draw title
                title_x, title_y = self._calculate_text_position(
                    x, y, width, height, title_text, title_font, text_align
                )
                draw.text((title_x, title_y), title_text, font=title_font, fill=font_color)
                
                # Draw content (if exists)
                if body_text:
                    # Calculate content start position (below title)
                    content_y = title_y + title_font_size + 10
                    
                    # Handle content line wrapping
                    wrapped_body = self._wrap_text_for_region(body_text, width, font, font_size)
                    
                    # Draw content
                    draw.multiline_text(
                        (title_x, content_y),
                        wrapped_body,
                        font=font,
                        fill=font_color,
                        align=text_align
                    )
                    
            else:
                # No title format, draw in original way
                wrapped_text = self._wrap_text_for_region(content, width, font, font_size)
                
                # Calculate text position
                text_x, text_y = self._calculate_text_position(
                    x, y, width, height, wrapped_text, font, text_align
                )
                
                # Draw text
                draw.multiline_text(
                    (text_x, text_y),
                    wrapped_text,
                    font=font,
                    fill=font_color,
                    align=text_align
                )
                
            
        except Exception as e:
            self.logger.warning(f"Failed to draw region: {str(e)}")
    
    def _wrap_text_for_region(self, text: str, region_width: int, font, font_size: int) -> str:
        """Wrap text for region (simplified version, ensure content is complete)"""
        try:
            if not text:
                return ""
            
            # Simplify line wrapping logic, estimate based on region width and font size
            # Chinese characters approximately take font size width
            chars_per_line = max(20, region_width // (font_size // 2))
            
            # If text is not long, return directly
            if len(text) <= chars_per_line:
                return text
            
            # Simple character-based wrapping, keep content complete
            lines = []
            for i in range(0, len(text), chars_per_line):
                line = text[i:i + chars_per_line]
                lines.append(line)
            
            return '\n'.join(lines)
            
        except Exception as e:
            self.logger.warning(f"Text wrapping failed: {str(e)}")
            return text
    
    def _calculate_text_position(self, x: int, y: int, width: int, height: int, 
                               text: str, font, align: str) -> tuple:
        """Calculate text position (simplified version)"""
        try:
            # Calculate X coordinate
            if align == 'center':
                text_x = x + width // 4  # Simple center
            elif align == 'right':
                text_x = x + width - width // 4
            else:  # left
                text_x = x + 20  # Left margin
            
            # Y coordinate starts from top of region, leave some margin
            text_y = y + 30
            
            return text_x, text_y
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate text position: {str(e)}")
            return x + 20, y + 30
    
    def _save_poster_image(self, img, parsed_data: Dict[str, Any]) -> str:
        """Save poster image"""
        try:
            # Create output directory
            output_dir = os.path.join("examples", "GovDoc2Poster", "posters")

            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename
            timestamp = int(time.time())
            rule_type = parsed_data.get('rule_type', 'unknown')
            file_name = f"government_poster_{rule_type}_{timestamp}.png"
            output_path = os.path.join(output_dir, file_name)
            
            # Save image
            img.save(output_path, 'PNG', quality=95)
            self.logger.info(f"Poster image saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to save poster image: {str(e)}")
            return ""

   