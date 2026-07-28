# Copyright 2024 New Government Agent Project
# New Government Poster Planner

import json
import logging
from typing import Dict, Any, Optional

# Import LLM related libraries
from openai import OpenAI

class NewGovernmentPosterPlanner:
    """
    New Government Poster Planner
    
    Responsible for planning poster layout and content structure based on parsed government document content
    Uses LLM for A4 paper area planning, ensuring no content is omitted
    """
    
    def __init__(self, llm_base_url: str, api_key: str, model="qwen-max"):
        """
        Initialize planner
        
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
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def plan_poster_layout(self, parsed_data: Dict[str, Any], quality_feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Plan poster layout (LLM-driven A4 paper area planning, no content omitted)
        
        Args:
            parsed_data: Parsed document data
            
        Returns:
            Dict: Layout planning result
        """
        try:
            rule_type = parsed_data.get('rule_type', 'notices')
            self.logger.info(f"Starting poster layout planning, document type: {rule_type}")
            
            # Use LLM for A4 paper area planning
            layout_plan = self._llm_plan_a4_layout(parsed_data, quality_feedback)
            
            if layout_plan:
                self.logger.info("LLM poster layout planning completed")
                return layout_plan
            else:
                self.logger.warning("LLM planning failed, please retry")
                return None
            
        except Exception as e:
            self.logger.error(f"Failed to plan poster layout: {str(e)}")
            return None
    
    def _llm_plan_a4_layout(self, parsed_data: Dict[str, Any], quality_feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Use LLM to plan horizontal A4 paper three-column layout (no content omitted)"""
        try:
            # Horizontal A4 paper dimensions (pixels, 300 DPI)
            a4_width = 3508  # 11.69 inches * 300 DPI
            a4_height = 2480  # 8.27 inches * 300 DPI
            
            # Prepare content information
            content_info = self._prepare_content_for_llm(parsed_data)
            
            # Build quality feedback information
            feedback_info = ""
            if quality_feedback:
                feedback_info = f"""
Quality Feedback Information:
- Current Score: {quality_feedback.get('current_score', 0):.3f}
- Target Score: {quality_feedback.get('target_score', 0.8):.3f}
- Evaluation Suggestions: {', '.join(quality_feedback.get('evaluation_suggestions', []))}
- Quality Issues: {', '.join(quality_feedback.get('quality_issues', []))}
- Optimization Iteration: {quality_feedback.get('iteration', 1)}

Please optimize layout design based on the above feedback, focusing on:
1. Resolve identified quality issues
2. Adopt evaluation suggestions
3. Improve overall quality score
"""
            
            prompt = f"""
Please design a flexible three-column layout poster for the following government document on horizontal A4 paper, requirements:
1. Horizontal A4 paper dimensions: {a4_width}x{a4_height} pixels
2. Decide the number of regions based on the actual content of the document, do not force segmentation
3. Content blocks can be any size: full width, half width, 1/3 width, etc.
4. Content blocks can span columns or occupy only half a column
5. Natural connection from top to bottom, left to right, make full use of space
6. Filter and reorganize information, but the content generated for each section should be complete and smooth
7. Fill actual specific content in the content field, do not use placeholders
8. Title blocks only need to fill the name field, content field is empty

Document Information:
{content_info}

{feedback_info}

Please analyze the document content, create an appropriate number of content regions based on actual needs (usually 3-6 regions are sufficient), and return JSON format layout plan:

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

Important Requirements:
- Decide the number of regions based on the actual content of the document, do not force segmentation
- Ensure all regions cover the entire horizontal A4 paper with no gaps
- Content blocks can span columns, half columns, full width, etc.
- Natural connection from top to bottom, left to right
- Font size should be appropriate for the region size (reference: title 84px, large block 36px, medium block 32px, small block 30px, adjust appropriately based on region size)
- Color scheme should conform to government document style, use different background colors to distinguish content blocks
- Fill actual specific content in the content field, do not use placeholders like "specific content" or "omitted here", title blocks have empty content field
- Filter and reorganize information, but the content generated for each section should be complete and smooth
- Reasonably allocate and combine information to each content block based on document content, place abstracts and important information first
- Region names should reflect actual content, such as "Basic Information", "Processing Flow", "Contact Information", etc.

{quality_feedback}
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8000,
                temperature=0.3
            )
            
            # Parse LLM response
            try:
                response_text = response.choices[0].message.content.strip()
                
                # Try to extract JSON part
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
                
                # Validate and adjust layout
                validated_layout = self._validate_and_adjust_layout(layout_data, a4_width, a4_height)
                
                return validated_layout
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"LLM response JSON parsing failed: {str(e)}")
                # Avoid printing the full LLM response (which may contain entire PDF text) to log, truncate to safe length
                try:
                    raw_resp = response.choices[0].message.content
                    preview = (raw_resp[:500] + '...') if len(raw_resp) > 500 else raw_resp
                    preview = preview.replace('\n', ' ')
                    self.logger.warning(f"Response content (truncated, showing first 500 characters): {preview}")
                except Exception:
                    self.logger.warning("Response content unavailable or too long, omitted from display")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to plan A4 layout with LLM: {str(e)}")
            return None
    
    def _prepare_content_for_llm(self, parsed_data: Dict[str, Any]) -> str:
        """Prepare content information for LLM (no content omitted)"""
        try:
            content_parts = []
            
            # Basic information
            content_parts.append(f"Document Type: {parsed_data.get('rule_type', 'unknown')}")
            content_parts.append(f"Document Title: {parsed_data.get('structured_content', {}).get('title', 'No Title')}")
            
            # Main content (not truncated)
            markdown_content = parsed_data.get('markdown_content', '')
            if markdown_content:
                content_parts.append(f"Document Content: {markdown_content}")
            
            # Key information
            key_info = parsed_data.get('key_info', {})
            if key_info:
                content_parts.append(f"Main Topic: {key_info.get('main_topic', '')}")
                content_parts.append(f"Purpose: {key_info.get('purpose', '')}")
                content_parts.append(f"Target Audience: {key_info.get('target_audience', '')}")
                content_parts.append(f"Responsible Department: {key_info.get('responsible_department', '')}")
            
            # Structured content
            structured = parsed_data.get('structured_content', {})
            if structured.get('key_points'):
                content_parts.append(f"Key Points: {', '.join(structured['key_points'])}")
            
            if structured.get('contact_info'):
                contact = structured['contact_info']
                if contact.get('phones'):
                    content_parts.append(f"Contact Phone: {', '.join(contact['phones'])}")
                if contact.get('emails'):
                    content_parts.append(f"Contact Email: {', '.join(contact['emails'])}")
                if contact.get('addresses'):
                    content_parts.append(f"Contact Address: {', '.join(contact['addresses'])}")
            
            if structured.get('deadlines'):
                content_parts.append(f"Deadlines: {', '.join(structured['deadlines'])}")
            
            if structured.get('requirements'):
                content_parts.append(f"Requirements: {', '.join(structured['requirements'])}")
            
            # Summary
            if parsed_data.get('summary'):
                content_parts.append(f"Summary: {parsed_data['summary']}")
            
            return '\n'.join(content_parts)
            
        except Exception as e:
            self.logger.warning(f"Failed to prepare LLM content: {str(e)}")
            return "Content preparation failed"
    
    def _validate_and_adjust_layout(self, layout_data: Dict[str, Any], canvas_width: int, canvas_height: int) -> Dict[str, Any]:
        """Validate and adjust layout data"""
        try:
            regions = layout_data.get('regions', [])
            
            if not regions:
                return None
            
            # Ensure all regions cover the entire canvas
            total_area = 0
            for region in regions:
                width = region.get('width', 0)
                height = region.get('height', 0)
                total_area += width * height
            
            canvas_area = canvas_width * canvas_height
            
            # If coverage is incomplete, adjust region sizes
            if total_area < canvas_area * 0.9:  # At least 90% coverage
                scale_factor = (canvas_area * 0.95) / total_area
                for region in regions:
                    region['width'] = int(region.get('width', 0) * scale_factor)
                    region['height'] = int(region.get('height', 0) * scale_factor)
            
            # Ensure coordinates and dimensions are within canvas bounds
            for region in regions:
                region['x'] = max(0, min(region.get('x', 0), canvas_width - region.get('width', 0)))
                region['y'] = max(0, min(region.get('y', 0), canvas_height - region.get('height', 0)))
                region['width'] = max(100, min(region.get('width', 100), canvas_width - region.get('x', 0)))
                region['height'] = max(50, min(region.get('height', 50), canvas_height - region.get('y', 0)))
            
            # Add metadata
            layout_data['metadata'] = {
                'canvas_width': canvas_width,
                'canvas_height': canvas_height,
                'total_regions': len(regions),
                'layout_type': 'a4_llm_planned'
            }
            
            return layout_data
            
        except Exception as e:
            self.logger.warning(f"Failed to validate and adjust layout: {str(e)}")
            return None
    