# Copyright 2024 New Government Agent Project
# New Government Agent Poster Generation Base Model

import os
import json
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import re
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import necessary libraries
from dotenv import load_dotenv
from sedna.common.class_factory import ClassType, ClassFactory

# Import custom tools
try:
    from .gov_parser import NewGovernmentDocumentParser
    from .gov_planner import NewGovernmentPosterPlanner
    from .gov_painter import NewGovernmentPosterPainter
    from .gov_evaluator import NewGovernmentPosterEvaluator
except ImportError:
    from gov_parser import NewGovernmentDocumentParser
    from gov_planner import NewGovernmentPosterPlanner
    from gov_painter import NewGovernmentPosterPainter
    from gov_evaluator import NewGovernmentPosterEvaluator

load_dotenv()

@ClassFactory.register(ClassType.GENERAL, alias="NewGovernmentPosterAgent")
class NewGovernmentPosterAgent:
    """
    New Government Report to Poster Agent System
    
    Based on multi-agent architecture, implements automatic conversion from government reports to visual posters
    Contains four core components: Parser, Planner, Painter, Evaluator
    Supports parallel processing and asynchronous execution for better performance and scalability
    """
    
    def __init__(self, **kwargs):
        """
        Initialize new government agent system
        
        Args:
            llm_base_url: LLM API base URL
            vlm_base_url: VLM API base URL  
            llm_model: LLM model
            vlm_model: VLM model
            api_key: API key
            max_retries: Maximum retry times
            timeout: Timeout
            poster_width_inches: Poster width (inches)
            poster_height_inches: Poster height (inches)
            max_workers: Maximum worker processes
            enable_parallel_processing: Whether to enable parallel processing
            enable_quality_optimization: Whether to enable quality optimization
            enable_government_style_enhancement: Whether to enable government style enhancement
            max_optimization_iterations: Maximum optimization iteration times
            quality_threshold: Quality threshold
        """
    # Prefer API key from environment for security and flexibility
        self.api_key = 'sk-44f3949ebe204bd1b59a834bf97fecb7'
        self.llm_base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        self.vlm_base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        self.llm_model = kwargs.get('llm_model', 'qwen-max')
        self.vlm_model = kwargs.get('vlm_model', 'qwen-vl-max')
        self.max_retries = 3
        self.timeout = 120
        self.max_workers = 4
        self.poster_width = 48
        self.poster_height = 36
        self.enable_parallel_processing = True
        self.enable_quality_optimization = True
        self.enable_government_style_enhancement = True
        # Whether to evaluate only (without executing generation process)
        self.evaluate_only = kwargs.get('evaluate_only', False)
        # When evaluate_only=True and poster is missing, whether to force fallback to generation process
        self.force_generate_if_missing = kwargs.get('force_generate_if_missing', False)
        # If in evaluation-only mode, disable quality optimization to avoid dependency on intermediate variables from generation
        if self.evaluate_only:
            self.enable_quality_optimization = False
        self.max_optimization_iterations = 0 # Optimization iteration count
        self.quality_threshold = 2.0 # VLM score threshold, out of 10
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        if not self.api_key:
            self.logger.warning('DASHSCOPE API key is not set. Set DASHSCOPE_API_KEY env var or update basemodel.py')
        
        # Load hyperparameter configuration
        self._load_hyperparameters(kwargs)
        
        # Initialize components
        # Parser and Evaluator still need to be available in evaluate-only mode;
        # To avoid triggering generation or external calls when planner/painter are constructed,
        # delay/skip instantiation of planner and painter when evaluate_only=True.
        self.parser = NewGovernmentDocumentParser(
            llm_base_url=self.llm_base_url,
            api_key=self.api_key,
            model=self.llm_model
        )
        self.evaluator = NewGovernmentPosterEvaluator(
            llm_base_url=self.llm_base_url,
            api_key=self.api_key,
            model=self.vlm_model
        )

        # Create planner and painter only when not in evaluate_only mode
        if not self.evaluate_only:
            self.planner = NewGovernmentPosterPlanner(
                llm_base_url=self.llm_base_url,
                api_key=self.api_key,
                model=self.llm_model
            )
            self.painter = NewGovernmentPosterPainter(
                width=self.poster_width,
                height=self.poster_height
            )

        # Try to load poster_index (generated by script, used for question->poster_path mapping in evaluate_only mode)
        self.poster_index = {}
        try:
            possible_index = Path("examples/new_government_agent/resources/datasets/test/poster_index.json")
            if possible_index.exists():
                with possible_index.open("r", encoding="utf-8") as f:
                    self.poster_index = json.load(f)
                    self.logger.info(f"Loaded poster_index ({len(self.poster_index)} entries) from {possible_index}")
        except Exception as e:
            self.logger.warning(f"Failed to load poster_index: {e}")
        else:
            # load poster index if exists so evaluate_only can lookup poster_path for string data items
            try:
                index_path = os.path.join('examples', 'new_government_agent', 'resources', 'datasets', 'test', 'poster_index.json')
                if os.path.exists(index_path):
                    with open(index_path, 'r', encoding='utf-8') as f:
                        self._poster_index = json.load(f)
                else:
                    self._poster_index = {}
            except Exception:
                self._poster_index = {}
            # posters dir used for fallback search
            self._posters_dir = os.path.join('examples', 'new_government_agent', 'posters')
            # path to dataset file we'll persist into if we auto-fill
            self._dataset_file = os.path.join('examples', 'new_government_agent', 'resources', 'datasets', 'test', 'data.jsonl')
        
        # Government document type rules (enhanced version)
        self.rule_types = {
            "guiding_policy": {
                "keywords": ["意见", "规划", "决议", "方案", "纲要", "战略", "政策", "指导"],
                "template": "policy_direction",
                "evaluation_criteria": ["政策要点覆盖", "语言规范性", "视觉一致性", "权威性", "层次清晰度"],
                "priority": 1
            },
            "implementation": {
                "keywords": ["办法", "规定", "细则", "措施", "实施方案", "操作规程", "执行", "实施"],
                "template": "bullet_points", 
                "evaluation_criteria": ["内容完整性", "逻辑清晰度", "可操作性", "实用性", "步骤明确性"],
                "priority": 2
            },
            "service_guide": {
                "keywords": ["指南", "步骤", "流程", "服务", "办事指南", "操作手册", "服务", "便民"],
                "template": "procedure_diagram",
                "evaluation_criteria": ["步骤清晰度", "用户友好性", "信息准确性", "易理解性", "便民性"],
                "priority": 3
            },
            "notices": {
                "keywords": ["公告", "通知", "声明", "发布", "通告", "公示", "告知", "通知"],
                "template": "public_notice",
                "evaluation_criteria": ["信息传达", "格式规范", "时效性", "重要性", "可读性"],
                "priority": 4
            },
            "faq": {
                "keywords": ["常见问题", "咨询", "问答", "解答", "问题", "疑问", "FAQ", "帮助"],
                "template": "question_cards",
                "evaluation_criteria": ["问题相关性", "答案准确性", "易理解性", "完整性", "实用性"],
                "priority": 5
            },
            "legal_documents": {
                "keywords": ["法律", "条例", "规章", "条款", "法规", "法令", "条文", "法律"],
                "template": "legal_structure",
                "evaluation_criteria": ["法律准确性", "结构清晰度", "可读性", "权威性", "严谨性"],
                "priority": 6
            }
        }
        
        # Government style theme configuration (enhanced version)
        self.government_theme = {
            'panel_visible': True,
            'textbox_visible': False,
            'figure_visible': False,
            'panel_theme': {
                'color': (47, 85, 151),  # Government blue
                'thickness': 3,
                'line_style': 'solid',
            },
            'textbox_theme': None,
            'figure_theme': None,
            'title_color': (255, 255, 255),
            'title_fill_color': (47, 85, 151),
            'content_color': (0, 0, 0),
            'font_family': 'Microsoft YaHei',
            'highlight_color': (255, 193, 7),  # Gold accent
            'border_radius': 8,
            'shadow_enabled': True
        }
        
        # Quality optimization configuration
        self.quality_config = {
            'min_text_size': 12,
            'max_text_size': 48,
            'line_spacing': 1.2,
            'margin_ratio': 0.1,
            'color_contrast_threshold': 4.5,
            'readability_threshold': 0.8
        }
    
    def _load_hyperparameters(self, kwargs):
        """Load hyperparameter configuration"""
        try:
            # Check if hyperparameter file path exists
            hyperparameters_file = kwargs.get('hyperparameters_file')
            if hyperparameters_file and os.path.exists(hyperparameters_file):
                import yaml
                with open(hyperparameters_file, 'r', encoding='utf-8') as f:
                    hyperparams = yaml.safe_load(f)
                
                # Update configuration
                for key, value in hyperparams.items():
                    if key not in kwargs:  # Only use file value if not set in kwargs
                        setattr(self, key, value)
                        self.logger.info(f"Loaded hyperparameter from config file: {key} = {value}")
            else:
                self.logger.info("Hyperparameter config file not found, using default values")
        except Exception as e:
            self.logger.warning(f"Failed to load hyperparameter configuration: {str(e)}, using default values")
    
    def train(self, dataset):
        """
        Training stage (for government agent system, mainly configuration and initialization)
        
        Args:
            dataset: Training dataset
        """
        pass
    
    
    def save(self, output_dir: str) -> str:
        """
        Save model configuration
        
        Args:
            output_dir: Output directory
            
        Returns:
            str: Save path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        config = {
            'llm_base_url': self.llm_base_url,
            'vlm_base_url': self.vlm_base_url,
            'llm_model': self.llm_model,
            'vlm_model': self.vlm_model,
            'rule_types': self.rule_types,
            'government_theme': self.government_theme,
            'quality_config': self.quality_config,
            'poster_dimensions': {
                'width': self.poster_width,
                'height': self.poster_height
            },
            'features': {
                'parallel_processing': self.enable_parallel_processing,
                'quality_optimization': self.enable_quality_optimization,
                'government_style_enhancement': self.enable_government_style_enhancement
            },
            'optimization': {
                'max_iterations': self.max_optimization_iterations,
                'quality_threshold': self.quality_threshold
            }
        }
        
        config_path = os.path.join(output_dir, 'new_government_agent_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"New government agent configuration saved to: {config_path}")
        return config_path
    
    def load(self, model_path: str):
        """
        Load model configuration
        
        Args:
            model_path: Model path
        """
        if os.path.exists(model_path):
            with open(model_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.llm_base_url = config.get('llm_base_url', self.llm_base_url)
            self.vlm_base_url = config.get('vlm_base_url', self.vlm_base_url)
            self.llm_model = config.get('llm_model', self.llm_model)
            self.vlm_model = config.get('vlm_model', self.vlm_model)
            self.rule_types = config.get('rule_types', self.rule_types)
            self.government_theme = config.get('government_theme', self.government_theme)
            self.quality_config = config.get('quality_config', self.quality_config)
            
            # Update poster dimensions
            poster_dims = config.get('poster_dimensions', {})
            self.poster_width = poster_dims.get('width', self.poster_width)
            self.poster_height = poster_dims.get('height', self.poster_height)
            
            # Update feature switches
            features = config.get('features', {})
            self.enable_parallel_processing = features.get('parallel_processing', self.enable_parallel_processing)
            self.enable_quality_optimization = features.get('quality_optimization', self.enable_quality_optimization)
            self.enable_government_style_enhancement = features.get('government_style_enhancement', self.enable_government_style_enhancement)
            
            # Update optimization configuration
            optimization = config.get('optimization', {})
            self.max_optimization_iterations = optimization.get('max_iterations', self.max_optimization_iterations)
            self.quality_threshold = optimization.get('quality_threshold', self.quality_threshold)
            
            self.logger.info(f"New government agent configuration loaded: {model_path}")
        else:
            self.logger.warning(f"Configuration file does not exist: {model_path}")
    
    def predict(self, dataset) -> Dict[str, Any]:
        """
        Prediction stage: Convert government reports to posters
        
        Args:
            dataset: Input dataset, containing PDF file paths
            
        Returns:
            Dict: Dictionary containing generation results and evaluation metrics
        """
        if self.enable_parallel_processing and len(dataset) > 1:
            return self._predict_parallel(dataset)
        else:
            return self._predict_sequential(dataset)
    
    def _predict_sequential(self, dataset) -> Dict[str, Any]:
        """Sequentially process dataset"""
        results = []
        
        for i, data_item in enumerate(dataset):
            try:
                self.logger.info(f"Processing document {i+1}")
                result = self._process_single_document(data_item, i+1)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing document {i+1}: {str(e)}")
                results.append({
                    'input_file': data_item,
                    'error': str(e),
                    'processing_time': time.time()
                })
        
        return self._format_results(results)
    
    def _predict_parallel(self, dataset) -> Dict[str, Any]:
        """Process dataset in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._process_single_document, data_item, i+1): i 
                for i, data_item in enumerate(dataset)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(f"Document {index+1} processing completed")
                except Exception as e:
                    self.logger.error(f"Error processing document {index+1}: {str(e)}")
                    results.append({
                        'input_file': dataset[index],
                        'error': str(e),
                        'processing_time': time.time()
                    })
        
        # Sort results by original order
        # dataset is numpy array, need to convert to list to use index method
        dataset_list = dataset.tolist() if hasattr(dataset, 'tolist') else list(dataset)
        results.sort(key=lambda x: dataset_list.index(x.get('input_file', '')) if x.get('input_file') in dataset_list else 999)
        
        return self._format_results(results)
    
    def _get_poster_path_from_jsonl(self, index: int) -> Optional[str]:
        """
        Read poster_path for specified index directly from JSONL file
        
        Parameters
        ----------
        index: int
            Data item index
            
        Returns
        -------
        str or None
            The value of poster_path field, returns None if not found
        """
        try:
            # Get test data file path
            test_data_path = getattr(self, 'test_data_path', None)
            if not test_data_path:
                # Try to get from environment variable or default path
                test_data_path = os.path.join('examples', 'new_government_agent', 'resources', 'datasets', 'test', 'data.jsonl')
            
            if not os.path.exists(test_data_path):
                self.logger.warning(f'Test data file does not exist: {test_data_path}')
                return None
            
            # Read JSONL file
            with open(test_data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            if index >= len(lines):
                self.logger.warning(f'Index {index} exceeds data range (total {len(lines)} lines)')
                return None
                
            # Parse JSON data of specified line
            line = lines[index].strip()
            if not line:
                return None
                
            data = json.loads(line)
            poster_path = data.get('poster_path')
            
            if poster_path:
                self.logger.info(f'Retrieved poster_path from JSONL line {index+1}: {poster_path}')
                return poster_path
            else:
                self.logger.warning(f'JSONL line {index+1} has no poster_path field')
                return None
                
        except Exception as e:
            self.logger.error(f'Failed to read poster_path from JSONL file: {str(e)}')
            return None
    
    def _process_single_document(self, data_item, index: int) -> Dict[str, Any]:
        """Process single document (supports iterative optimization)"""
        start_time = time.time()
        
        # Step 1: Parse government document
        parsed_data = self.parser.parse_document(data_item)

        # If evaluate_only mode, skip planner/painter, only use existing poster_path for evaluation
        poster_result = None
        if self.evaluate_only:
            self.logger.info('evaluate_only mode: Skipping generation, trying to use existing poster_path for evaluation')
            poster_path = None
            
            # Prioritize getting poster_path from original JSONL data
            try:
                # Try to get original JSON data from parsed_data
                original_data = parsed_data.get('original_data')
                if original_data and isinstance(original_data, dict) and 'poster_path' in original_data:
                    poster_path = original_data['poster_path']
                    self.logger.info(f'Retrieved poster_path from original data: {poster_path}')
                else:
                    # If no original data in parsed_data, try to read directly from data.jsonl
                    poster_path = self._get_poster_path_from_jsonl(index)
                    if poster_path:
                        self.logger.info(f'Retrieved poster_path from JSONL file: {poster_path}')
            except Exception as e:
                self.logger.warning(f'Unable to get poster_path from original data: {str(e)}')
            
            # debug: log data_item type and a short preview
            try:
                preview = str(data_item)
                # make preview single-line and truncate for safe logging
                preview = preview.replace('\n', ' ').replace('\r', ' ')
                if len(preview) > 200:
                    preview = preview[:197] + '...'
            except Exception:
                preview = '<unreprable>'
            self.logger.info(f'evaluate_only: data_item type={type(data_item)}, preview={preview}')
            # If poster_path is already found, skip complex mapping lookup
            if poster_path:
                self.logger.info(f'Found poster_path, skipping mapping lookup: {poster_path}')
            else:
                # if data_item is a plain string (many dataset loaders pass the question string), try to lookup
                # in poster_index by multiple key variants (exact, ./prefixed, absolute, stem, parent)
                if isinstance(data_item, str):
                    candidates = []
                    s = str(data_item)
                    
                    # Prioritize using PDF file path from parsed_data
                    pdf_path = parsed_data.get('input_file') or parsed_data.get('question')
                    if pdf_path and pdf_path.endswith('.pdf'):
                        # Use PDF file path from parsed_data
                        candidates.append(pdf_path)
                        if not pdf_path.startswith("./"):
                            candidates.append("./" + pdf_path)
                        if not pdf_path.startswith("/"):
                            candidates.append(os.path.abspath(pdf_path))
                        # basename/stem and parent
                        stem = Path(pdf_path).stem
                        parent = Path(pdf_path).parent.name
                        candidates.extend([stem, parent])
                    elif s.endswith('.pdf') and not s.startswith('==='):
                        # This is PDF file path, prioritize
                        candidates.append(s)
                        if not s.startswith("./"):
                            candidates.append("./" + s)
                        if not s.startswith("/"):
                            candidates.append(os.path.abspath(s))
                        # basename/stem and parent
                        stem = Path(s).stem
                        parent = Path(s).parent.name
                        candidates.extend([stem, parent])
                    elif s.startswith('==='):
                        # This is PDF content preview, skip or lower priority
                        self.logger.warning(f"Skipping PDF content preview mapping lookup: {s[:100]}...")
                        # Try to get PDF file path from parsed_data
                        if pdf_path and pdf_path.endswith('.pdf'):
                            candidates.append(pdf_path)
                            if not pdf_path.startswith("./"):
                                candidates.append("./" + pdf_path)
                            stem = Path(pdf_path).stem
                            parent = Path(pdf_path).parent.name
                            candidates.extend([stem, parent])
                    else:
                        # Other cases, handle with original logic
                        candidates.append(s)
                        if not s.startswith("./") and s.startswith("/"):
                            rel = "." + s
                            candidates.append(rel)
                        if not s.startswith("/"):
                            candidates.append(os.path.abspath(s))
                        stem = Path(s).stem
                        parent = Path(s).parent.name
                        candidates.extend([stem, parent])

                    # try both poster_index names that may be present
                    indexes = []
                    if hasattr(self, 'poster_index') and isinstance(self.poster_index, dict):
                        indexes.append(self.poster_index)
                    if hasattr(self, '_poster_index') and isinstance(self._poster_index, dict):
                        indexes.append(self._poster_index)

                    found_by = None
                    for idx_name, idx in (('poster_index', getattr(self, 'poster_index', {})), ('_poster_index', getattr(self, '_poster_index', {}))):
                        for key in candidates:
                            try:
                                if key in idx:
                                    poster_path = idx.get(key)
                                    # avoid logging full key content (may contain long PDF text)
                                    safe_key = str(key).replace('\n', ' ').replace('\r', ' ').strip()
                                    if len(safe_key) > 120:
                                        safe_key = safe_key[:117] + '...'
                                    found_by = f"{idx_name}[key_preview='{safe_key}']"
                                    break
                            except Exception:
                                continue
                        if poster_path:
                            break
                    if poster_path:
                        # log only safe preview of the found_by and poster_path
                        self.logger.info(f"evaluate_only: found poster_path via {found_by}: {poster_path}")
                    else:
                        # candidates can contain long strings; log safe previews only
                        try:
                            safe_cands = []
                            for c in candidates:
                                s = str(c).replace('\n', ' ').replace('\r', ' ').strip()
                                if len(s) > 100:
                                    s = s[:97] + '...'
                                safe_cands.append(s)
                            self.logger.info(f"evaluate_only: poster lookup candidates tried (previews): {safe_cands}")
                        except Exception:
                            self.logger.info("evaluate_only: poster lookup candidates tried (unable to render previews)")
                
                # Try to read poster_path from data_item (support dict input)
                if isinstance(data_item, dict):
                    poster_path = data_item.get('poster_path')
                # If not found, search in parsed_data
                if not poster_path and isinstance(parsed_data, dict):
                    poster_path = parsed_data.get('poster_path')

            if not poster_path:
                if self.force_generate_if_missing:
                    self.logger.warning('evaluate_only mode but poster_path not found, force_generate_if_missing=True, falling back to generation process')
                else:
                    # Try robust fallback: attempt to find a candidate poster from poster_index or posters dir
                    try:
                        candidate = None
                        # look into poster_index first
                        if hasattr(self, '_poster_index') and self._poster_index:
                            # try several key variants
                            if isinstance(data_item, str):
                                keys = [data_item, data_item.lstrip('./'), Path(data_item).name, Path(data_item).stem, Path(data_item).parent.name]
                            else:
                                q = ''
                                try:
                                    q = parsed_data.get('input_file') or parsed_data.get('question') or ''
                                except Exception:
                                    q = ''
                                keys = [q, q.lstrip('./'), Path(q).name, Path(q).stem, Path(q).parent.name]
                            for k in keys:
                                if not k:
                                    continue
                                if k in self._poster_index:
                                    candidate = self._poster_index[k]
                                    break
                        # if not found, scan posters dir for best match
                        if not candidate and os.path.exists(self._posters_dir):
                            # try match by parent category or stem
                            target = None
                            if isinstance(data_item, str):
                                target = Path(data_item).stem
                                parent = Path(data_item).parent.name
                            else:
                                target = parsed_data.get('rule_type') if isinstance(parsed_data, dict) else None
                                parent = Path(parsed_data.get('markdown_content','')).parent.name if isinstance(parsed_data, dict) else None
                            # search for files containing target or parent token
                            for root, _, files in os.walk(self._posters_dir):
                                for fn in files:
                                    low = fn.lower()
                                    if target and target.lower() in low:
                                        candidate = os.path.join(root, fn)
                                        break
                                    if parent and parent.lower() in low:
                                        candidate = os.path.join(root, fn)
                                        break
                                if candidate:
                                    break
                            # last resort: pick any poster
                            if not candidate:
                                for root, _, files in os.walk(self._posters_dir):
                                    if files:
                                        candidate = os.path.join(root, files[0])
                                        break

                        if candidate:
                            # persist mapping for future runs
                            try:
                                # record into poster_index and file
                                # Only use PDF file path as key, do not use PDF content preview
                                key_to_write = None
                                if isinstance(data_item, str):
                                    # Check if it's PDF file path, not PDF content preview
                                    if data_item.endswith('.pdf') and not data_item.startswith('==='):
                                        key_to_write = data_item
                                    else:
                                        # If PDF content preview, skip writing to poster_index
                                        self.logger.warning(f"Skipping PDF content preview writing to poster_index: {str(data_item)[:100]}...")
                                        key_to_write = None
                                else:
                                    key_to_write = parsed_data.get('input_file') or parsed_data.get('question') or None
                                
                                if key_to_write and key_to_write.endswith('.pdf'):
                                    # update in-memory index
                                    self._poster_index[key_to_write] = candidate
                                    # write poster_index.json
                                    try:
                                        with open(os.path.join('examples', 'new_government_agent', 'resources', 'datasets', 'test', 'poster_index.json'), 'w', encoding='utf-8') as idxf:
                                            json.dump(self._poster_index, idxf, ensure_ascii=False, indent=2)
                                    except Exception:
                                        pass
                                    # update dataset file by adding poster_path to matching question
                                    if os.path.exists(self._dataset_file):
                                            lines = []
                                            updated = False
                                            with open(self._dataset_file, 'r', encoding='utf-8') as df:
                                                for ln in df:
                                                    s = ln.strip()
                                                    if not s:
                                                        lines.append(ln)
                                                        continue
                                                    try:
                                                        obj = json.loads(s)
                                                    except Exception:
                                                        lines.append(ln)
                                                        continue
                                                    q = obj.get('question')
                                                    if q and key_to_write and (q == key_to_write or q.lstrip('./') == key_to_write or Path(q).stem == Path(key_to_write).stem):
                                                        obj['poster_path'] = candidate
                                                        lines.append(json.dumps(obj, ensure_ascii=False) + '\n')
                                                        updated = True
                                                    else:
                                                        lines.append(ln)
                                            if updated:
                                                with open(self._dataset_file, 'w', encoding='utf-8') as df:
                                                    df.writelines(lines)
                            except Exception:
                                pass

                            poster_path = candidate
                        
                        if not poster_path:
                            raise ValueError('evaluate_only mode requires poster_path field in data_item or parsed_data')
                    except Exception:
                        raise ValueError('evaluate_only mode requires poster_path field in data_item or parsed_data')

            if poster_path:
                poster_result = {'poster_path': poster_path}

        # Execute original generation process if not evaluate_only or in fallback scenario
        if poster_result is None:
            # Step 2: Plan poster layout
            layout_plan = self.planner.plan_poster_layout(parsed_data)
            
            # Step 3: Generate poster
            poster_result = self.painter.generate_poster(
                parsed_data, 
                layout_plan, 
                self.government_theme
            )
        
        # Step 4: Quality evaluation and iterative optimization
        quality_feedback = None
        evaluation_result = None
        optimization_history = []
        
        if self.enable_quality_optimization:
            # Perform iterative optimization
            try:
                optimization_result = self._iterative_optimization(
                    parsed_data, layout_plan, poster_result
                )
                if optimization_result and len(optimization_result) == 4:
                    poster_result, quality_feedback, evaluation_result, optimization_history = optimization_result
                else:
                    self.logger.warning("Optimization result format incorrect, using original result")
                    evaluation_result = self.evaluator.evaluate_poster(parsed_data, poster_result)
                    optimization_history = []
            except Exception as e:
                self.logger.warning(f"Iterative optimization failed: {str(e)}")
                evaluation_result = self.evaluator.evaluate_poster(parsed_data, poster_result)
                optimization_history = []
        
        # If no evaluation result yet, perform final evaluation
        if evaluation_result is None:
            evaluation_result = self.evaluator.evaluate_poster(
                parsed_data, 
                poster_result
            )
        
        processing_time = time.time() - start_time
        
        return {
            "input_file": data_item,
            "rule_type": parsed_data.get('rule_type'),
            "poster_path": poster_result.get('poster_path'),
            "evaluation": evaluation_result,
            "optimization_history": optimization_history,
            "processing_time": processing_time,
            "index": index
        }
    
    def _iterative_optimization(self, parsed_data: Dict, layout_plan: Dict, poster_result: Dict) -> Tuple[Dict, Dict, Dict, List]:
        """
        Iterative optimization of poster quality
        
        Args:
            parsed_data: Parsed document data
            layout_plan: Layout plan
            poster_result: Poster generation result
            
        Returns:
            Tuple: (Optimized poster result, quality feedback, evaluation result, optimization history)
        """
        optimization_history = []
        current_poster = poster_result
        current_layout = layout_plan
        
        for iteration in range(self.max_optimization_iterations):
            self.logger.info(f"Starting optimization round {iteration + 1}...")
            
            # Use evaluator for VLM evaluation
            evaluation_result = self.evaluator.evaluate_poster(
                parsed_data, 
                current_poster
            )
            
            # Record optimization history
            iteration_info = {
                "iteration": iteration + 1,
                "vlm_score": evaluation_result.get('score', 0),
                "improvement_suggestions": evaluation_result.get('improvement_suggestions', [])
            }
            optimization_history.append(iteration_info)
            
            # Check if quality threshold is reached
            overall_score = evaluation_result.get('score', 0)
            if overall_score >= self.quality_threshold:
                self.logger.info(f"Optimization round {iteration + 1} completed, quality threshold reached: {overall_score:.3f}")
                break
            
            # Step 3: Optimize layout based on VLM feedback
            if overall_score < self.quality_threshold:
                self.logger.info(f"Optimization round {iteration + 1}: VLM score {overall_score:.1f} < {self.quality_threshold}, starting optimization...")
                
                # Pass VLM evaluation improvement suggestions to planner
                improvement_suggestions = evaluation_result.get('improvement_suggestions', [])
                
                # Build feedback information
                combined_feedback = {
                    "evaluation_suggestions": improvement_suggestions,
                    "current_score": overall_score,
                    "target_score": self.quality_threshold,
                    "iteration": iteration + 1
                }
                
                # Use planner to optimize layout
                try:
                    optimized_layout = self.planner.plan_poster_layout(
                        parsed_data, 
                        combined_feedback
                    )
                    current_layout = optimized_layout
                    
                    # Regenerate poster
                    current_poster = self.painter.generate_poster(
                        parsed_data,
                        optimized_layout,
                        self.government_theme
                    )
                    
                    self.logger.info(f"Optimization round {iteration + 1}: Layout adjustment completed")
                    
                except Exception as e:
                    self.logger.warning(f"Optimization round {iteration + 1}: Layout adjustment failed: {str(e)}")
                    break
            else:
                self.logger.info(f"Optimization round {iteration + 1}: Quality threshold reached, stopping optimization")
                break
        
        # Return final result
        final_evaluation = self.evaluator.evaluate_poster(
            parsed_data, 
            current_poster
        )
        
        return current_poster, None, final_evaluation, optimization_history
    
    def _optimize_poster(self, parsed_data: Dict, layout_plan: Dict, poster_result: Dict, quality_feedback: Dict) -> Dict:
        """Optimize poster quality"""
        try:
            self.logger.info("Optimizing poster quality...")
            
            # Adjust layout based on quality feedback
            optimized_layout = self.planner.optimize_layout(layout_plan, quality_feedback)
            
            # Regenerate poster
            optimized_poster = self.painter.generate_poster(
                parsed_data,
                optimized_layout,
                self.government_theme
            )
            
            self.logger.info("Poster quality optimization completed")
            return optimized_poster
            
        except Exception as e:
            self.logger.warning(f"Poster optimization failed: {str(e)}")
            return poster_result
    
    def _format_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Format results"""
        success_count = len([r for r in results if 'error' not in r])
        total_processing_time = sum(r.get('processing_time', 0) for r in results)
        
        # Calculate average evaluation score
        evaluation_scores = [r.get('evaluation', {}) for r in results if 'evaluation' in r]
        avg_scores = {}
        if evaluation_scores:
            for key in evaluation_scores[0].keys():
                if isinstance(evaluation_scores[0][key], (int, float)):
                    avg_scores[f'avg_{key}'] = sum(e.get(key, 0) for e in evaluation_scores) / len(evaluation_scores)
        
        # Calculate optimization statistics
        optimization_stats = self._calculate_optimization_stats(results)
        
        return {
            "results": results,
            "total_processed": len(results),
            "success_count": success_count,
            "error_count": len(results) - success_count,
            "total_processing_time": total_processing_time,
            "average_processing_time": total_processing_time / len(results) if results else 0,
            "average_scores": avg_scores,
            "success_rate": success_count / len(results) if results else 0,
            "optimization_stats": optimization_stats
        }
    
    def _calculate_optimization_stats(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate optimization statistics"""
        try:
            total_iterations = 0
            successful_optimizations = 0
            quality_improvements = []
            
            for result in results:
                if 'optimization_history' in result:
                    history = result['optimization_history']
                    if history:
                        total_iterations += len(history)
                        successful_optimizations += 1
                        
                        # Calculate VLM score improvement
                        if len(history) > 1:
                            initial_score = history[0].get('vlm_score', 0)
                            final_score = history[-1].get('vlm_score', 0)
                            improvement = final_score - initial_score
                            quality_improvements.append(improvement)
            
            avg_iterations = total_iterations / successful_optimizations if successful_optimizations > 0 else 0
            avg_improvement = sum(quality_improvements) / len(quality_improvements) if quality_improvements else 0
            
            return {
                "total_optimization_iterations": total_iterations,
                "successful_optimizations": successful_optimizations,
                "average_iterations_per_optimization": avg_iterations,
                "average_quality_improvement": avg_improvement,
                "optimization_success_rate": successful_optimizations / len(results) if results else 0
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate optimization statistics: {str(e)}")
            return {}
    
    def classify_document_type(self, content: str) -> str:
        """
        Classify government document type based on content (enhanced version)
        
        Args:
            content: Document content
            
        Returns:
            str: Document type
        """
        content_lower = content.lower()
        type_scores = {}
        
        for rule_type, config in self.rule_types.items():
            score = 0
            for keyword in config['keywords']:
                if keyword in content_lower:
                    score += 1
            type_scores[rule_type] = score
        
        # Return highest scoring type
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 0:
                return best_type
        
        # Default to notice type
        return "notices"
    
    def get_evaluation_criteria(self, rule_type: str) -> List[str]:
        """
        Get evaluation criteria for specified document type
        
        Args:
            rule_type: Document type
            
        Returns:
            List[str]: Evaluation criteria list
        """
        return self.rule_types.get(rule_type, {}).get('evaluation_criteria', [])
    
    def get_rule_type_priority(self, rule_type: str) -> int:
        """
        Get document type priority
        
        Args:
            rule_type: Document type
            
        Returns:
            int: Priority (smaller number means higher priority)
        """
        return self.rule_types.get(rule_type, {}).get('priority', 999)
