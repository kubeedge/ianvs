# Copyright 2024 New Government Agent Project
# 新版政府代理海报生成基础模型

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

# 导入必要的库
from dotenv import load_dotenv
from sedna.common.class_factory import ClassType, ClassFactory

# 导入自定义工具
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
    新版政府报告转海报代理系统
    
    基于多智能体架构，实现政府报告到可视化海报的自动转换
    包含Parser、Planner、Painter、Evaluator四个核心组件
    支持并行处理和异步执行，提供更好的性能和可扩展性
    """
    
    def __init__(self, **kwargs):
        """
        初始化新版政府代理系统
        
        Args:
            llm_base_url: LLM API基础URL
            vlm_base_url: VLM API基础URL  
            llm_model: LLM模型
            vlm_model: VLM模型
            api_key: API密钥
            max_retries: 最大重试次数
            timeout: 超时时间
            poster_width_inches: 海报宽度（英寸）
            poster_height_inches: 海报高度（英寸）
            max_workers: 最大工作进程数
            enable_parallel_processing: 是否启用并行处理
            enable_quality_optimization: 是否启用质量优化
            enable_government_style_enhancement: 是否启用政府风格增强
            max_optimization_iterations: 最大优化迭代次数
            quality_threshold: 质量阈值
        """
        self.llm_base_url = kwargs.get('llm_base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        self.vlm_base_url = kwargs.get('vlm_base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        self.llm_model = kwargs.get('llm_model', 'qwen-max')
        self.vlm_model = kwargs.get('vlm_model', 'qwen-vl-max')
        self.api_key = kwargs.get('api_key', os.getenv('GOVERNMENT_AGENT_API_KEY'))
        self.max_retries = kwargs.get('max_retries', 3)
        self.timeout = kwargs.get('timeout', 60)
        self.poster_width = kwargs.get('poster_width_inches', 48)
        self.poster_height = kwargs.get('poster_height_inches', 36)
        self.max_workers = kwargs.get('max_workers', 4)
        self.enable_parallel_processing = kwargs.get('enable_parallel_processing', True)
        self.enable_quality_optimization = kwargs.get('enable_quality_optimization', True)
        self.enable_government_style_enhancement = kwargs.get('enable_government_style_enhancement', True)
        self.max_optimization_iterations = kwargs.get('max_optimization_iterations', 3)
        self.quality_threshold = kwargs.get('quality_threshold', 8.0)  # VLM评分阈值，10分制
        
        # 初始化日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 加载超参数配置
        self._load_hyperparameters(kwargs)
        
        # 初始化组件
        self.parser = NewGovernmentDocumentParser(
            llm_base_url=self.llm_base_url,
            api_key=self.api_key,
            model=self.llm_model
        )
        self.planner = NewGovernmentPosterPlanner(
            llm_base_url=self.llm_base_url,
            api_key=self.api_key,
            model=self.llm_model
        )
        self.painter = NewGovernmentPosterPainter(
            width=self.poster_width,
            height=self.poster_height
        )
        self.evaluator = NewGovernmentPosterEvaluator(
            llm_base_url=self.llm_base_url,
            api_key=self.api_key,
            model=self.vlm_model
        )
        
        # 政府文档类型规则（增强版）
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
        
        # 政府风格主题配置（增强版）
        self.government_theme = {
            'panel_visible': True,
            'textbox_visible': False,
            'figure_visible': False,
            'panel_theme': {
                'color': (47, 85, 151),  # 政府蓝
                'thickness': 3,
                'line_style': 'solid',
            },
            'textbox_theme': None,
            'figure_theme': None,
            'title_color': (255, 255, 255),
            'title_fill_color': (47, 85, 151),
            'content_color': (0, 0, 0),
            'font_family': 'Microsoft YaHei',
            'highlight_color': (255, 193, 7),  # 金色强调
            'border_radius': 8,
            'shadow_enabled': True
        }
        
        # 质量优化配置
        self.quality_config = {
            'min_text_size': 12,
            'max_text_size': 48,
            'line_spacing': 1.2,
            'margin_ratio': 0.1,
            'color_contrast_threshold': 4.5,
            'readability_threshold': 0.8
        }
    
    def _load_hyperparameters(self, kwargs):
        """加载超参数配置"""
        try:
            # 检查是否有超参数文件路径
            hyperparameters_file = kwargs.get('hyperparameters_file')
            if hyperparameters_file and os.path.exists(hyperparameters_file):
                import yaml
                with open(hyperparameters_file, 'r', encoding='utf-8') as f:
                    hyperparams = yaml.safe_load(f)
                
                # 更新配置
                for key, value in hyperparams.items():
                    if key not in kwargs:  # 只有在kwargs中没有设置时才使用文件中的值
                        setattr(self, key, value)
                        self.logger.info(f"从配置文件加载超参数: {key} = {value}")
            else:
                self.logger.info("未找到超参数配置文件，使用默认值")
        except Exception as e:
            self.logger.warning(f"加载超参数配置失败: {str(e)}，使用默认值")
    
    def train(self, dataset):
        """
        训练阶段（对于政府代理系统，主要是配置和初始化）
        
        Args:
            dataset: 训练数据集
        """
        pass
    
    
    def save(self, output_dir: str) -> str:
        """
        保存模型配置
        
        Args:
            output_dir: 输出目录
            
        Returns:
            str: 保存路径
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
        
        self.logger.info(f"新版政府代理配置已保存到: {config_path}")
        return config_path
    
    def load(self, model_path: str):
        """
        加载模型配置
        
        Args:
            model_path: 模型路径
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
            
            # 更新海报尺寸
            poster_dims = config.get('poster_dimensions', {})
            self.poster_width = poster_dims.get('width', self.poster_width)
            self.poster_height = poster_dims.get('height', self.poster_height)
            
            # 更新功能开关
            features = config.get('features', {})
            self.enable_parallel_processing = features.get('parallel_processing', self.enable_parallel_processing)
            self.enable_quality_optimization = features.get('quality_optimization', self.enable_quality_optimization)
            self.enable_government_style_enhancement = features.get('government_style_enhancement', self.enable_government_style_enhancement)
            
            # 更新优化配置
            optimization = config.get('optimization', {})
            self.max_optimization_iterations = optimization.get('max_iterations', self.max_optimization_iterations)
            self.quality_threshold = optimization.get('quality_threshold', self.quality_threshold)
            
            self.logger.info(f"新版政府代理配置已加载: {model_path}")
        else:
            self.logger.warning(f"配置文件不存在: {model_path}")
    
    def predict(self, dataset) -> Dict[str, Any]:
        """
        预测阶段：将政府报告转换为海报
        
        Args:
            dataset: 输入数据集，包含PDF文件路径
            
        Returns:
            Dict: 包含生成结果和评估指标的字典
        """
        if self.enable_parallel_processing and len(dataset) > 1:
            return self._predict_parallel(dataset)
        else:
            return self._predict_sequential(dataset)
    
    def _predict_sequential(self, dataset) -> Dict[str, Any]:
        """顺序处理数据集"""
        results = []
        
        for i, data_item in enumerate(dataset):
            try:
                self.logger.info(f"处理第 {i+1} 个文档: {data_item}")
                result = self._process_single_document(data_item, i+1)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"处理第 {i+1} 个文档时出错: {str(e)}")
                results.append({
                    'input_file': data_item,
                    'error': str(e),
                    'processing_time': time.time()
                })
        
        return self._format_results(results)
    
    def _predict_parallel(self, dataset) -> Dict[str, Any]:
        """并行处理数据集"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(self._process_single_document, data_item, i+1): i 
                for i, data_item in enumerate(dataset)
            }
            
            # 收集结果
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(f"第 {index+1} 个文档处理完成")
                except Exception as e:
                    self.logger.error(f"处理第 {index+1} 个文档时出错: {str(e)}")
                    results.append({
                        'input_file': dataset[index],
                        'error': str(e),
                        'processing_time': time.time()
                    })
        
        # 按原始顺序排序结果
        # dataset是numpy数组，需要转换为列表才能使用index方法
        dataset_list = dataset.tolist() if hasattr(dataset, 'tolist') else list(dataset)
        results.sort(key=lambda x: dataset_list.index(x.get('input_file', '')) if x.get('input_file') in dataset_list else 999)
        
        return self._format_results(results)
    
    def _process_single_document(self, data_item, index: int) -> Dict[str, Any]:
        """处理单个文档（支持迭代优化）"""
        start_time = time.time()
        
        # 步骤1: 解析政府文档
        # 处理Ianvs数据集对象
        if hasattr(data_item, 'x'):
            file_path = data_item.x
        elif hasattr(data_item, 'file_path'):
            file_path = data_item.file_path
        elif isinstance(data_item, dict):
            file_path = data_item.get('file_path', data_item)
        elif isinstance(data_item, list) and len(data_item) > 0:
            file_path = data_item[0]
        else:
            file_path = data_item
            
        parsed_data = self.parser.parse_document(file_path)
        
        # 步骤2: 规划海报布局
        layout_plan = self.planner.plan_poster_layout(parsed_data)
        
        # 步骤3: 生成海报
        poster_result = self.painter.generate_poster(
            parsed_data, 
            layout_plan, 
            self.government_theme
        )
        
        # 步骤4: 质量评估和迭代优化
        quality_feedback = None
        evaluation_result = None
        optimization_history = []
        
        if self.enable_quality_optimization:
            # 进行迭代优化
            try:
                optimization_result = self._iterative_optimization(
                    parsed_data, layout_plan, poster_result
                )
                if optimization_result and len(optimization_result) == 4:
                    poster_result, quality_feedback, evaluation_result, optimization_history = optimization_result
                else:
                    self.logger.warning("优化结果格式不正确，使用原始结果")
                    evaluation_result = self.evaluator.evaluate_poster(parsed_data, poster_result)
                    optimization_history = []
            except Exception as e:
                self.logger.warning(f"迭代优化失败: {str(e)}")
                evaluation_result = self.evaluator.evaluate_poster(parsed_data, poster_result)
                optimization_history = []
        
        # 如果还没有评估结果，进行最终评估
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
        迭代优化海报质量
        
        Args:
            parsed_data: 解析后的文档数据
            layout_plan: 布局规划
            poster_result: 海报生成结果
            
        Returns:
            Tuple: (优化后的海报结果, 质量反馈, 评估结果, 优化历史)
        """
        optimization_history = []
        current_poster = poster_result
        current_layout = layout_plan
        
        for iteration in range(self.max_optimization_iterations):
            self.logger.info(f"开始第 {iteration + 1} 轮优化...")
            
            # 使用evaluator进行VLM评估
            evaluation_result = self.evaluator.evaluate_poster(
                parsed_data, 
                current_poster
            )
            
            # 记录优化历史
            iteration_info = {
                "iteration": iteration + 1,
                "vlm_score": evaluation_result.get('score', 0),
                "improvement_suggestions": evaluation_result.get('improvement_suggestions', [])
            }
            optimization_history.append(iteration_info)
            
            # 检查是否达到质量阈值
            overall_score = evaluation_result.get('score', 0)
            if overall_score >= self.quality_threshold:
                self.logger.info(f"第 {iteration + 1} 轮优化完成，质量达标: {overall_score:.3f}")
                break
            
            # 步骤3: 基于VLM反馈优化布局
            if overall_score < self.quality_threshold:
                self.logger.info(f"第 {iteration + 1} 轮优化：VLM评分 {overall_score:.1f} < {self.quality_threshold}，开始优化...")
                
                # 将VLM评估的改进建议传递给planner
                improvement_suggestions = evaluation_result.get('improvement_suggestions', [])
                
                # 构建反馈信息
                combined_feedback = {
                    "evaluation_suggestions": improvement_suggestions,
                    "current_score": overall_score,
                    "target_score": self.quality_threshold,
                    "iteration": iteration + 1
                }
                
                # 使用planner优化布局
                try:
                    optimized_layout = self.planner.plan_poster_layout(
                        parsed_data, 
                        combined_feedback
                    )
                    current_layout = optimized_layout
                    
                    # 重新生成海报
                    current_poster = self.painter.generate_poster(
                        parsed_data,
                        optimized_layout,
                        self.government_theme
                    )
                    
                    self.logger.info(f"第 {iteration + 1} 轮优化：布局调整完成")
                    
                except Exception as e:
                    self.logger.warning(f"第 {iteration + 1} 轮优化：布局调整失败: {str(e)}")
                    break
            else:
                self.logger.info(f"第 {iteration + 1} 轮优化：质量已达标，停止优化")
                break
        
        # 返回最终结果
        final_evaluation = self.evaluator.evaluate_poster(
            parsed_data, 
            current_poster
        )
        
        return current_poster, None, final_evaluation, optimization_history
    
    def _optimize_poster(self, parsed_data: Dict, layout_plan: Dict, poster_result: Dict, quality_feedback: Dict) -> Dict:
        """优化海报质量"""
        try:
            self.logger.info("正在优化海报质量...")
            
            # 基于质量反馈调整布局
            optimized_layout = self.planner.optimize_layout(layout_plan, quality_feedback)
            
            # 重新生成海报
            optimized_poster = self.painter.generate_poster(
                parsed_data,
                optimized_layout,
                self.government_theme
            )
            
            self.logger.info("海报质量优化完成")
            return optimized_poster
            
        except Exception as e:
            self.logger.warning(f"海报优化失败: {str(e)}")
            return poster_result
    
    def _format_results(self, results: List[Dict]) -> Dict[str, Any]:
        """格式化结果"""
        success_count = len([r for r in results if 'error' not in r])
        total_processing_time = sum(r.get('processing_time', 0) for r in results)
        
        # 计算平均评估分数
        evaluation_scores = [r.get('evaluation', {}) for r in results if 'evaluation' in r]
        avg_scores = {}
        if evaluation_scores:
            for key in evaluation_scores[0].keys():
                if isinstance(evaluation_scores[0][key], (int, float)):
                    avg_scores[f'avg_{key}'] = sum(e.get(key, 0) for e in evaluation_scores) / len(evaluation_scores)
        
        # 计算优化统计
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
        """计算优化统计信息"""
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
                        
                        # 计算VLM评分提升
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
            self.logger.warning(f"计算优化统计失败: {str(e)}")
            return {}
    
    def classify_document_type(self, content: str) -> str:
        """
        根据文档内容分类政府文档类型（增强版）
        
        Args:
            content: 文档内容
            
        Returns:
            str: 文档类型
        """
        content_lower = content.lower()
        type_scores = {}
        
        for rule_type, config in self.rule_types.items():
            score = 0
            for keyword in config['keywords']:
                if keyword in content_lower:
                    score += 1
            type_scores[rule_type] = score
        
        # 返回得分最高的类型
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 0:
                return best_type
        
        # 默认分类为通知类型
        return "notices"
    
    def get_evaluation_criteria(self, rule_type: str) -> List[str]:
        """
        获取指定文档类型的评估标准
        
        Args:
            rule_type: 文档类型
            
        Returns:
            List[str]: 评估标准列表
        """
        return self.rule_types.get(rule_type, {}).get('evaluation_criteria', [])
    
    def get_rule_type_priority(self, rule_type: str) -> int:
        """
        获取文档类型优先级
        
        Args:
            rule_type: 文档类型
            
        Returns:
            int: 优先级（数字越小优先级越高）
        """
        return self.rule_types.get(rule_type, {}).get('priority', 999)
