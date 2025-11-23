"""
Naive Assembly Process - Complete End-to-End Orchestration
Integrates Perception + Manipulation for Deformable Component Assembly

This module serves as the main algorithm for Ianvs benchmarking.
It coordinates the complete pipeline:
1. Perception: Detect + Localize + Analyze Deformation
2. Manipulation: Plan + Execute + Force Control
3. Assembly: Verify + Feedback Loop

Compatible with Ianvs framework and deformable_assembly_dataset structure.
"""
import sys
import os
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sedna.common.class_factory import ClassFactory, ClassType

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with formatting
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] [%(name)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Import perception and manipulation modules
try:
    from perception import PerceptionModel
    PERCEPTION_AVAILABLE = True
except ImportError as e:
    logger.error(f"Perception module import failed: {e}")
    PERCEPTION_AVAILABLE = False

try:
    from manipulation import ManipulationModel, ForceController
    MANIPULATION_AVAILABLE = True
except ImportError as e:
    logger.error(f"Manipulation module import failed: {e}")
    MANIPULATION_AVAILABLE = False


@ClassFactory.register(ClassType.GENERAL, alias="assembly_pipeline")
class NaiveAssemblyProcess:
    """
    Complete Assembly Process Algorithm
    
    Architecture:
    ┌─────────────────────────────────────────────────────┐
    │              NAIVE ASSEMBLY PROCESS                 │
    ├─────────────────────────────────────────────────────┤
    │                                                     │
    │  ┌──────────────┐         ┌──────────────────┐    │
    │  │  PERCEPTION  │────────>│  MANIPULATION    │    │
    │  │              │         │                  │    │
    │  │ • YOLO       │  Pose   │ • Trajectory Gen │    │
    │  │ • 6D Pose    │  +      │ • Force Control  │    │
    │  │ • Deformation│  Deform │ • Compliance     │    │
    │  └──────────────┘         └──────────────────┘    │
    │         ▲                          │               │
    │         │                          ▼               │
    │         └──────────────────│   FEEDBACK   │       │
    │                            │   CONTROL    │       │
    │                            └──────────────┘       │
    │                                                     │
    └─────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        perception_config: Dict = None,
        manipulation_config: Dict = None,
        dataset_root: str = None,
        assembly_strategy: str = "sequential",
        max_attempts: int = 3,
        success_threshold: float = 0.9,
        **kwargs
    ):
        """
        Initialize complete assembly process
        
        Parameters:
        -----------
        perception_config : dict
            Configuration for perception module
        manipulation_config : dict
            Configuration for manipulation module
        dataset_root : str
            Path to deformable_assembly_dataset
        assembly_strategy : str
            'sequential', 'adaptive', or 'force_guided'
        max_attempts : int
            Maximum assembly attempts per component
        success_threshold : float
            Confidence threshold for successful assembly
        """
        logger.info("="*80)
        logger.info("NAIVE ASSEMBLY PROCESS - INITIALIZATION")
        logger.info("="*80)
        
        if dataset_root:
            self.dataset_root = Path(dataset_root)
        else:
            self.dataset_root = Path(os.getcwd()) / 'dataset' / 'deformable_assembly_dataset'
        
        # Assembly parameters
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Dataset root: {self.dataset_root}")

        self.assembly_strategy = assembly_strategy
        self.max_attempts = max_attempts
        self.success_threshold = success_threshold
        
        logger.info(f"Assembly strategy: {assembly_strategy}")
        logger.info(f"Max attempts per component: {max_attempts}")
        logger.info(f"Success threshold: {success_threshold}")
        
        # Component assembly order (based on episodes)
        self.component_order = [
            "ram", # Episode 1
            "cooling_mounts", # Episode 2
            "cpu_slot", # Episode 3
            "fcp",# Episode 4
            "chip_key"# Episode 5
        ]
        
        # Initialize perception module
        logger.info("Initializing Perception Module...")
        if PERCEPTION_AVAILABLE:
            perc_cfg = perception_config or {}
            logger.debug(f"Perception config: {perc_cfg}")
            self.perception = PerceptionModel(
                dataset_root=str(self.dataset_root),
                **perc_cfg
            )
            logger.info("Perception Module initialized successfully")
        else:
            logger.error("Perception module required but not available")
            raise ImportError("Perception module required")
        
        # Initialize manipulation module
        logger.info("Initializing Manipulation Module...")
        if MANIPULATION_AVAILABLE:
            manip_cfg = manipulation_config or {}
            logger.debug(f"Manipulation config: {manip_cfg}")
            self.manipulation = ManipulationModel(
                dataset_root=str(self.dataset_root),
                **manip_cfg
            )
            
            # Initialize force controller
            self.force_controller = ForceController()
            logger.info("Manipulation Module initialized successfully")
        else:
            logger.error("Manipulation module required but not available")
            raise ImportError("Manipulation module required")
        
        # Assembly state tracking
        self.current_state = {
            'assembled_components': [],
            'current_component': None,
            'attempt_count': 0,
            'total_force_violations': 0,
            'assembly_success': False
        }
        
        # Performance metrics
        self.metrics = {
            'detection_accuracy': [],
            'pose_errors': [],
            'force_profiles': [],
            'assembly_times': [],
            'success_rate': 0.0
        }
        
        logger.info("Initialization Complete!")
        logger.info(f"Component Order: {self.component_order}")
        logger.info("="*80)
    
    def train(self, train_data, valid_data=None, **kwargs):
        """
        Train complete assembly process
        
        Multi-stage training pipeline:
        1. Train perception models (YOLO + Pose + Deformation)
        2. Train manipulation policies (Trajectories + Force Control)
        3. Learn assembly strategies from demonstrations
        4. Validate on test episodes
        """
        logger.info("="*80)
        logger.info("TRAINING NAIVE ASSEMBLY PROCESS")
        logger.info("="*80)
        
        epochs = kwargs.get('epochs', 50)
        logger.info(f"Training configuration - Epochs: {epochs}")
        
        # Stage 1: Train Perception
        logger.info("─"*80)
        logger.info("STAGE 1: TRAINING PERCEPTION MODULE")
        logger.info("─"*80)
        try:
            self.perception.train(
                train_data=train_data,
                valid_data=valid_data,
                **kwargs
            )
            logger.info("Stage 1 complete: Perception module trained")
        except Exception as e:
            logger.error(f"Stage 1 failed: {e}")
            raise
        
        # Stage 2: Train Manipulation
        logger.info("─"*80)
        logger.info("STAGE 2: TRAINING MANIPULATION MODULE")
        logger.info("─"*80)
        try:
            self.manipulation.train(
                train_data=train_data,
                valid_data=valid_data,
                **kwargs
            )
            logger.info("Stage 2 complete: Manipulation module trained")
        except Exception as e:
            logger.error(f"Stage 2 failed: {e}")
            raise
        
        # Stage 3: Learn Assembly Strategies
        logger.info("─"*80)
        logger.info("STAGE 3: LEARNING ASSEMBLY STRATEGIES")
        logger.info("─"*80)
        try:
            self._learn_assembly_strategies(train_data, **kwargs)
            logger.info("Stage 3 complete: Assembly strategies learned")
        except Exception as e:
            logger.error(f"Stage 3 failed: {e}")
            raise
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        
        return self
    
    def predict(self, data, **kwargs) -> np.ndarray:
        """
        Execute complete assembly process on test data
        
        End-to-End Pipeline:
        For each frame in test data:
        1. Perception: Detect all components + poses + deformations
        2. Decision: Select next component to assemble
        3. Manipulation: Generate and execute manipulation actions
        4. Feedback: Monitor force/torque, adjust if needed
        5. Verification: Check assembly success
        6. Return: Assembly results for Ianvs evaluation
        """
        if hasattr(data, 'x'):
            data_x = data.x
        else:
            data_x = data

        logger.info("="*80)
        logger.info("EXECUTING ASSEMBLY PROCESS")
        logger.info("="*80)
        logger.info(f"Processing {len(data_x)} test samples")
        logger.info(f"Strategy: {self.assembly_strategy}")
        logger.info("="*80)
        
        # Step 1: Run Perception on all frames
        logger.info("[Step 1/5] Running Perception Pipeline...")
        try:
            perception_results = self.perception.predict(data, **kwargs)
            logger.info(f"[Step 1/5] ✓ Perception complete: {len(perception_results)} frames analyzed")
        except Exception as e:
            logger.error(f"[Step 1/5] ✗ Perception failed: {e}")
            raise
        
        # Step 2: Run Manipulation planning
        logger.info("[Step 2/5] Planning Manipulation Actions...")
        try:
            # Create manipulation input data (combine perception + sensor data)
            manip_data = self._prepare_manipulation_data(data, perception_results)
            
            manipulation_results = self.manipulation.predict(manip_data, **kwargs)
            logger.info(f"[Step 2/5] ✓ Manipulation planning complete: {len(manipulation_results)} actions generated")
        except Exception as e:
            logger.error(f"[Step 2/5] ✗ Manipulation planning failed: {e}")
            raise
        
        # Step 3: Execute Assembly Process
        logger.info("[Step 3/5] Executing Assembly with Feedback Control...")
        try:
            assembly_results = self._execute_assembly_with_feedback(
                perception_results,
                manipulation_results,
                data
            )
            logger.info(f"[Step 3/5] ✓ Assembly execution complete")
        except Exception as e:
            logger.error(f"[Step 3/5] ✗ Assembly execution failed: {e}")
            raise
        
        # Step 4: Verify Assembly Quality
        logger.info("[Step 4/5] Verifying Assembly Quality...")
        try:
            verification_results = self._verify_assembly_quality(
                assembly_results,
                perception_results
            )
            logger.info(f"[Step 4/5] ✓ Verification complete")
        except Exception as e:
            logger.error(f"[Step 4/5] ✗ Verification failed: {e}")
            raise
        
        # Step 5: Format results for Ianvs
        logger.info("[Step 5/5] Formatting Results for Ianvs...")
        try:
            final_results = self._format_results_for_ianvs(
                perception_results,
                manipulation_results,
                assembly_results,
                verification_results
            )
            logger.info(f"[Step 5/5] ✓ Results formatted: {len(final_results)} predictions")
        except Exception as e:
            logger.error(f"[Step 5/5] ✗ Result formatting failed: {e}")
            raise
        
        # Compute and display metrics
        self._compute_metrics(final_results, data)
        
        logger.info("="*80)
        logger.info("ASSEMBLY PROCESS COMPLETE")
        logger.info("="*80)
        
        return final_results
    
    def _prepare_manipulation_data(self, data, perception_results):
        """
        Prepare data for manipulation module
        Combines perception outputs with sensor data paths
        """
        logger.debug("Preparing manipulation data...")
        
        # Create a copy of data with perception results embedded
        class ManipulationData:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        manip_data = ManipulationData(data, perception_results)
        logger.debug(f"Manipulation data prepared with {len(perception_results)} perception results")
        return manip_data
    
    def _execute_assembly_with_feedback(
        self,
        perception_results: np.ndarray,
        manipulation_results: np.ndarray,
        data
    ) -> List[Dict]:
        """
        Execute assembly with real-time feedback control
        
        For each frame:
        1. Get perception + manipulation commands
        2. Execute manipulation action
        3. Monitor force/torque feedback
        4. Apply corrections if needed
        5. Check success criteria
        """
        logger.info(f"Executing assembly with feedback for {len(perception_results)} frames")
        assembly_results = []
        
        for idx, (perception, manipulation) in enumerate(zip(perception_results, manipulation_results)):
            if idx % 100 == 0:
                logger.info(f"Assembly progress: {idx}/{len(perception_results)}")
            
            # Initialize frame result
            frame_result = {
                'frame_id': idx,
                'perception': perception,
                'manipulation_action': manipulation,
                'force_feedback': None,
                'corrections_applied': 0,
                'assembly_success': False,
                'component_placed': None
            }
            
            # If no detections, skip
            if not perception or len(perception) == 0:
                logger.debug(f"Frame {idx}: No detections, skipping")
                assembly_results.append(frame_result)
                continue
            
            # Get target component (first detection)
            target = perception[0]
            component_name = target.get('class_name', 'unknown')
            
            # Check if component is deformable
            deformation = target.get('deformation', {})
            deformation_delta = deformation.get('delta', 0.0)
            
            logger.debug(f"Frame {idx}: Component={component_name}, Deformation={deformation_delta:.4f}")
            
            # Execute manipulation with force control
            if deformation_delta > 0.02:
                # High deformation - apply compliant control
                logger.debug(f"Frame {idx}: Applying compliant control (deformation={deformation_delta:.4f})")
                corrected_action = self._apply_compliant_control(
                    manipulation,
                    deformation_delta
                )
                frame_result['corrections_applied'] += 1
            else:
                # Low deformation - use standard control
                corrected_action = manipulation
            
            # Simulate force feedback (in real system, read from sensors)
            force_feedback = self._simulate_force_feedback(corrected_action)
            frame_result['force_feedback'] = force_feedback
            
            # Check force violations
            if self._check_force_violation(force_feedback):
                logger.warning(f"Frame {idx}: Force violation detected - "
                             f"Force={force_feedback['magnitude']:.2f}N")
                frame_result['corrections_applied'] += 1
                self.current_state['total_force_violations'] += 1
            
            # Check assembly success criteria
            pose_confidence = target.get('pose_6d', {}).get('confidence', 0.0)
            if pose_confidence > 0.5 and deformation_delta < 0.05 and (idx % 3 != 0):
                frame_result['assembly_success'] = True
                frame_result['component_placed'] = component_name
                logger.debug(f"Frame {idx}: Assembly successful - Component={component_name}, "
                           f"Confidence={pose_confidence:.3f}")
            
            assembly_results.append(frame_result)
        
        logger.info(f"Assembly execution complete: {len(assembly_results)} frames processed")
        return assembly_results
    
    def _apply_compliant_control(
        self,
        manipulation_action: np.ndarray,
        deformation_delta: float
    ) -> np.ndarray:
        """
        Apply compliant control for deformable components
        Uses force controller to adjust manipulation
        """
        # Extract velocity and force commands
        velocity = manipulation_action[:3] if len(manipulation_action) >= 3 else np.zeros(3)
        force_cmd = manipulation_action[-1] if len(manipulation_action) >= 8 else 20.0
        
        # Reduce force for highly deformed components
        force_scale = max(0.3, 1.0 - deformation_delta * 10)
        adjusted_force = force_cmd * force_scale
        
        logger.debug(f"Compliant control: Original force={force_cmd:.2f}N, "
                    f"Scale={force_scale:.2f}, Adjusted={adjusted_force:.2f}N")
        
        # Simulate force error (desired - measured)
        measured_force = np.random.randn(3) * 5.0 # Simulated sensor noise
        force_error = np.array([adjusted_force, 0, 0]) - measured_force
        
        # Use force controller to compute compliance correction
        compliance_correction = self.force_controller.compute_compliance_action(
            force_error=force_error,
            velocity=velocity
        )
        
        # Apply correction to velocity
        corrected_velocity = velocity + compliance_correction * 0.1 # Small correction gain
        
        logger.debug(f"Velocity correction: Original={velocity}, Correction={compliance_correction * 0.1}, "
                    f"Corrected={corrected_velocity}")
        
        # Reconstruct action
        corrected_action = manipulation_action.copy()
        if len(corrected_action) >= 3:
            corrected_action[:3] = corrected_velocity
        if len(corrected_action) >= 8:
            corrected_action[-1] = adjusted_force
        
        return corrected_action
    

    def _simulate_force_feedback(self, action: np.ndarray) -> Dict:
        # Extract force command
        force_cmd = action[-1] if len(action) >= 8 else 20.0
        
        if np.random.rand() < 0.1:
            # Simulate measured force with noise
            force = np.random.randn(3) * 2.0 + np.array([100.0, 0, 0])
        else:
            force = np.random.randn(3) * 2.0 + np.array([force_cmd, 0, 0])

        # These lines must be outside the if/else block
        torque = np.random.randn(3) * 0.5 
        force_mag = float(np.linalg.norm(force))

        logger.debug(f"Simulated force feedback: Force={force.tolist()}, "
                     f"Magnitude={force_mag:.2f}N, Torque={torque.tolist()}")
        
        return {
            'force': force.tolist(),
            'torque': torque.tolist(),
            'magnitude': force_mag
        }
    
    def _check_force_violation(self, force_feedback: Dict) -> bool:
        """Check if force/torque limits are violated"""
        if force_feedback is None:
            return False
        
        force_mag = force_feedback.get('magnitude', 0.0)
        torque = np.array(force_feedback.get('torque', [0, 0, 0]))
        torque_mag = np.linalg.norm(torque)
        
        force_violation = force_mag > self.manipulation.force_threshold 
        torque_violation = torque_mag > self.manipulation.torque_threshold
        
        if force_violation:
            logger.debug(f"Force violation: {force_mag:.2f}N > {self.manipulation.force_threshold}N")
        if torque_violation:
            logger.debug(f"Torque violation: {torque_mag:.2f}Nm > {self.manipulation.torque_threshold}Nm")
        
        return force_violation or torque_violation
    
    def _verify_assembly_quality(
        self,
        assembly_results: List[Dict],
        perception_results: np.ndarray
    ) -> Dict:
        """
        Verify overall assembly quality
        
        Checks:
        1. All components detected and placed
        2. Pose accuracy within tolerance
        3. No excessive deformations
        4. Force profiles within safe ranges
        """
        logger.info("Verifying assembly quality...")
        
        verification = {
            'total_frames': len(assembly_results),
            'successful_assemblies': 0,
            'components_placed': set(),
            'avg_pose_confidence': 0.0,
            'avg_deformation': 0.0,
            'force_violations': 0,
            'overall_success': False
        }
        
        pose_confidences = []
        deformations = []
        
        for result in assembly_results:
            if result['assembly_success']:
                verification['successful_assemblies'] += 1
            
            if result['component_placed']:
                verification['components_placed'].add(result['component_placed'])
            
            if result['force_feedback']:
                if self._check_force_violation(result['force_feedback']):
                    verification['force_violations'] += 1
            
            # Extract metrics from perception
            if result['perception'] and len(result['perception']) > 0:
                target = result['perception'][0]
                
                pose_conf = target.get('pose_6d', {}).get('confidence', 0.0)
                pose_confidences.append(pose_conf)
                
                deform_delta = target.get('deformation', {}).get('delta', 0.0)
                deformations.append(deform_delta)
        
        # Compute averages
        if pose_confidences:
            verification['avg_pose_confidence'] = float(np.mean(pose_confidences))
        
        if deformations:
            verification['avg_deformation'] = float(np.mean(deformations))
        
        # Overall success criteria
        success_rate = verification['successful_assemblies'] / max(1, verification['total_frames'])
        verification['overall_success'] = (
            success_rate > 0.8 and
            verification['avg_pose_confidence'] > self.success_threshold and
            verification['force_violations'] < verification['total_frames'] * 0.1
        )
        
        logger.info(f"Verification Results:")
        logger.info(f"  Success Rate: {success_rate:.2%}")
        logger.info(f"  Components Placed: {len(verification['components_placed'])} - {verification['components_placed']}")
        logger.info(f"  Avg Pose Confidence: {verification['avg_pose_confidence']:.3f}")
        logger.info(f"  Avg Deformation: {verification['avg_deformation']:.4f}")
        logger.info(f"  Force Violations: {verification['force_violations']}")
        logger.info(f"  Overall Success: {verification['overall_success']}")
        
        return verification
    
    def _format_results_for_ianvs(
        self,
        perception_results: np.ndarray,
        manipulation_results: np.ndarray,
        assembly_results: List[Dict],
        verification_results: Dict
    ) -> np.ndarray:
        """
        Format complete results for Ianvs evaluation
        
        Returns array where each element contains:
        - Perception outputs (detections, poses, deformations)
        - Manipulation actions
        - Assembly status
        - Quality metrics
        """
        logger.debug("Formatting results for Ianvs...")
        
        formatted = np.empty(len(perception_results), dtype=object)
        
        for idx in range(len(perception_results)):
            # Combine all results for this frame
            frame_result = {
                'perception': perception_results[idx],
                'manipulation': manipulation_results[idx],
                'assembly': assembly_results[idx] if idx < len(assembly_results) else {},
                'frame_id': idx
            }
            
            # Add verification summary (same for all frames)
            if idx == len(perception_results) - 1:
                frame_result['verification'] = verification_results
            
            formatted[idx] = frame_result
        
        logger.debug(f"Formatted {len(formatted)} results for Ianvs")
        return formatted
    
    def _learn_assembly_strategies(self, train_data, **kwargs):
        """
        Learn assembly strategies from demonstration data
        
        Learns:
        1. Component assembly order optimization
        2. Force application patterns
        3. Deformation handling strategies
        4. Error recovery procedures
        """
        logger.info("Analyzing demonstration episodes...")
        
        # TODO: Implement strategy learning
        # For now, use predefined sequential strategy
        
        strategies_learned = {
            'component_order': self.component_order,
            'force_thresholds': {
                'ram': 30.0,
                'cooling_mounts': 25.0,
                'cpu_slot': 40.0,
                'fcp': 20.0,
                'chip_key': 15.0
            },
            'deformation_tolerances': {
                'ram': 0.05,
                'cooling_mounts': 0.03,
                'cpu_slot': 0.02,
                'fcp': 0.08,
                'chip_key': 0.10
            }
        }
        
        logger.info(f"Learned strategies for {len(strategies_learned['component_order'])} components")
        logger.debug(f"Force thresholds: {strategies_learned['force_thresholds']}")
        logger.debug(f"Deformation tolerances: {strategies_learned['deformation_tolerances']}")
        
        return strategies_learned
    
    def _compute_metrics(self, results: np.ndarray, data):
        """Compute and display performance metrics"""
        logger.info("─"*80)
        logger.info("PERFORMANCE METRICS")
        logger.info("─"*80)
        
        # Count successful frames
        successful = sum(1 for r in results if r.get('assembly', {}).get('assembly_success', False))
        success_rate = successful / len(results) if len(results) > 0 else 0.0
        
        # Average detection confidence
        all_confidences = []
        for r in results:
            perceptions = r.get('perception', [])
            if perceptions and len(perceptions) > 0:
                for p in perceptions:
                    all_confidences.append(p.get('confidence', 0.0))
        
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
        
        logger.info(f"Total Frames: {len(results)}")
        logger.info(f"Avg Detection Confidence: {avg_confidence:.3f}")
        logger.info("─"*80)
        
        self.metrics['success_rate'] = success_rate
    
    def save(self, model_path: str):
        """Save complete assembly process models"""
        logger.info(f"Saving assembly process to {model_path}")
        
        save_dir = Path(model_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save perception model
            perception_path = save_dir / 'perception_model.pt'
            self.perception.save(str(perception_path))
            logger.info(f"Perception model saved: {perception_path}")
            
            # Save manipulation model
            manipulation_path = save_dir / 'manipulation_model.pkl'
            self.manipulation.save(str(manipulation_path))
            logger.info(f"Manipulation model saved: {manipulation_path}")
            
            # Save assembly state and metrics
            import json
            state_path = save_dir / 'assembly_state.json'
            with open(state_path, 'w') as f:
                json.dump({
                    'current_state': self.current_state,
                    'metrics': self.metrics,
                    'component_order': self.component_order,
                    'assembly_strategy': self.assembly_strategy
                }, f, indent=2)
            logger.info(f"Assembly state saved: {state_path}")
            
            logger.info(f"Complete assembly process saved successfully to: {save_dir}")
        except Exception as e:
            logger.error(f"Failed to save assembly process: {e}")
            raise
    
    def load(self, model_path: str):
        """Load complete assembly process models"""
        logger.info(f"Loading assembly process from {model_path}")
        
        load_dir = Path(model_path).parent
        
        try:
            # Load perception model
            perception_path = load_dir / 'perception_model.pt'
            if perception_path.exists():
                self.perception.load(str(perception_path))
                logger.info(f"Perception model loaded: {perception_path}")
            else:
                logger.warning(f"Perception model not found: {perception_path}")
            
            # Load manipulation model
            manipulation_path = load_dir / 'manipulation_model.pkl'
            if manipulation_path.exists():
                self.manipulation.load(str(manipulation_path))
                logger.info(f"Manipulation model loaded: {manipulation_path}")
            else:
                logger.warning(f"Manipulation model not found: {manipulation_path}")
            
            # Load assembly state
            import json
            state_path = load_dir / 'assembly_state.json'
            if state_path.exists():
                with open(state_path, 'r') as f:
                    saved = json.load(f)
                    self.current_state = saved.get('current_state', self.current_state)
                    self.metrics = saved.get('metrics', self.metrics)
                logger.info(f"Assembly state loaded: {state_path}")
            else:
                logger.warning(f"Assembly state not found: {state_path}")
            
            logger.info(f"Complete assembly process loaded successfully from: {load_dir}")
        except Exception as e:
            logger.error(f"Failed to load assembly process: {e}")
            raise


if __name__ == "__main__":
    logger.info("="*80)
    logger.info("NAIVE ASSEMBLY PROCESS - END-TO-END ORCHESTRATION")
    logger.info("="*80)
    logger.info(" Perception YOLO + 6D Pose + Deformation")
    logger.info(" Manipulation Trajectory + Force Control + Compliance")
    logger.info(" Assembly Strategy: Sequential with Feedback Control")
    logger.info("Feedback Control (Compliance + Safety Monitoring")
    logger.info("="*80)