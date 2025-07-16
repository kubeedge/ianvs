# Copyright 2024 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import, division, print_function

import os
from PIL import Image
from dashscope import MultiModalConversation
from core.common.log import LOGGER
from sedna.common.class_factory import ClassFactory, ClassType

os.environ['BACKEND_TYPE'] = 'TORCH'
__all__ = ["QwenSemanticSegmentation"]

# Image resolution settings based on risk levels
RISK_LEVELS = {
    "low": "640x480",
    "medium": "1280x760",
    "high": "2048x1024"
}

@ClassFactory.register(ClassType.GENERAL, alias="CloudModel")
class CloudModel:
    """Cloud-based semantic segmentation service wrapper for Qwen multimodal models.
    
    Provides remote image segmentation capabilities through DashScope's API.
    Handles image encoding and API communication automatically.
    
    Attributes:
        api_key (str): Required API key from environment variable DASHSCOPE_API_KEY
        command_text (str): Instruction text for the model (default segmentation task)
        model_name (str): Qwen model version (default: qwen-vl-max)
    """
    def __init__(self,**kwargs):
        """Initialize CloudModel with API credentials and configuration.
        
        Args:
            **kwargs: Optional configuration parameters:
                - command_text (str): Custom instruction text
                - model_name (str): Alternative Qwen model version
        """
        self.api_key = os.getenv("DASHSCOPE_API_KEY") # Replace with your actual API key
        self.command_text = kwargs.get("command_text",
                                       "Please perform semantic segmentation on the image and return the result.")   
        self.model_name = kwargs.get("model_name","qwen-vl-max")
        # self.tmp_dir = kwargs.get("tmp_dir","./tmp")
        # os.makedirs(self.tmp_dir, exist_ok=True)

    # def _resize_img(self, image_path):
    #     """Resize image to specified width while maintaining aspect ratio.
    #     Args:
    #         image_path (str): Path to input image file
    #         new_width (int): Desired width in pixels (default: 640)
    #     Returns:
    #         PIL.Image: Resized image object
    #     """
    #     img = Image.open(image_path)
    #     width, height = img.size
    #     risk = os.getenv("CLOUD_OFFLOADING_DECISION","low")
    #     new_width = RISK_LEVELS.get(risk,640)
    #     new_height = int((new_width * height) / width)  # Maintain aspect ratio 
    #     resized_img = img.resize((new_width, new_height))
    #     tmp_path = os.path.join(self.tmp_dir, f"resized_{os.path.basename(image_path)}")
    #     resized_img.save(tmp_path)
    #     return os.path.abspath(tmp_path)
  
    def _relocate_img(self, image_path):
        """Relocate image to the relative(low, medium, high resolution image) directory.
        Args:
            image_path (str): Path to input image file
        Returns:
            str: New path to relocated image file
        """
        risk = os.getenv("CLOUD_OFFLOADING_DECISION","low")
        new_img = RISK_LEVELS.get(risk,"2048x1024")
        tmp_path = os.path.abspath(image_path)
        high_res_str = RISK_LEVELS["high"]
        if high_res_str in tmp_path:
            tmp_path = tmp_path.replace(high_res_str, new_img)
        img = Image.open(tmp_path)
        LOGGER.info(f"Using {risk} resolution image for cloud inference: {img.size}")
        return tmp_path

    def inference(self, image_path, post_process=None, **kwargs):
        """Perform cloud-based semantic segmentation.
        
        Args:
            image_path (str): Path to input image file
            post_process (callable, optional): Function for processing raw output
            **kwargs: Additional parameters for API call
            
        Returns:
            dashscope.MultiModalConversation.Response: Raw API response
            
        Example:
            >>> model = CloudModel()
            >>> response = model.inference("test.jpg")
            >>> if response.status_code == 200:
            ...     print(response.output)
        """
        response = MultiModalConversation.call(
            model=self.model_name,
            api_key=self.api_key,
            messages=[{
                'role': 'user',
                'content': [
                    {'image': self._relocate_img(image_path)},
                    {'text': self.command_text}
                ]
            }],
        )
        if response.status_code == 200:
            print(response.output['choices'][0]['message']['content'])
        else:
            print("segmentation failure:", response.get("message", "unknown error"))
        
        return response 
    

