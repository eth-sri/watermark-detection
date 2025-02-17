from pydantic import BaseModel, root_validator
from src.configs.watermark_configs import WatermarkType, WatermarkConfiguration
from src.configs.test_configs import TestType, TestConfiguration
    
class MainConfiguration(BaseModel):
    # Watermark settings
    watermark_type: WatermarkType
    watermark_config: WatermarkConfiguration

    # Test settings
    test_type: TestType
    test_config: TestConfiguration
    disable_watermark_every: int = 0  # Disable watermark every n requests

    # Model settings
    model_name: str
    temperature: float

    @root_validator(pre=True)
    def assign_watermark_config(cls, values):
        """
        Before validation, check the watermark type and parse the watermark_config field
        into the appropriate subclass.
        """
        
        # Watermark dynamic config loading
        wm_type = values.get("watermark_type")
        wm_config = values.get("watermark_config")
        
        wm_type = WatermarkType(wm_type)
        values["watermark_config"] = wm_type.get_config(wm_config)
        
        # Test dynamic config loading
        test_type = values.get("test_type")
        test_config = values.get("test_config")
        
        test_type = TestType(test_type)
        values["test_config"] = test_type.get_config(test_config)
        
        return values