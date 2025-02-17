from pydantic import BaseModel
from strenum import StrEnum
from typing import List

from src.watermarks.kgw_watermarks import KGW_watermark
from src.watermarks.no_watermark import No_watermark
from src.watermarks.KTH_watermark import KTH_watermark
from src.watermarks.AAR_watermark import AARWatermark
from src.watermarks.DipMark_watermark import DipMark_watermark
from src.watermarks.DeltaReweight_watermark import DeltaReweight_watermark
from src.watermarks.synthid_watermark import SynthIDWatermark



class WatermarkType(StrEnum):
    KGW = "KGW"
    KTH = "KTH"
    AAR = "AAR"
    DiPMark = "DiPMark"
    DeltaR = "DeltaR"
    SynthID = "SynthID"
    no_watermark = "no_watermark"
    
    def get_config(self, config):
        if self == WatermarkType.KGW:
            return KGWWatermarkConfiguration.parse_obj(config)
        elif self == WatermarkType.no_watermark:
            return WatermarkConfiguration.parse_obj(config)
        elif self == WatermarkType.KTH:
            return KTHWatermarkConfiguration.parse_obj(config)
        elif self == WatermarkType.AAR:
            return AARWatermarkConfiguration.parse_obj(config)
        elif self == WatermarkType.DiPMark:
            return DiPMarkWatermarkConfiguration.parse_obj(config)
        elif self == WatermarkType.DeltaR:
            return DeltaRWatermarkConfiguration.parse_obj(config)
        elif self == WatermarkType.SynthID:
            return SynthIDWatermarkConfiguration.parse_obj(config)
        else:
            raise NotImplementedError
    
    def get_watermarked_model(
        self,
        model,
        tokenizer,
        watermark_config,
        disable_watermark_every: int = 0
    ):
        if self == WatermarkType.KGW:
            return KGW_watermark(
                model=model,
                tokenizer=tokenizer,
                delta=watermark_config.delta,
                gamma=watermark_config.gamma,
                seeding_scheme=watermark_config.seeding_scheme,
                seeds=watermark_config.seeds,
                disable_watermark_every=disable_watermark_every,
            )
        elif self == WatermarkType.no_watermark:
            return No_watermark(
                model=model,
                tokenizer=tokenizer
            )
        elif self == WatermarkType.KTH:
            return KTH_watermark(
                model=model,
                tokenizer=tokenizer,
                key_length=watermark_config.key_length,
                seeds=watermark_config.seeds,
                disable_watermark_every=disable_watermark_every,
            )
        elif self == WatermarkType.AAR:
            return AARWatermark(
                model=model,
                tokenizer=tokenizer,
                context_size=watermark_config.context_size,
                lambd=watermark_config.lambd,
                seed=watermark_config.seed
            )
        elif self == WatermarkType.DiPMark:
            return DipMark_watermark(
                model=model,
                tokenizer=tokenizer,
                context_size=watermark_config.context_size,
                seeds=watermark_config.seeds,
                alpha=watermark_config.alpha,
                use_cache=watermark_config.use_cache,
                disable_watermark_every=disable_watermark_every,
            )
        elif self == WatermarkType.DeltaR:
            return DeltaReweight_watermark(
                model=model,
                tokenizer=tokenizer,
                context_size=watermark_config.context_size,
                seeds=watermark_config.seeds,
                use_cache=watermark_config.use_cache,
                disable_watermark_every=disable_watermark_every,
            )
        elif self == WatermarkType.SynthID:
            return SynthIDWatermark(
                model_name = watermark_config.model_name,
            )


class WatermarkConfiguration(BaseModel):
    type: WatermarkType
    
    def get_watermarked_model(
        self,
        model,
        tokenizer,
        disable_watermark_every: int = 0 
    ):
        return self.type.get_watermarked_model(
            model=model,
            tokenizer=tokenizer,
            watermark_config=self,
            disable_watermark_every=disable_watermark_every,
        )


class KGWWatermarkConfiguration(WatermarkConfiguration):
    delta: float
    gamma: float
    context_size: int
    seeding_scheme: str # lefthash or selfhash.
    seeds: List[int]
    
class KTHWatermarkConfiguration(WatermarkConfiguration):
    key_length: int
    seeds: List[int]
    
class AARWatermarkConfiguration(WatermarkConfiguration):
    context_size: int
    lambd: float
    seed: int
    
class DiPMarkWatermarkConfiguration(WatermarkConfiguration):
    seeds: List[int]
    alpha: float
    use_cache: bool
    context_size: int
    
class DeltaRWatermarkConfiguration(WatermarkConfiguration):
    context_size: int
    seeds: List[int]
    use_cache: bool
    
class SynthIDWatermarkConfiguration(WatermarkConfiguration):
    model_name: str