"""
Schema Generation Module

This module contains classes and utilities for generating database schemas
from unstructured documents using LLMs.
"""

from .schemagen_basic import SchemaGenBasic
from .schemagen_gpt import SchemaGenGPT
from .schemagen_deepseek import SchemaGenDeepSeek
from .schemagen_together import SchemaGenTogether
from .schemagen_siliconflow import SchemaGenSiliconFlow

# Adaptive sampling optimization (optional import)
try:
    from .opt import AdaptiveSampler, SchemaEntropyCalculator, AdaptiveSamplingMixin
    __all__ = [
        "SchemaGenBasic",
        "SchemaGenGPT",
        "SchemaGenDeepSeek", 
        "SchemaGenTogether",
        "SchemaGenSiliconFlow",
        "AdaptiveSampler",
        "SchemaEntropyCalculator",
        "AdaptiveSamplingMixin",
    ]
except ImportError:
    __all__ = [
        "SchemaGenBasic",
        "SchemaGenGPT",
        "SchemaGenDeepSeek", 
        "SchemaGenTogether",
        "SchemaGenSiliconFlow",
    ]
