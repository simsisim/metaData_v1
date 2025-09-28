"""
Transformation Factory
======================

Factory class for creating origin-specific transformation strategies.
Routes processing based on file_origin to appropriate strategy implementation.
"""

from typing import Dict, Type
import logging
from ..strategies.base_strategy import TransformationStrategy
from ..strategies.tw_strategy import TWTransformationStrategy
from ..strategies.gf_strategy import GFTransformationStrategy

logger = logging.getLogger(__name__)


class TransformationFactory:
    """
    Factory for creating transformation strategies based on file origin.

    Supported origins:
    - TW: TradingView files
    - GF: Google Finance files
    """

    # Registry of available strategies
    _strategies: Dict[str, Type[TransformationStrategy]] = {
        'TW': TWTransformationStrategy,
        'GF': GFTransformationStrategy
    }

    @classmethod
    def get_strategy(cls, file_origin: str) -> TransformationStrategy:
        """
        Get transformation strategy for specified file origin.

        Args:
            file_origin: Origin identifier (TW, GF, etc.)

        Returns:
            Appropriate transformation strategy instance

        Raises:
            ValueError: If file_origin is not supported
        """
        origin_upper = file_origin.upper().strip()

        if origin_upper not in cls._strategies:
            available_origins = list(cls._strategies.keys())
            raise ValueError(
                f"Unsupported file_origin: '{file_origin}'. "
                f"Available origins: {available_origins}"
            )

        strategy_class = cls._strategies[origin_upper]
        strategy_instance = strategy_class()

        logger.info(f"Created {origin_upper} transformation strategy")
        return strategy_instance

    @classmethod
    def register_strategy(cls, file_origin: str, strategy_class: Type[TransformationStrategy]):
        """
        Register a new transformation strategy.

        Args:
            file_origin: Origin identifier
            strategy_class: Strategy class implementing TransformationStrategy

        Raises:
            TypeError: If strategy_class doesn't implement TransformationStrategy
        """
        if not issubclass(strategy_class, TransformationStrategy):
            raise TypeError(
                f"Strategy class must inherit from TransformationStrategy. "
                f"Got: {strategy_class}"
            )

        origin_upper = file_origin.upper().strip()
        cls._strategies[origin_upper] = strategy_class

        logger.info(f"Registered strategy for origin '{origin_upper}': {strategy_class.__name__}")

    @classmethod
    def get_supported_origins(cls) -> list:
        """
        Get list of supported file origins.

        Returns:
            List of supported origin identifiers
        """
        return list(cls._strategies.keys())

    @classmethod
    def is_origin_supported(cls, file_origin: str) -> bool:
        """
        Check if file origin is supported.

        Args:
            file_origin: Origin identifier to check

        Returns:
            True if origin is supported, False otherwise
        """
        return file_origin.upper().strip() in cls._strategies

    @classmethod
    def get_strategy_info(cls, file_origin: str) -> Dict:
        """
        Get information about a specific strategy.

        Args:
            file_origin: Origin identifier

        Returns:
            Strategy information dictionary

        Raises:
            ValueError: If file_origin is not supported
        """
        if not cls.is_origin_supported(file_origin):
            raise ValueError(f"Unsupported file_origin: '{file_origin}'")

        origin_upper = file_origin.upper().strip()
        strategy_class = cls._strategies[origin_upper]

        return {
            'origin': origin_upper,
            'strategy_class': strategy_class.__name__,
            'module': strategy_class.__module__,
            'docstring': strategy_class.__doc__ or "No description available"
        }

    @classmethod
    def create_strategy_for_rule(cls, rule: Dict) -> TransformationStrategy:
        """
        Create strategy from configuration rule.

        Args:
            rule: Configuration rule containing file_origin

        Returns:
            Appropriate transformation strategy

        Raises:
            KeyError: If rule doesn't contain file_origin
            ValueError: If file_origin is not supported
        """
        if 'file_origin' not in rule:
            raise KeyError("Rule must contain 'file_origin' field")

        file_origin = rule['file_origin']
        strategy = cls.get_strategy(file_origin)

        logger.debug(f"Created strategy for rule with origin '{file_origin}'")
        return strategy

    @classmethod
    def get_factory_summary(cls) -> Dict:
        """
        Get summary of factory configuration.

        Returns:
            Factory summary information
        """
        summary = {
            'total_strategies': len(cls._strategies),
            'supported_origins': cls.get_supported_origins(),
            'strategies': {}
        }

        for origin, strategy_class in cls._strategies.items():
            summary['strategies'][origin] = {
                'class_name': strategy_class.__name__,
                'module': strategy_class.__module__
            }

        return summary