"""External data-provider adapters."""

from eurostress.sources.ecb import EcbClient
from eurostress.sources.yahoo import YahooClient

__all__ = ["EcbClient", "YahooClient"]

