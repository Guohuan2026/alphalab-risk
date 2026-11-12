"""Risk-forecasting features. Every value stamped at session t uses data published by the
end of day t; the label (next-day loss) lives in ``targets`` and is never an input."""

from alphalab.features.core import align_asof, assemble_inputs, build_features, series_name
from alphalab.features.targets import next_day_loss

__all__ = ["align_asof", "assemble_inputs", "build_features", "next_day_loss", "series_name"]
