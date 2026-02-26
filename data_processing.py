"""
Data Processing Module for Belt Tracking Swing Arm Analysis

Takes collected data and applies processing steps (e.g. averaging per measurement stream).
Outputs are consumed by the visualization module for the report.
"""

import re
from typing import Any

import pandas as pd


def _get_value_column(df: pd.DataFrame) -> str | None:
    """Determine the value column name in a DataFrame."""
    if "mean" in df.columns:
        return "mean"
    if "value" in df.columns:
        return "value"
    return None


class DataProcessor:
    """
    Processes collected test data for visualization.
    Takes raw data from DataCollector and produces derived quantities (e.g. averages by position).
    """

    def __init__(self, designators: list[str] | None = None):
        """
        Initialize the processor.

        Args:
            designators: List of sensor designator names (e.g. R1_LEFT_ANGLE, ...).
                         If None, inferred from the first test's keys when processing.
        """
        self.designators = designators

    def process(
        self,
        all_test_data: dict[str, dict[str, pd.DataFrame]],
    ) -> dict[str, Any]:
        """
        Run all processing steps on the collected data.

        Args:
            all_test_data: From DataCollector.collect_all_tests(): test name -> designator -> DataFrame.

        Returns:
            Dict of processing outputs, e.g. {"average_angles_by_position": {...}}.
        """
        if not all_test_data:
            return {}
        designators = self.designators or list(next(iter(all_test_data.values())).keys())
        avg = self.average_by_position(all_test_data, designators)
        delta = self.delta_mistracking_baseline(avg)
        centerline = self.centerline_comparison(delta)
        return {
            "average_angles_by_position": avg,
            "delta_mistracking_baseline": delta,
            "centerline_comparison": centerline,
            "re_tracking_summary": self.re_tracking_summary(centerline),
        }

    def average_by_position(
        self,
        all_test_data: dict[str, dict[str, pd.DataFrame]],
        designators: list[str] | None = None,
    ) -> dict[str, dict[str, list]]:
        """
        First processing step: average (and std) of each measurement stream per test.
        Splits results by left/right and by roller position (R1, R3, R5, R7).

        Args:
            all_test_data: test name -> designator -> DataFrame with value column.
            designators: Optional list; if None, uses self.designators or keys from first test.

        Returns:
            For each test name, a dict with keys:
            left_x, left_y, left_std, right_x, right_y, right_std
            (lists of positions and values for spline plot and table).
        """
        if designators is None:
            designators = self.designators or list(
                next(iter(all_test_data.values())).keys()
            )
        result: dict[str, dict[str, list]] = {}
        for test_name, test_data in all_test_data.items():
            result[test_name] = self._average_one_test(test_data, designators)
        return result

    def _average_one_test(
        self,
        test_data: dict[str, pd.DataFrame],
        designators: list[str],
    ) -> dict[str, list]:
        """Compute average and std per designator for one test; return left/right spline data."""
        left_positions: list[float] = []
        left_avgs: list[float] = []
        left_stds: list[float] = []
        right_positions: list[float] = []
        right_avgs: list[float] = []
        right_stds: list[float] = []

        for designator in designators:
            df = test_data.get(designator, pd.DataFrame())
            value_col = _get_value_column(df)
            if df.empty or not value_col:
                continue
            avg = float(df[value_col].mean())
            std = df[value_col].std()
            std = 0.0 if pd.isna(std) else float(std)
            match = re.match(r"R(\d+)", designator, re.IGNORECASE)
            pos = float(match.group(1)) if match else 0.0
            if "LEFT" in designator.upper():
                left_positions.append(pos)
                left_avgs.append(avg)
                left_stds.append(std)
            elif "RIGHT" in designator.upper():
                right_positions.append(pos)
                right_avgs.append(avg)
                right_stds.append(std)

        if left_positions:
            sorted_lr = sorted(zip(left_positions, left_avgs, left_stds))
            left_positions = [t[0] for t in sorted_lr]
            left_avgs = [t[1] for t in sorted_lr]
            left_stds = [t[2] for t in sorted_lr]
        if right_positions:
            sorted_lr = sorted(zip(right_positions, right_avgs, right_stds))
            right_positions = [t[0] for t in sorted_lr]
            right_avgs = [t[1] for t in sorted_lr]
            right_stds = [t[2] for t in sorted_lr]

        return {
            "left_x": left_positions,
            "left_y": left_avgs,
            "left_std": left_stds,
            "right_x": right_positions,
            "right_y": right_avgs,
            "right_std": right_stds,
        }

    # Test names and trace spec for Step 2: delta (mistrack − baseline 300 FPM)
    PAT_BASELINE_300 = "PAT Baseline - 300 FPM"
    STANDARD_BASELINE_300 = "Standard Baseline - 300 FPM"
    PAT_MISTRACK_LEFT = "PAT Mistracked - Mining Left"
    PAT_MISTRACK_RIGHT = "PAT Mistracked - Mining Right"
    STANDARD_MISTRACK_LEFT = "Standard Mistracked - Mining Left"
    STANDARD_MISTRACK_RIGHT = "Standard Mistracked - Mining Right"
    DELTA_TRACE_SPECS = [
        (PAT_BASELINE_300, PAT_MISTRACK_LEFT, "left", "PAT Left, Mistrack Left − Baseline"),
        (PAT_BASELINE_300, PAT_MISTRACK_LEFT, "right", "PAT Right, Mistrack Left − Baseline"),
        (PAT_BASELINE_300, PAT_MISTRACK_RIGHT, "left", "PAT Left, Mistrack Right − Baseline"),
        (PAT_BASELINE_300, PAT_MISTRACK_RIGHT, "right", "PAT Right, Mistrack Right − Baseline"),
        (STANDARD_BASELINE_300, STANDARD_MISTRACK_LEFT, "left", "Standard Left, Mistrack Left − Baseline"),
        (STANDARD_BASELINE_300, STANDARD_MISTRACK_LEFT, "right", "Standard Right, Mistrack Left − Baseline"),
        (STANDARD_BASELINE_300, STANDARD_MISTRACK_RIGHT, "left", "Standard Left, Mistrack Right − Baseline"),
        (STANDARD_BASELINE_300, STANDARD_MISTRACK_RIGHT, "right", "Standard Right, Mistrack Right − Baseline"),
    ]

    def delta_mistracking_baseline(
        self,
        average_angles_by_position: dict[str, dict[str, list]],
    ) -> list[dict[str, Any]]:
        """
        Step 2: compute delta (mistrack − baseline 300 FPM) by position for each trace.
        Uses average_angles_by_position from average_by_position().

        Returns:
            List of 8 dicts, each with keys "name", "x", "y" (lists of position and delta angle).
        """
        def get_xy(spline: dict, side: str) -> tuple[list[float], list[float]]:
            x_key = "left_x" if side == "left" else "right_x"
            y_key = "left_y" if side == "left" else "right_y"
            x = spline.get(x_key, [])
            y = spline.get(y_key, [])
            return (list(x), [float(v) for v in y])

        result: list[dict[str, Any]] = []
        positions = [1.0, 3.0, 5.0, 7.0]
        for baseline_name, mistrack_name, side, trace_name in self.DELTA_TRACE_SPECS:
            baseline = average_angles_by_position.get(baseline_name, {})
            mistrack = average_angles_by_position.get(mistrack_name, {})
            _, base_y = get_xy(baseline, side)
            mx, mist_y = get_xy(mistrack, side)
            if not base_y or not mist_y:
                result.append({"name": trace_name, "x": [], "y": []})
                continue
            n = min(len(base_y), len(mist_y))
            x = mx[:n] if mx else positions[:n]
            delta_y = [abs(mist_y[j] - base_y[j]) for j in range(n)]
            result.append({"name": trace_name, "x": x, "y": delta_y})
        return result

    # Step 3: centerline = average of left and right delta traces from Step 2 (pairs of indices)
    CENTERLINE_DISPLAY_NAMES = [
        "PAT Mistracked Left",
        "PAT Mistracked Right",
        "Standard Mistracked Left",
        "Standard Mistracked Right",
    ]

    def centerline_comparison(
        self,
        delta_mistracking_baseline: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Step 3: centerline = (left delta + right delta) / 2 at each position for each mistrack condition.
        Takes the output of delta_mistracking_baseline() (Step 2): 8 traces in order
        [PAT L/R mistrack left, PAT L/R mistrack right, Standard L/R mistrack left, Standard L/R mistrack right].
        Each pair (0,1), (2,3), (4,5), (6,7) is averaged to produce one centerline trace.

        Returns:
            List of 4 dicts, each with keys "name", "x", "y" (position and centerline delta [deg]).
        """
        result: list[dict[str, Any]] = []
        positions = [1.0, 3.0, 5.0, 7.0]
        for pair_idx in range(4):
            i0, i1 = pair_idx * 2, pair_idx * 2 + 1
            display_name = self.CENTERLINE_DISPLAY_NAMES[pair_idx]
            t0 = delta_mistracking_baseline[i0] if i0 < len(delta_mistracking_baseline) else {}
            t1 = delta_mistracking_baseline[i1] if i1 < len(delta_mistracking_baseline) else {}
            y0 = t0.get("y", [])
            y1 = t1.get("y", [])
            x0 = t0.get("x", [])
            if not y0 or not y1:
                result.append({"name": display_name, "x": [], "y": []})
                continue
            n = min(len(y0), len(y1))
            x = x0[:n] if x0 else positions[:n]
            centerline_y = [(float(y0[j]) + float(y1[j])) / 2.0 for j in range(n)]
            result.append({"name": display_name, "x": x, "y": centerline_y})
        return result

    def re_tracking_summary(
        self,
        centerline_comparison: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Step 4: Re-Tracking Summary = average of the two centerline traces (left/right mistrack) per system.
        Takes the output of centerline_comparison() (Step 3): 4 traces in order
        [PAT Mistracked Left, PAT Mistracked Right, Standard Mistracked Left, Standard Mistracked Right].
        PAT = (trace 0 + trace 1) / 2, Standard = (trace 2 + trace 3) / 2.

        Returns:
            List of 2 dicts: {"name", "x", "y", "slope", "intercept"} for PAT and Standard.
        """
        result: list[dict[str, Any]] = []
        # Step 4 x-axis in feet: map R1,R3,R5,R7 -> 0, 10, 20, 30 ft
        x_feet = [0.0, 10.0, 20.0, 30.0]
        for pair_idx, display_name in enumerate(["PAT", "Standard"]):
            i0, i1 = pair_idx * 2, pair_idx * 2 + 1
            t0 = centerline_comparison[i0] if i0 < len(centerline_comparison) else {}
            t1 = centerline_comparison[i1] if i1 < len(centerline_comparison) else {}
            y0 = t0.get("y", [])
            y1 = t1.get("y", [])
            if not y0 or not y1:
                result.append({"name": display_name, "x": [], "y": [], "slope": None, "intercept": None})
                continue
            n = min(len(y0), len(y1))
            x = x_feet[:n]
            summary_y = [(float(y0[j]) + float(y1[j])) / 2.0 for j in range(n)]
            # Linear regression: y = slope * x + intercept
            x_mean = sum(x) / n
            y_mean = sum(summary_y) / n
            ss_xy = sum((x[j] - x_mean) * (summary_y[j] - y_mean) for j in range(n))
            ss_xx = sum((x[j] - x_mean) ** 2 for j in range(n))
            slope = ss_xy / ss_xx if ss_xx != 0 else 0.0
            intercept = y_mean - slope * x_mean
            result.append({
                "name": display_name,
                "x": x,
                "y": summary_y,
                "slope": slope,
                "intercept": intercept,
            })
        return result
