"""
Visualization Module for Belt Tracking Swing Arm Analysis
Uses Plotly to generate interactive HTML reports
"""

import html as html_module
import json

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
from datetime import datetime


# Base URL for Grafana dashboard; {from} and {to} are replaced with test time range (ISO 8601)
GRAFANA_BASE_URL = (
    "https://data.prufrock.dev/d/joj6pc4/swing-arm-belt-edge-detection"
    "?folderUid=dd515790-d4d4-480a-88ca-6d1ba733be9c&orgId=1"
    "&from={from}&to={to}&timezone=browser"
    "&var-query0=&var-datasource=p_EKvrBnk&var-query0-2=&var-policy=autogen"
    "&var-query0-3=&var-filter=&var-vin=&refresh=auto"
)


class Visualizer:
    """Handles Plotly visualizations and HTML report generation."""

    # Color palette for designators (visually distinct, colorblind-friendly)
    COLORS = [
        '#2E86AB',  # Steel Blue
        '#A23B72',  # Mulberry
        '#F18F01',  # Orange
        '#C73E1D',  # Vermillion
        '#3B1F2B',  # Dark Purple
        '#95C623',  # Yellow Green
        '#5C4D7D',  # Purple
        '#00A6A6',  # Teal
    ]
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize Visualizer.
        
        Args:
            output_dir: Directory for HTML reports. Defaults to 'docs/' folder.
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / "docs"
        self.output_dir = output_dir
        self._ensure_output_dir()
    
    def _ensure_output_dir(self) -> None:
        """Ensure the output directory exists."""
        self.output_dir.mkdir(exist_ok=True)
    
    def _generate_filename(self, test_name: str) -> Path:
        """Generate a unique filename for the report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_dir / f"{test_name}_{timestamp}.html"
    
    @staticmethod
    def _get_value_column(df: pd.DataFrame) -> str:
        """Determine the value column name in a DataFrame."""
        if 'mean' in df.columns:
            return 'mean'
        elif 'value' in df.columns:
            return 'value'
        return None
    
    def plot_single(self, df: pd.DataFrame, designator: str) -> go.Figure:
        """Create a simple line plot for a single designator."""
        fig = go.Figure()
        value_col = self._get_value_column(df)
        if not df.empty and value_col:
            fig.add_trace(go.Scatter(
                x=df['time'],
                y=df[value_col],
                mode='lines',
                name=designator
            ))
        fig.update_layout(
            title=designator,
            xaxis_title="Time",
            yaxis_title="Angle [deg]",
            template="plotly_white"
        )
        return fig
    
    def plot_grid(self, data: dict[str, pd.DataFrame], title: str = "Test") -> go.Figure:
        """Create a subplot grid with all designators."""
        designators = list(data.keys())
        n_designators = len(designators)
        n_rows = (n_designators + 1) // 2
        fig = make_subplots(
            rows=n_rows,
            cols=2,
            subplot_titles=designators,
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        for i, (designator, df) in enumerate(data.items()):
            row, col = (i // 2) + 1, (i % 2) + 1
            value_col = self._get_value_column(df)
            if not df.empty and value_col:
                fig.add_trace(
                    go.Scatter(x=df['time'], y=df[value_col], mode='lines', name=designator, showlegend=False),
                    row=row, col=col
                )
        for i in range(n_designators):
            yaxis_name = f"yaxis{i + 1}" if i > 0 else "yaxis"
            fig.update_layout(**{yaxis_name: dict(title="Angle [deg]")})
        fig.update_layout(title=f"Belt Tracking Swing Arm Analysis - {title}", height=300 * n_rows, template="plotly_white")
        return fig
    
    def plot_comparison(self, data: dict[str, pd.DataFrame], title: str = "Test") -> go.Figure:
        """Create an overlay plot comparing all designators."""
        fig = go.Figure()
        for designator, df in data.items():
            value_col = self._get_value_column(df)
            if not df.empty and value_col:
                fig.add_trace(go.Scatter(x=df['time'], y=df[value_col], mode='lines', name=designator))
        fig.update_layout(
            title=f"All Designators Comparison - {title}",
            xaxis_title="Time",
            yaxis_title="Angle [deg]",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
    
    def plot_multi_test_grid(
        self,
        all_test_data: dict[str, dict[str, pd.DataFrame]],
        average_angles_by_position: dict[str, dict[str, list]] | None = None,
        title: str = "Multi-Test Analysis",
    ) -> go.Figure:
        """
        Create a subplot grid with dropdown to switch between tests.
        Includes comparison plot and average angle by position (two splines).
        average_angles_by_position: From DataProcessor.process()['average_angles_by_position'].
        """
        test_names = list(all_test_data.keys())
        if average_angles_by_position is None:
            average_angles_by_position = {}
        if not test_names:
            raise ValueError("No test data provided")

        first_test_data = all_test_data[test_names[0]]
        designators = list(first_test_data.keys())
        n_designators = len(designators)
        n_grid_rows = (n_designators + 1) // 2
        n_total_rows = n_grid_rows + 3

        subplot_titles = designators + ["All Designators Comparison", "Average Angle by Position", ""]
        row_heights = [1] * n_grid_rows + [1.5, 1.0, 1.1]
        specs = [[{}, {}] for _ in range(n_grid_rows)]
        specs.append([{"colspan": 2}, None])
        specs.append([{"colspan": 2}, None])
        specs.append([{"type": "table", "colspan": 2}, None])

        fig = make_subplots(
            rows=n_total_rows,
            cols=2,
            subplot_titles=subplot_titles,
            vertical_spacing=0.06,
            horizontal_spacing=0.1,
            row_heights=row_heights,
            specs=specs,
            shared_xaxes=True
        )
        
        traces_per_test = n_designators * 2 + 2
        
        for test_idx, (test_name, test_data) in enumerate(all_test_data.items()):
            visible = (test_idx == 0)
            
            for i, designator in enumerate(designators):
                row, col = (i // 2) + 1, (i % 2) + 1
                color = self.COLORS[i % len(self.COLORS)]
                df = test_data.get(designator, pd.DataFrame())
                value_col = self._get_value_column(df)
                if not df.empty and value_col:
                    fig.add_trace(
                        go.Scatter(
                            x=df['time'], y=df[value_col], mode='lines',
                            name=f"{designator} ({test_name})", showlegend=False, visible=visible,
                            line=dict(color=color, width=1.5), hovertemplate='%{y:.2f} deg<extra></extra>'
                        ),
                        row=row, col=col
                    )
                else:
                    fig.add_trace(
                        go.Scatter(x=[], y=[], mode='lines', name=f"{designator} ({test_name})",
                                   showlegend=False, visible=visible, line=dict(color=color, width=1.5)),
                        row=row, col=col
                    )
            
            comparison_row = n_total_rows - 2
            for i, designator in enumerate(designators):
                df = test_data.get(designator, pd.DataFrame())
                value_col = self._get_value_column(df)
                color = self.COLORS[i % len(self.COLORS)]
                if not df.empty and value_col:
                    fig.add_trace(
                        go.Scatter(
                            x=df['time'], y=df[value_col], mode='lines', name=designator,
                            showlegend=True, visible=visible,
                            line=dict(color=color, width=1.5), hovertemplate='%{y:.2f} deg<extra></extra>'
                        ),
                        row=comparison_row, col=1
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=[], y=[], mode='lines', name=designator, showlegend=True, visible=visible,
                            line=dict(color=color, width=1.5)
                        ),
                        row=comparison_row, col=1
                    )
            
            spline = average_angles_by_position.get(test_name, {})
            left_x = spline.get("left_x", [])
            left_y = spline.get("left_y", [])
            left_std = spline.get("left_std", [])
            right_x = spline.get("right_x", [])
            right_y = spline.get("right_y", [])
            right_std = spline.get("right_std", [])
            if left_x and left_y:
                fig.add_trace(
                    go.Scatter(
                        x=left_x, y=left_y, mode='lines+markers', name='Left',
                        showlegend=False, visible=visible,
                        line=dict(shape='spline', width=2, color='#2E86AB'),
                        error_y=dict(
                            type='data',
                            array=[float(s) for s in left_std],
                            symmetric=True,
                            color='#2E86AB',
                            thickness=1.5,
                            width=4,
                            visible=True,
                        ),
                        customdata=[float(s) for s in left_std],
                        hovertemplate='R%{x:.0f} Left: %{y:.2f} ± %{customdata:.2f} deg<extra></extra>'
                    ),
                    row=n_total_rows - 1, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(x=[], y=[], mode='lines', name='Left', showlegend=False, visible=visible),
                    row=n_total_rows - 1, col=1
                )
            if right_x and right_y:
                fig.add_trace(
                    go.Scatter(
                        x=right_x, y=right_y, mode='lines+markers', name='Right',
                        showlegend=False, visible=visible,
                        line=dict(shape='spline', width=2, color='#C73E1D'),
                        error_y=dict(
                            type='data',
                            array=[float(s) for s in right_std],
                            symmetric=True,
                            color='#C73E1D',
                            thickness=1.5,
                            width=4,
                            visible=True,
                        ),
                        customdata=[float(s) for s in right_std],
                        hovertemplate='R%{x:.0f} Right: %{y:.2f} ± %{customdata:.2f} deg<extra></extra>'
                    ),
                    row=n_total_rows - 1, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(x=[], y=[], mode='lines', name='Right', showlegend=False, visible=visible),
                    row=n_total_rows - 1, col=1
                )

            # Table: same data as Average Angle by Position (Position, Left/Right Avg, then Left/Right Std)
            positions = [1, 3, 5, 7]
            pos_labels = [f"R{p}" for p in positions]
            left_avg_str = []
            left_std_str = []
            right_avg_str = []
            right_std_str = []
            for p in positions:
                if left_x and p in left_x:
                    idx = left_x.index(p)
                    left_avg_str.append(f"{left_y[idx]:.2f}")
                    left_std_str.append(f"{left_std[idx]:.2f}")
                else:
                    left_avg_str.append("—")
                    left_std_str.append("—")
                if right_x and p in right_x:
                    idx = right_x.index(p)
                    right_avg_str.append(f"{right_y[idx]:.2f}")
                    right_std_str.append(f"{right_std[idx]:.2f}")
                else:
                    right_avg_str.append("—")
                    right_std_str.append("—")
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=["Position", "Left Avg [deg]", "Right Avg [deg]", "Left Std Dev [deg]", "Right Std Dev [deg]"],
                        fill_color="#F0F0F0",
                        align="center",
                        font=dict(size=11, color="#2C3E50", family="Arial, sans-serif"),
                    ),
                    cells=dict(
                        values=[pos_labels, left_avg_str, right_avg_str, left_std_str, right_std_str],
                        fill_color="white",
                        align="center",
                        font=dict(size=11, color="#333", family="Arial, sans-serif"),
                        height=28,
                    ),
                    visible=visible,
                ),
                row=n_total_rows,
                col=1,
            )

        yaxis_title_style = dict(text="Angle [deg]", font=dict(size=12, color="#2C3E50", family="Arial, sans-serif"))
        for i in range(n_designators):
            yaxis_name = f"yaxis{i + 1}" if i > 0 else "yaxis"
            fig.update_layout(**{yaxis_name: dict(
                title=yaxis_title_style,
                gridcolor='#E5E5E5', zerolinecolor='#E5E5E5'
            )})
        comparison_yaxis = f"yaxis{n_designators + 1}"
        fig.update_layout(**{comparison_yaxis: dict(
            title=yaxis_title_style,
            gridcolor='#E5E5E5', zerolinecolor='#E5E5E5'
        )})
        avg_plot_yaxis = f"yaxis{n_designators + 2}"
        avg_plot_xaxis = f"xaxis{n_designators + 2}"
        fig.update_layout(**{avg_plot_yaxis: dict(
            title=yaxis_title_style,
            gridcolor='#E5E5E5', zerolinecolor='#E5E5E5'
        )})
        if avg_plot_xaxis in fig.layout:
            fig.layout[avg_plot_xaxis].title = dict(text="Sensor position", font=dict(size=11, color='#555'))
            fig.layout[avg_plot_xaxis].tickvals = [1, 3, 5, 7]
            fig.layout[avg_plot_xaxis].ticktext = ['R1', 'R3', 'R5', 'R7']

        fig.update_xaxes(
            gridcolor='#E5E5E5', zerolinecolor='#E5E5E5', tickfont=dict(size=10, color='#555'),
            showspikes=True, spikemode='across', spikesnap='cursor', spikethickness=1,
            spikecolor='#888888', spikedash='dot'
        )
        n_xaxes = n_designators + 1
        for i in range(1, n_xaxes + 1):
            xaxis_name = "xaxis" if i == 1 else f"xaxis{i}"
            if xaxis_name in fig.layout:
                fig.layout[xaxis_name].showticklabels = True
        for i in range(2, n_xaxes + 1):
            xaxis_name = f"xaxis{i}"
            if xaxis_name in fig.layout:
                fig.layout[xaxis_name].matches = "x"
        fig.update_yaxes(
            tickfont=dict(size=10, color='#555'),
            showspikes=True, spikemode='across', spikesnap='cursor', spikethickness=1,
            spikecolor='#888888', spikedash='dot'
        )
        
        fig.update_layout(
            title=dict(
                text=f"<b>Belt Tracking Swing Arm Raw Data</b><br><span style='font-size:14px;color:#666'>{test_names[0]}</span>",
                font=dict(size=22, color='#2C3E50', family='Arial, sans-serif'), x=0.5, xanchor='center'
            ),
            height=300 * n_grid_rows + 550 + 420,
            template="plotly_white",
            paper_bgcolor='#FAFAFA',
            plot_bgcolor='white',
            font=dict(family='Arial, sans-serif'),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.28,
                xanchor="center",
                x=0.5,
                xref="paper",
                yref="paper",
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#E5E5E5',
                borderwidth=1,
                font=dict(size=10),
                itemwidth=40,
                tracegroupgap=0
            ),
            margin=dict(t=140, b=80, l=60, r=40),
            hoverlabel=dict(bgcolor='white', bordercolor='#E5E5E5', font=dict(size=12, color='#333')),
            hovermode='x'
        )
        
        for annotation in fig['layout']['annotations']:
            if (annotation['text'] in designators or
                    annotation['text'] == "All Designators Comparison" or
                    annotation['text'] == "Average Angle by Position"):
                annotation['font'] = dict(size=18, color="#2C3E50", family="Arial, sans-serif")
        
        # Ensure Average Angle traces never appear in legend
        for trace in fig.data:
            if getattr(trace, 'name', None) in ('Left', 'Right'):
                trace.showlegend = False
        
        return fig

    # Test order for Data Processing "Step 1: Averaging" grid (3x2)
    AVERAGING_GRID_TESTS = [
        "PAT Baseline - 300 FPM",
        "Standard Baseline - 300 FPM",
        "PAT Mistracked - Mining Left",
        "PAT Mistracked - Mining Right",
        "Standard Mistracked - Mining Left",
        "Standard Mistracked - Mining Right",
    ]

    def plot_averaging_step_grid(
        self,
        average_angles_by_position: dict[str, dict[str, list]],
    ) -> go.Figure:
        """
        Build a 3x2 grid of average angle by position plots for the Data Processing page.
        Row 1: PAT Baseline 300 FPM, Standard Baseline 300 FPM.
        Row 2: PAT Mistracked Mining Left, PAT Mistracked Mining Right.
        Row 3: Standard Mistracked Mining Left, Standard Mistracked Mining Right.
        """
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=[t for t in self.AVERAGING_GRID_TESTS],
            vertical_spacing=0.08,
            horizontal_spacing=0.06,
        )
        for idx, test_name in enumerate(self.AVERAGING_GRID_TESTS):
            row, col = (idx // 2) + 1, (idx % 2) + 1
            spline = average_angles_by_position.get(test_name, {})
            left_x = spline.get("left_x", [])
            left_y = spline.get("left_y", [])
            left_std = spline.get("left_std", [])
            right_x = spline.get("right_x", [])
            right_y = spline.get("right_y", [])
            right_std = spline.get("right_std", [])
            if left_x and left_y:
                fig.add_trace(
                    go.Scatter(
                        x=left_x,
                        y=left_y,
                        mode="lines+markers",
                        name="Left",
                        line=dict(shape="spline", width=2, color="#2E86AB"),
                        error_y=dict(
                            type="data",
                            array=[float(s) for s in left_std],
                            symmetric=True,
                            color="#2E86AB",
                            thickness=1,
                            width=3,
                        ),
                    ),
                    row=row,
                    col=col,
                )
            if right_x and right_y:
                fig.add_trace(
                    go.Scatter(
                        x=right_x,
                        y=right_y,
                        mode="lines+markers",
                        name="Right",
                        line=dict(shape="spline", width=2, color="#C73E1D"),
                        error_y=dict(
                            type="data",
                            array=[float(s) for s in right_std],
                            symmetric=True,
                            color="#C73E1D",
                            thickness=1,
                            width=3,
                        ),
                    ),
                    row=row,
                    col=col,
                )
        fig.update_xaxes(
            tickvals=[1, 3, 5, 7],
            ticktext=["R1", "R3", "R5", "R7"],
            title_text="",
        )
        fig.update_yaxes(title_text="Angle [deg]")
        fig.update_layout(
            height=720,
            width=None,
            showlegend=False,
            template="plotly_white",
            margin=dict(t=60, b=50, l=50, r=30),
            font=dict(size=12, family="Arial, sans-serif"),
            autosize=True,
        )
        for ann in fig.layout.annotations:
            ann.font = dict(size=18, color="#2C3E50", family="Arial, sans-serif")
        return fig

    # Step 2 delta plot: blue pair, green pair, red pair, orange pair (light then dark each)
    DELTA_PLOT_COLORS = [
        "#5E9FC4", "#1E5F8C",   # blue (slightly darker light), dark blue
        "#5BA85E", "#2E7D32",   # green (slightly darker light), dark green
        "#C95A5A", "#C62828",   # red (slightly darker light), dark red
        "#E09A38", "#E65100",   # orange (slightly darker light), dark orange
    ]

    def plot_delta_mistracking_baseline(
        self,
        delta_traces: list[dict],
    ) -> go.Figure:
        """
        Plot precomputed delta traces (from DataProcessor.delta_mistracking_baseline).
        Each trace dict has "name", "x", "y". Plotting only; no math.
        """
        fig = go.Figure()
        for i, trace in enumerate(delta_traces):
            name = trace.get("name", "")
            x = trace.get("x", [])
            y = trace.get("y", [])
            if not x or not y:
                continue
            color = self.DELTA_PLOT_COLORS[i] if i < len(self.DELTA_PLOT_COLORS) else self.COLORS[i % len(self.COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name=name,
                    line=dict(shape="spline", width=3, color=color),
                    marker=dict(size=10),
                )
            )
        fig.update_xaxes(
            tickvals=[1, 3, 5, 7],
            ticktext=["R1", "R3", "R5", "R7"],
            title_text="",
        )
        fig.update_yaxes(
            title_text="Angle delta [deg]",
            zeroline=True,
            zerolinecolor="#888",
            zerolinewidth=1,
            rangemode="tozero",
        )
        fig.update_layout(
            height=500,
            width=None,
            template="plotly_white",
            margin=dict(t=60, b=50, l=50, r=30),
            font=dict(size=12, family="Arial, sans-serif"),
            autosize=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01, font=dict(size=11)),
            title=dict(text="Mistracking PAT vs Standard Belt Edge Comparison", font=dict(size=18, color="#2C3E50", family="Arial, sans-serif")),
            hoverlabel=dict(namelength=-1, font=dict(size=12)),
        )
        return fig

    # Step 3 centerline plot: 4 traces (PAT/Standard × Mistrack Left/Right)
    CENTERLINE_PLOT_COLORS = ["#2E86AB", "#2E7D32", "#C73E1D", "#E65100"]  # blue, green, red, orange

    def plot_centerline_comparison(self, centerline_traces: list[dict]) -> go.Figure:
        """
        Plot precomputed centerline traces (from DataProcessor.centerline_comparison).
        Each trace dict has "name", "x", "y". Four splines: PAT Mistracked Left/Right, Standard Mistracked Left/Right.
        """
        fig = go.Figure()
        for i, trace in enumerate(centerline_traces):
            name = trace.get("name", "")
            x = trace.get("x", [])
            y = trace.get("y", [])
            if not x or not y:
                continue
            color = (
                self.CENTERLINE_PLOT_COLORS[i]
                if i < len(self.CENTERLINE_PLOT_COLORS)
                else self.COLORS[i % len(self.COLORS)]
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name=name,
                    line=dict(shape="spline", width=3, color=color),
                    marker=dict(size=10),
                )
            )
        fig.update_xaxes(
            tickvals=[1, 3, 5, 7],
            ticktext=["R1", "R3", "R5", "R7"],
            title_text="",
        )
        fig.update_yaxes(
            title_text="Angle delta [deg]",
            zeroline=True,
            zerolinecolor="#888",
            zerolinewidth=1,
            rangemode="tozero",
        )
        fig.update_layout(
            height=450,
            width=None,
            template="plotly_white",
            margin=dict(t=60, b=50, l=50, r=30),
            font=dict(size=12, family="Arial, sans-serif"),
            autosize=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01, font=dict(size=11)),
            title=dict(text="Mistracking PAT vs Standard Centerline Comparison", font=dict(size=18, color="#2C3E50", family="Arial, sans-serif")),
            hoverlabel=dict(namelength=-1, font=dict(size=12)),
        )
        return fig

    # Step 4 Re-Tracking Summary: PAT (blue), Standard (red)
    RE_TRACKING_SUMMARY_PLOT_COLORS = ["#2E86AB", "#C73E1D"]  # blue, red

    def plot_re_tracking_summary(self, re_tracking_summary_traces: list[dict]) -> go.Figure:
        """
        Plot precomputed Re-Tracking Summary traces (from DataProcessor.re_tracking_summary).
        Two traces: PAT, Standard. Blue and red. Each includes data points and a trendline with equation.
        """
        fig = go.Figure()
        for i, trace in enumerate(re_tracking_summary_traces):
            name = trace.get("name", "")
            x = trace.get("x", [])
            y = trace.get("y", [])
            slope = trace.get("slope")
            intercept = trace.get("intercept")
            if not x or not y:
                continue
            color = (
                self.RE_TRACKING_SUMMARY_PLOT_COLORS[i]
                if i < len(self.RE_TRACKING_SUMMARY_PLOT_COLORS)
                else self.COLORS[i % len(self.COLORS)]
            )
            # Data trace
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name=name,
                    line=dict(shape="spline", width=3, color=color),
                    marker=dict(size=10),
                )
            )
            # Trendline (line of best fit) and equation in legend
            if slope is not None and intercept is not None:
                y_fit = [slope * xi + intercept for xi in x]
                eq_str = f"y = {slope:.3g}x {intercept:+.3g}"
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y_fit,
                        mode="lines",
                        name=f"{name} trend: {eq_str}",
                        line=dict(dash="dash", width=2, color=color),
                    )
                )
        fig.update_xaxes(
            tickvals=[0, 10, 20, 30],
            ticktext=["0", "10", "20", "30"],
            title_text="Distance from Mistrack Start (ft)",
        )
        fig.update_yaxes(
            title_text="Angle delta [deg]",
            zeroline=True,
            zerolinecolor="#888",
            zerolinewidth=1,
            rangemode="tozero",
        )
        fig.update_layout(
            height=450,
            width=None,
            template="plotly_white",
            margin=dict(t=60, b=50, l=50, r=30),
            font=dict(size=12, family="Arial, sans-serif"),
            autosize=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01, font=dict(size=11)),
            title=dict(text="Re-Tracking Summary", font=dict(size=18, color="#2C3E50", family="Arial, sans-serif")),
            hoverlabel=dict(namelength=-1, font=dict(size=12)),
        )
        return fig

    def generate_report(
        self,
        data: dict[str, pd.DataFrame],
        test_name: str,
        plot_type: str = "grid"
    ) -> Path:
        """Generate an HTML report for a single test."""
        output_path = self._generate_filename(test_name)
        if plot_type == "grid":
            fig = self.plot_grid(data, test_name)
        elif plot_type == "comparison":
            fig = self.plot_comparison(data, test_name)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}. Use 'grid' or 'comparison'.")
        fig.write_html(output_path)
        print(f"Report saved to: {output_path}")
        return output_path
    
    def _build_grafana_url(self, from_iso: str, to_iso: str) -> str:
        """Build Grafana dashboard URL for a given time range (ISO 8601)."""
        return GRAFANA_BASE_URL.replace("{from}", from_iso).replace("{to}", to_iso)

    def _build_per_test_updates(
        self, test_names: list[str], n_designators: int
    ) -> list[dict]:
        """Build visibility and title update for each test (same logic as Plotly dropdown buttons)."""
        updates = []
        for test_idx, test_name in enumerate(test_names):
            visibility = []
            for t_idx in range(len(test_names)):
                is_selected = t_idx == test_idx
                for _ in range(n_designators):
                    visibility.append(is_selected)
                for _ in range(n_designators):
                    visibility.append(is_selected)
                for _ in range(2):
                    visibility.append(is_selected)
                for _ in range(1):
                    visibility.append(is_selected)
            title = dict(
                text=f"<b>Belt Tracking Swing Arm Raw Data</b><br><span style='font-size:14px;color:#666'>{test_name}</span>",
                font=dict(size=22, color='#2C3E50', family='Arial, sans-serif'),
                x=0.5,
                xanchor='center',
            )
            updates.append({"visible": visibility, "title": title})
        return updates

    def _inject_grafana_links(
        self,
        output_path: Path,
        test_names: list[str],
        time_ranges: dict,
        n_designators: int,
        *,
        averaging_plots_html: str = "",
        delta_plots_html: str = "",
        centerline_plots_html: str = "",
        re_tracking_summary_plots_html: str = "",
    ) -> None:
        """
        Inject nav, Report/Data Processing pages, and optionally one test dropdown
        and one Grafana Data link. If time_ranges and test_names are present,
        the dropdown and link are added; otherwise only nav and pages are injected.
        """
        with open(output_path, "r", encoding="utf-8") as f:
            html = f.read()

        report_page_extra = ""
        if test_names and time_ranges:
            updates = self._build_per_test_updates(test_names, n_designators)
            tr0 = time_ranges.get(test_names[0], {})
            initial_url = self._build_grafana_url(
                tr0.get("from", ""), tr0.get("to", "")
            ) if tr0 else "#"

            payload = {
                "testNames": test_names,
                "timeRanges": time_ranges,
                "baseUrl": GRAFANA_BASE_URL,
                "updates": updates,
            }
            data_script = (
                "<script type=\"text/javascript\">\n"
                "window.REPORT_DATA = " + json.dumps(payload) + ";\n"
                "</script>\n"
            )
            options_html = "".join(
                f'<option value="{i}">{html_module.escape(name)}</option>'
                for i, name in enumerate(test_names)
            )
            link_block = (
                '<div id="report-controls" style="'
                'position:absolute;top:60px;left:12px;z-index:20;'
                'font-family:Arial,sans-serif;font-size:12px;'
                'background-color:#F0F0F0;border:1px solid #CCCCCC;border-radius:4px;'
                'padding:8px 10px;box-shadow:0 1px 2px rgba(0,0,0,0.05);">'
                '<label for="test-select" style="color:#2C3E50;font-weight:600;margin-right:6px;">Test</label>'
                f'<select id="test-select" style="'
                'background:#fff;border:1px solid #CCCCCC;border-radius:3px;'
                'color:#333;font-size:12px;padding:4px 6px;min-width:180px;'
                'font-family:Arial,sans-serif;cursor:pointer;">'
                f'{options_html}</select>'
                '<span style="margin:0 8px;color:#CCCCCC;">|</span>'
                f'<a id="grafana-link" href="{html_module.escape(initial_url)}" target="_blank" rel="noopener" style="'
                'color:#2E86AB;text-decoration:none;font-weight:500;">Grafana Data</a>'
                "</div>\n"
                "<style>#grafana-link:hover{text-decoration:underline;color:#2C3E50;}</style>\n"
            )
            report_page_extra = data_script + link_block

        update_script = ""
        if test_names and time_ranges:
            update_script = """
<script type="text/javascript">
(function() {
  var d = window.REPORT_DATA;
  if (!d || !d.updates || !d.testNames) return;
  var select = document.getElementById('test-select');
  var link = document.getElementById('grafana-link');
  if (!select || !link) return;
  function setTest(index) {
    var u = d.updates[index];
    var tr = d.timeRanges[d.testNames[index]];
    if (u) {
      var gd = document.querySelector('.plotly-graph-div');
      if (gd && typeof Plotly !== 'undefined') {
        Plotly.restyle(gd, {visible: u.visible});
        Plotly.relayout(gd, {title: u.title});
      }
    }
    if (tr && tr.from && tr.to) {
      link.href = d.baseUrl.replace('{from}', tr.from).replace('{to}', tr.to);
    }
  }
  select.addEventListener('change', function() {
    setTest(parseInt(select.value, 10));
  });
})();
</script>
"""

        nav_and_pages_css = """
<style type="text/css">
  .site-nav { background:#2C3E50; padding:14px 24px; font-family:Arial,sans-serif; }
  .site-nav a {
    color:#ecf0f1; text-decoration:none; margin-right:12px; font-weight:600;
    padding:10px 20px; border-radius:6px; cursor:pointer;
    border:1px solid rgba(255,255,255,0.2); background:rgba(255,255,255,0.05);
    transition:background 0.2s, color 0.2s, border-color 0.2s;
  }
  .site-nav a:hover {
    color:#fff; background:rgba(255,255,255,0.15); border-color:rgba(255,255,255,0.35);
  }
  .site-nav a.active {
    color:#fff; background:#2E86AB; border-color:#2E86AB;
  }
  .site-nav a.active:hover { background:#2573a0; border-color:#2573a0; }
  .page { display:none; font-family:Arial,sans-serif; }
  .page.active { display:block; }
  #report-page { width:100%; max-width:100%; padding:0; margin:0; box-sizing:border-box; }
  .data-processing-page { width:100%; max-width:100%; padding:24px; margin:0; box-sizing:border-box; }
  .data-processing-page h1 { color:#2C3E50; font-size:28px; margin-bottom:16px; }
  .data-processing-page h2 { color:#555; font-size:20px; margin:24px 0 12px; }
  .data-processing-page p { color:#333; line-height:1.6; margin-bottom:16px; }
  .placeholder-box { background:#f5f5f5; border:1px dashed #ccc; color:#888; padding:40px; text-align:center; margin:16px 0; border-radius:4px; }
  .placeholder-img { min-height:200px; }
  .placeholder-plot { min-height:300px; }
  #data-processing-page { min-width: 100%; }
  .data-processing-page .averaging-plots-wrapper { width:100% !important; min-width:100% !important; max-width:100% !important; box-sizing:border-box; }
  .data-processing-page .averaging-plots-wrapper .plotly-graph-div { width:100% !important; min-width:100% !important; max-width:100% !important; }
  .data-processing-page .js-plotly-plot { width:100% !important; }
  .data-processing-page .plotly-container { width:100% !important; }
</style>
"""
        nav_html = (
            '<nav class="site-nav">'
            '<a href="#" id="nav-report" class="active">Raw Data</a>'
            '<a href="#" id="nav-data-processing">Data Processing</a>'
            '</nav>'
        )
        data_processing_page = """
<div id="data-processing-page" class="page data-processing-page">
  <h1>Data Processing</h1>
  <p>This page will step through the data processing pipeline and methodology.</p>
  <h2>Step 1: Averaging</h2>
  <p>Average the angle value over the entire run duration by position (R1, R3, R5, R7). We will be using the Baseline - 300FPM and mistracked runs.<br>
  This is done to reduce the noise and variability in the data and extract a single variable for 5-15 minutes of data collection per trial.</p>
  <!-- AVERAGING_PLOTS -->
  <h2>Step 2: Delta between Mistracking and Baseline</h2>
  <p>Absolute difference (mistrack run minus baseline 300 FPM) by sensor position for PAT and Standard, left and right angles.<br>
  This step is used to eliminate variability in sensor installation and belt hardware changes. Comparing mistracking relative to a baseline ensures the angles we are observing are not due to sensor installation issues or belt hardware changes.</p>
  <!-- DELTA_PLOT -->
  <h2>Step 3: Centerline Comparison</h2>
  <p>Centerline delta at each position: average of the left and right delta traces from Step 2, (left + right) / 2, for each mistracked run.<br>
  This can be considered a proxy for belt centerline position, since it is the average of the left and right angles.</p>
  <!-- CENTERLINE_PLOT -->
  <h2>Step 4: Re-Tracking Summary</h2>
  <p>Re-tracking summary at each position: average of Step 3's PAT Mistracked Left and PAT Mistracked Right (PAT), and average of Standard Mistracked Left and Standard Mistracked Right (Standard).</p>
  <!-- RE_TRACKING_SUMMARY_PLOT -->
  <p>The figure of merit for belt tracking testing is the slope of the trendlines shown above. This slope shows how quickly the belt returns to the centerline after being mistracked, in units of deg/ft.<br>
  The greater the slope, the quicker the belt returns to the centerline after being mistracked, and the tighter the minimum possible turn radius can be.</p>
  <div class="placeholder-box" style="overflow-x:auto;">
    <table style="width:100%; border-collapse:collapse; font-size:14px;">
      <thead>
        <tr style="background:#2C3E50; color:#ecf0f1;">
          <th style="border:1px solid #ddd; padding:10px; text-align:left;">Placeholder column 1</th>
          <th style="border:1px solid #ddd; padding:10px; text-align:left;">Placeholder column 2</th>
          <th style="border:1px solid #ddd; padding:10px; text-align:left;">Placeholder column 3</th>
        </tr>
      </thead>
      <tbody>
        <tr><td style="border:1px solid #ddd; padding:8px;">—</td><td style="border:1px solid #ddd; padding:8px;">—</td><td style="border:1px solid #ddd; padding:8px;">—</td></tr>
        <tr style="background:#f9f9f9;"><td style="border:1px solid #ddd; padding:8px;">—</td><td style="border:1px solid #ddd; padding:8px;">—</td><td style="border:1px solid #ddd; padding:8px;">—</td></tr>
      </tbody>
    </table>
    <p style="margin:8px 0 0; color:#888; font-size:12px;">Placeholder table — replace with re-tracking summary data as needed.</p>
  </div>
  <h2>Image placeholder</h2>
  <div class="placeholder-box placeholder-img">Image placeholder — add diagram or screenshot here</div>
  <h2>Processing pipeline</h2>
  <p>Placeholder for pipeline description and optional Plotly graph below.</p>
  <div class="placeholder-box placeholder-plot">Plotly graph placeholder — add chart or visualization here</div>
  <h2>Additional section</h2>
  <p>Placeholder for more text, figures, or tables as needed.</p>
</div>
"""
        data_processing_page = data_processing_page.replace(
            "<!-- AVERAGING_PLOTS -->", averaging_plots_html
        ).replace("<!-- DELTA_PLOT -->", delta_plots_html).replace(
            "<!-- CENTERLINE_PLOT -->", centerline_plots_html
        ).replace("<!-- RE_TRACKING_SUMMARY_PLOT -->", re_tracking_summary_plots_html)
        page_switch_script = """
<script type="text/javascript">
(function() {
  var navReport = document.getElementById('nav-report');
  var navDataProcessing = document.getElementById('nav-data-processing');
  var reportPage = document.getElementById('report-page');
  var dataProcessingPage = document.getElementById('data-processing-page');
  if (!navReport || !navDataProcessing || !reportPage || !dataProcessingPage) return;
  function showReport() {
    reportPage.classList.add('active');
    dataProcessingPage.classList.remove('active');
    navReport.classList.add('active');
    navDataProcessing.classList.remove('active');
  }
  function showDataProcessing() {
    reportPage.classList.remove('active');
    dataProcessingPage.classList.add('active');
    navReport.classList.remove('active');
    navDataProcessing.classList.add('active');
    var graphs = document.querySelectorAll('#data-processing-page .plotly-graph-div');
    if (typeof Plotly !== 'undefined') {
      for (var i = 0; i < graphs.length; i++) { Plotly.Plots.resize(graphs[i]); }
    }
  }
  navReport.addEventListener('click', function(e) { e.preventDefault(); showReport(); });
  navDataProcessing.addEventListener('click', function(e) { e.preventDefault(); showDataProcessing(); });
})();
</script>
"""

        html = html.replace(
            "<body>",
            "<body>\n"
            + nav_and_pages_css
            + nav_html
            + '\n<div id="report-page" class="page active">\n'
            + report_page_extra
            + update_script,
            1,
        )
        html = html.replace(
            "</body>",
            "</div>\n" + data_processing_page + "\n" + page_switch_script + "\n</body>",
            1,
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    def generate_multi_test_report(
        self,
        all_test_data: dict[str, dict[str, pd.DataFrame]],
        average_angles_by_position: dict[str, dict[str, list]] | None = None,
        delta_mistracking_baseline: list[dict] | None = None,
        centerline_comparison: list[dict] | None = None,
        re_tracking_summary: list[dict] | None = None,
        report_name: str = "report",
        time_ranges: dict = None,
    ) -> Path:
        """Generate an HTML report with dropdown to switch between multiple tests. Overwrites existing file.
        average_angles_by_position: From DataProcessor.process()['average_angles_by_position'].
        delta_mistracking_baseline: From DataProcessor.process()['delta_mistracking_baseline'] (Step 2 plot data).
        centerline_comparison: From DataProcessor.process()['centerline_comparison'] (Step 3 plot data).
        re_tracking_summary: From DataProcessor.process()['re_tracking_summary'] (Step 4 plot data).
        If time_ranges is provided, injects one test dropdown and one Grafana Data link (one link per test view)."""
        output_path = self.output_dir / "index.html"
        test_names = list(all_test_data.keys())
        fig = self.plot_multi_test_grid(
            all_test_data, average_angles_by_position=average_angles_by_position
        )
        fig.write_html(output_path)

        averaging_plots_html = ""
        delta_plots_html = ""
        centerline_plots_html = ""
        re_tracking_summary_plots_html = ""
        if average_angles_by_position:
            avg_fig = self.plot_averaging_step_grid(average_angles_by_position)
            plot_div = avg_fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                config={"responsive": True},
                default_width="100%",
                default_height="720px",
            )
            averaging_plots_html = (
                '<div id="averaging-plots-wrapper" class="averaging-plots-wrapper" '
                'style="width:100%; min-width:100%; max-width:100%;">'
                + plot_div
                + "</div>"
            )
        if delta_mistracking_baseline:
            delta_fig = self.plot_delta_mistracking_baseline(delta_mistracking_baseline)
            delta_div = delta_fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                config={"responsive": True},
                default_width="100%",
                default_height="500px",
            )
            delta_plots_html = (
                '<div class="averaging-plots-wrapper" style="width:100%; min-width:100%; max-width:100%;">'
                + delta_div
                + "</div>"
            )
        if centerline_comparison:
            centerline_fig = self.plot_centerline_comparison(centerline_comparison)
            centerline_div = centerline_fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                config={"responsive": True},
                default_width="100%",
                default_height="450px",
            )
            centerline_plots_html = (
                '<div class="averaging-plots-wrapper" style="width:100%; min-width:100%; max-width:100%;">'
                + centerline_div
                + "</div>"
            )
        if re_tracking_summary:
            re_tracking_summary_fig = self.plot_re_tracking_summary(re_tracking_summary)
            re_tracking_summary_div = re_tracking_summary_fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                config={"responsive": True},
                default_width="100%",
                default_height="450px",
            )
            re_tracking_summary_plots_html = (
                '<div class="averaging-plots-wrapper" style="width:100%; min-width:100%; max-width:100%;">'
                + re_tracking_summary_div
                + "</div>"
            )

        designators = (
            list(next(iter(all_test_data.values())).keys())
            if all_test_data
            else []
        )
        n_designators = len(designators)
        self._inject_grafana_links(
            output_path,
            test_names,
            time_ranges or {},
            n_designators,
            averaging_plots_html=averaging_plots_html,
            delta_plots_html=delta_plots_html,
            centerline_plots_html=centerline_plots_html,
            re_tracking_summary_plots_html=re_tracking_summary_plots_html,
        )
        print(f"Report saved to: {output_path}")
        return output_path
    
    def show(self, fig: go.Figure) -> None:
        """Display a figure in the browser."""
        fig.show()


if __name__ == "__main__":
    print("Belt Tracking Swing Arm - Visualization Module")
    print("=" * 50)
    print("\nExample: from data_collection import DataCollector; from visualization import Visualizer")
    print("  collector = DataCollector(); visualizer = Visualizer()")
    print("  all_data = collector.collect_all_tests()")
    print("  visualizer.generate_multi_test_report(all_data)")
