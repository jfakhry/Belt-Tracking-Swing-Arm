"""
Visualization Module for Belt Tracking Swing Arm Analysis
Uses Plotly to generate interactive HTML reports
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
from datetime import datetime


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
            output_dir: Directory for HTML reports. Defaults to 'reports/' folder.
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / "reports"
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
        """
        Create a simple line plot for a single designator.
        
        Args:
            df: DataFrame with 'time' and value columns
            designator: Name of the designator for the title
        
        Returns:
            Plotly Figure object
        """
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
        """
        Create a subplot grid with all designators.
        
        Args:
            data: Dictionary mapping designator names to DataFrames
            title: Title for the figure
        
        Returns:
            Plotly Figure object
        """
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
            row = (i // 2) + 1
            col = (i % 2) + 1
            value_col = self._get_value_column(df)
            
            if not df.empty and value_col:
                fig.add_trace(
                    go.Scatter(
                        x=df['time'],
                        y=df[value_col],
                        mode='lines',
                        name=designator,
                        showlegend=False
                    ),
                    row=row,
                    col=col
                )
        
        # Add y-axis labels to all subplots
        for i in range(n_designators):
            yaxis_name = f"yaxis{i + 1}" if i > 0 else "yaxis"
            fig.update_layout(**{yaxis_name: dict(title="Angle [deg]")})
        
        fig.update_layout(
            title=f"Belt Tracking Swing Arm Analysis - {title}",
            height=300 * n_rows,
            template="plotly_white"
        )
        
        return fig
    
    def plot_comparison(self, data: dict[str, pd.DataFrame], title: str = "Test") -> go.Figure:
        """
        Create an overlay plot comparing all designators.
        
        Args:
            data: Dictionary mapping designator names to DataFrames
            title: Title for the figure
        
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        for designator, df in data.items():
            value_col = self._get_value_column(df)
            
            if not df.empty and value_col:
                fig.add_trace(go.Scatter(
                    x=df['time'],
                    y=df[value_col],
                    mode='lines',
                    name=designator
                ))
        
        fig.update_layout(
            title=f"All Designators Comparison - {title}",
            xaxis_title="Time",
            yaxis_title="Angle [deg]",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_multi_test_grid(
        self,
        all_test_data: dict[str, dict[str, pd.DataFrame]],
        title: str = "Multi-Test Analysis"
    ) -> go.Figure:
        """
        Create a subplot grid with dropdown to switch between tests.
        Includes a comparison plot at the bottom with all designators overlaid.
        
        Args:
            all_test_data: Dictionary mapping test names to their data dictionaries
                           e.g., {"test1": {"R1_LEFT_ANGLE": df, ...}, "test2": {...}}
            title: Title for the figure
        
        Returns:
            Plotly Figure object with dropdown selector
        """
        test_names = list(all_test_data.keys())
        if not test_names:
            raise ValueError("No test data provided")
        
        # Use first test to determine grid structure
        first_test_data = all_test_data[test_names[0]]
        designators = list(first_test_data.keys())
        n_designators = len(designators)
        n_grid_rows = (n_designators + 1) // 2
        
        # Add one more row for the comparison plot (spanning both columns)
        n_total_rows = n_grid_rows + 1
        
        # Create subplot titles (individual plots + comparison)
        subplot_titles = designators + ["All Designators Comparison"]
        
        # Define row heights: equal for grid rows, larger for comparison
        row_heights = [1] * n_grid_rows + [1.5]
        
        # Create specs: 2 columns for grid rows, 1 spanning column for comparison
        specs = [[{}, {}] for _ in range(n_grid_rows)]
        specs.append([{"colspan": 2}, None])  # Comparison row spans both columns
        
        fig = make_subplots(
            rows=n_total_rows,
            cols=2,
            subplot_titles=subplot_titles,
            vertical_spacing=0.06,
            horizontal_spacing=0.1,
            row_heights=row_heights,
            specs=specs
        )
        
        # Track trace indices for visibility toggling
        # Each test has: n_designators individual traces + n_designators comparison traces
        traces_per_test = n_designators * 2
        
        # Add traces for ALL tests (we'll toggle visibility)
        for test_idx, (test_name, test_data) in enumerate(all_test_data.items()):
            # First test is visible, others are hidden
            visible = (test_idx == 0)
            
            # Add individual subplot traces
            for i, designator in enumerate(designators):
                row = (i // 2) + 1
                col = (i % 2) + 1
                color = self.COLORS[i % len(self.COLORS)]
                
                df = test_data.get(designator, pd.DataFrame())
                value_col = self._get_value_column(df)
                
                if not df.empty and value_col:
                    fig.add_trace(
                        go.Scatter(
                            x=df['time'],
                            y=df[value_col],
                            mode='lines',
                            name=f"{designator} ({test_name})",
                            showlegend=False,
                            visible=visible,
                            line=dict(color=color, width=1.5),
                            hovertemplate='%{y:.2f} deg<extra></extra>'
                        ),
                        row=row,
                        col=col
                    )
                else:
                    # Add empty trace to maintain indexing
                    fig.add_trace(
                        go.Scatter(
                            x=[],
                            y=[],
                            mode='lines',
                            name=f"{designator} ({test_name})",
                            showlegend=False,
                            visible=visible,
                            line=dict(color=color, width=1.5)
                        ),
                        row=row,
                        col=col
                    )
            
            # Add comparison plot traces (all designators overlaid)
            for i, designator in enumerate(designators):
                df = test_data.get(designator, pd.DataFrame())
                value_col = self._get_value_column(df)
                color = self.COLORS[i % len(self.COLORS)]
                
                if not df.empty and value_col:
                    fig.add_trace(
                        go.Scatter(
                            x=df['time'],
                            y=df[value_col],
                            mode='lines',
                            name=designator,
                            showlegend=True,  # Always show legend (invisible traces won't appear)
                            visible=visible,
                            legendgroup=designator,
                            line=dict(color=color, width=1.5),
                            hovertemplate='%{y:.2f} deg<extra></extra>'
                        ),
                        row=n_total_rows,
                        col=1
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=[],
                            y=[],
                            mode='lines',
                            name=designator,
                            showlegend=True,
                            visible=visible,
                            legendgroup=designator,
                            line=dict(color=color, width=1.5)
                        ),
                        row=n_total_rows,
                        col=1
                    )
        
        # Create dropdown buttons
        buttons = []
        for test_idx, test_name in enumerate(test_names):
            # Create visibility array: True for this test's traces, False for others
            visibility = []
            showlegend_updates = []
            
            for t_idx in range(len(test_names)):
                is_selected = (t_idx == test_idx)
                # Individual plot traces (no legend)
                for _ in range(n_designators):
                    visibility.append(is_selected)
                # Comparison plot traces (legend only for selected)
                for _ in range(n_designators):
                    visibility.append(is_selected)
            
            buttons.append(dict(
                label=test_name,
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": dict(
                        text=f"<b>Belt Tracking Swing Arm Analysis</b><br><span style='font-size:14px;color:#666'>{test_name}</span>",
                        font=dict(size=22, color='#2C3E50', family='Arial, sans-serif'),
                        x=0.5,
                        xanchor='center'
                    )}
                ]
            ))
        
        # Add y-axis labels to individual subplots
        for i in range(n_designators):
            yaxis_name = f"yaxis{i + 1}" if i > 0 else "yaxis"
            fig.update_layout(**{yaxis_name: dict(
                title=dict(text="Angle [deg]", font=dict(size=11, color='#555')),
                gridcolor='#E5E5E5',
                zerolinecolor='#E5E5E5'
            )})
        
        # Add y-axis label to comparison plot
        comparison_yaxis = f"yaxis{n_designators + 1}"
        fig.update_layout(**{comparison_yaxis: dict(
            title=dict(text="Angle [deg]", font=dict(size=11, color='#555')),
            gridcolor='#E5E5E5',
            zerolinecolor='#E5E5E5'
        )})
        
        # Update all x-axes with crosshair spike lines
        fig.update_xaxes(
            gridcolor='#E5E5E5',
            zerolinecolor='#E5E5E5',
            tickfont=dict(size=10, color='#555'),
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikethickness=1,
            spikecolor='#888888',
            spikedash='dot'
        )
        
        # Update all y-axes with crosshair spike lines
        fig.update_yaxes(
            tickfont=dict(size=10, color='#555'),
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikethickness=1,
            spikecolor='#888888',
            spikedash='dot'
        )
        
        fig.update_layout(
            title=dict(
                text=f"<b>Belt Tracking Swing Arm Analysis</b><br><span style='font-size:14px;color:#666'>{test_names[0]}</span>",
                font=dict(size=22, color='#2C3E50', family='Arial, sans-serif'),
                x=0.5,
                xanchor='center'
            ),
            height=300 * n_grid_rows + 450,  # Extra height for comparison plot
            template="plotly_white",
            paper_bgcolor='#FAFAFA',
            plot_bgcolor='white',
            font=dict(family='Arial, sans-serif'),
            updatemenus=[
                dict(
                    active=0,
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.12,
                    yanchor="top",
                    bgcolor="#F0F0F0",
                    bordercolor="#CCCCCC",
                    borderwidth=1,
                    font=dict(size=12, color='#333333'),
                    pad=dict(l=5, r=5, t=3, b=3)
                )
            ],
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.03,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#E5E5E5',
                borderwidth=1,
                font=dict(size=11)
            ),
            margin=dict(t=140, b=80, l=60, r=40),
            hoverlabel=dict(
                bgcolor='white',
                bordercolor='#E5E5E5',
                font=dict(size=12, color='#333')
            ),
            hovermode='x'
        )
        
        # Update subplot title annotations to be styled
        for annotation in fig['layout']['annotations']:
            if annotation['text'] in designators or annotation['text'] == "All Designators Comparison":
                annotation['font'] = dict(size=13, color='#2C3E50', family='Arial, sans-serif')
        
        return fig
    
    def generate_report(
        self,
        data: dict[str, pd.DataFrame],
        test_name: str,
        plot_type: str = "grid"
    ) -> Path:
        """
        Generate an HTML report for a single test.
        
        Args:
            data: Dictionary mapping designator names to DataFrames
            test_name: Name of the test
            plot_type: Type of plot ('grid' or 'comparison')
        
        Returns:
            Path to the generated HTML file
        """
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
    
    def generate_multi_test_report(
        self,
        all_test_data: dict[str, dict[str, pd.DataFrame]],
        report_name: str = "Example Report"
    ) -> Path:
        """
        Generate an HTML report with dropdown to switch between multiple tests.
        Overwrites existing report file.
        
        Args:
            all_test_data: Dictionary mapping test names to their data dictionaries
            report_name: Name for the report file (without extension)
        
        Returns:
            Path to the generated HTML file
        """
        output_path = self.output_dir / f"{report_name}.html"
        
        fig = self.plot_multi_test_grid(all_test_data)
        fig.write_html(output_path)
        print(f"Report saved to: {output_path}")
        
        return output_path
    
    def show(self, fig: go.Figure) -> None:
        """Display a figure in the browser."""
        fig.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Belt Tracking Swing Arm - Visualization Module")
    print("=" * 50)
    print("\nThis module provides the Visualizer class.")
    print("Example usage for single test:")
    print()
    print("  from data_collection import DataCollector")
    print("  from visualization import Visualizer")
    print()
    print("  collector = DataCollector()")
    print("  visualizer = Visualizer()")
    print()
    print("  data = collector.collect_test('my_test')")
    print("  report_path = visualizer.generate_report(data, 'my_test')")
    print()
    print("Example usage for multiple tests with dropdown:")
    print()
    print("  all_data = {}")
    print("  for test_name in collector.test_config.list():")
    print("      all_data[test_name] = collector.collect_test(test_name)")
    print("  report_path = visualizer.generate_multi_test_report(all_data)")
