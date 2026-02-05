"""
Visualization Module for Belt Tracking Swing Arm Analysis
Uses Plotly to generate interactive HTML reports
"""

import html as html_module
import json
import re

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
    
    def _compute_average_angles_splines(
        self,
        test_data: dict[str, pd.DataFrame],
        designators: list[str]
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """
        Compute average angle per designator and split into left/right for spline plot.
        Returns (left_x, left_y, right_x, right_y) where x is roller position (1, 3, 5, 7).
        """
        left_positions = []
        left_avgs = []
        right_positions = []
        right_avgs = []
        
        for designator in designators:
            df = test_data.get(designator, pd.DataFrame())
            value_col = self._get_value_column(df)
            if df.empty or not value_col:
                continue
            avg = df[value_col].mean()
            match = re.match(r'R(\d+)', designator, re.IGNORECASE)
            pos = int(match.group(1)) if match else 0
            if 'LEFT' in designator.upper():
                left_positions.append(float(pos))
                left_avgs.append(avg)
            elif 'RIGHT' in designator.upper():
                right_positions.append(float(pos))
                right_avgs.append(avg)
        
        if left_positions:
            sorted_lr = sorted(zip(left_positions, left_avgs))
            left_positions, left_avgs = [t[0] for t in sorted_lr], [t[1] for t in sorted_lr]
        if right_positions:
            sorted_lr = sorted(zip(right_positions, right_avgs))
            right_positions, right_avgs = [t[0] for t in sorted_lr], [t[1] for t in sorted_lr]
        
        return (left_positions, left_avgs, right_positions, right_avgs)
    
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
        title: str = "Multi-Test Analysis"
    ) -> go.Figure:
        """
        Create a subplot grid with dropdown to switch between tests.
        Includes comparison plot and average angle by position (two splines).
        """
        test_names = list(all_test_data.keys())
        if not test_names:
            raise ValueError("No test data provided")
        
        first_test_data = all_test_data[test_names[0]]
        designators = list(first_test_data.keys())
        n_designators = len(designators)
        n_grid_rows = (n_designators + 1) // 2
        n_total_rows = n_grid_rows + 2
        
        subplot_titles = designators + ["All Designators Comparison", "Average Angle by Position"]
        row_heights = [1] * n_grid_rows + [1.5, 1.0]
        specs = [[{}, {}] for _ in range(n_grid_rows)]
        specs.append([{"colspan": 2}, None])
        specs.append([{"colspan": 2}, None])
        
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
            
            comparison_row = n_total_rows - 1
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
            
            left_x, left_y, right_x, right_y = self._compute_average_angles_splines(test_data, designators)
            if left_x and left_y:
                fig.add_trace(
                    go.Scatter(
                        x=left_x, y=left_y, mode='lines+markers', name='Left',
                        showlegend=False, visible=visible,
                        line=dict(shape='spline', width=2, color='#2E86AB'),
                        hovertemplate='R%{x:.0f} Left: %{y:.2f} deg<extra></extra>'
                    ),
                    row=n_total_rows, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(x=[], y=[], mode='lines', name='Left', showlegend=False, visible=visible),
                    row=n_total_rows, col=1
                )
            if right_x and right_y:
                fig.add_trace(
                    go.Scatter(
                        x=right_x, y=right_y, mode='lines+markers', name='Right',
                        showlegend=False, visible=visible,
                        line=dict(shape='spline', width=2, color='#C73E1D'),
                        hovertemplate='R%{x:.0f} Right: %{y:.2f} deg<extra></extra>'
                    ),
                    row=n_total_rows, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(x=[], y=[], mode='lines', name='Right', showlegend=False, visible=visible),
                    row=n_total_rows, col=1
                )
        
        for i in range(n_designators):
            yaxis_name = f"yaxis{i + 1}" if i > 0 else "yaxis"
            fig.update_layout(**{yaxis_name: dict(
                title=dict(text="Angle [deg]", font=dict(size=11, color='#555')),
                gridcolor='#E5E5E5', zerolinecolor='#E5E5E5'
            )})
        comparison_yaxis = f"yaxis{n_designators + 1}"
        fig.update_layout(**{comparison_yaxis: dict(
            title=dict(text="Angle [deg]", font=dict(size=11, color='#555')),
            gridcolor='#E5E5E5', zerolinecolor='#E5E5E5'
        )})
        avg_plot_yaxis = f"yaxis{n_designators + 2}"
        avg_plot_xaxis = f"xaxis{n_designators + 2}"
        fig.update_layout(**{avg_plot_yaxis: dict(
            title=dict(text="Angle [deg]", font=dict(size=11, color='#555')),
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
                text=f"<b>Belt Tracking Swing Arm Analysis</b><br><span style='font-size:14px;color:#666'>{test_names[0]}</span>",
                font=dict(size=22, color='#2C3E50', family='Arial, sans-serif'), x=0.5, xanchor='center'
            ),
            height=300 * n_grid_rows + 550,
            template="plotly_white",
            paper_bgcolor='#FAFAFA',
            plot_bgcolor='white',
            font=dict(family='Arial, sans-serif'),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.15,
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
                    annotation['text'] == "Average Angular Mistracking by Position"):
                annotation['font'] = dict(size=13, color='#2C3E50', family='Arial, sans-serif')
        
        # Ensure Average Angle traces never appear in legend
        for trace in fig.data:
            if getattr(trace, 'name', None) in ('Left', 'Right'):
                trace.showlegend = False
        
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
            title = dict(
                text=f"<b>Belt Tracking Swing Arm Analysis</b><br><span style='font-size:14px;color:#666'>{test_name}</span>",
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
    ) -> None:
        """
        Inject one test dropdown and one Grafana Data link. Changing the dropdown
        updates the plot (visibility + title) and the link to that test's time range.
        """
        with open(output_path, "r", encoding="utf-8") as f:
            html = f.read()

        updates = self._build_per_test_updates(test_names, n_designators)
        tr0 = time_ranges.get(test_names[0], {})
        initial_url = self._build_grafana_url(
            tr0.get("from", ""), tr0.get("to", "")
        ) if tr0 else "#"

        # Data for JS: test names, time ranges, base URL, and per-test plot updates
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

        html = html.replace(
            "<body>",
            "<body>\n" + data_script + link_block + update_script,
            1,
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    def generate_multi_test_report(
        self,
        all_test_data: dict[str, dict[str, pd.DataFrame]],
        report_name: str = "report",
        time_ranges: dict = None,
    ) -> Path:
        """Generate an HTML report with dropdown to switch between multiple tests. Overwrites existing file.
        If time_ranges is provided, injects one test dropdown and one Grafana Data link (one link per test view)."""
        output_path = self.output_dir / f"{report_name}.html"
        test_names = list(all_test_data.keys())
        fig = self.plot_multi_test_grid(all_test_data)
        fig.write_html(output_path)
        if time_ranges and test_names:
            designators = list(next(iter(all_test_data.values())).keys())
            n_designators = len(designators)
            self._inject_grafana_links(
                output_path, test_names, time_ranges, n_designators
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
