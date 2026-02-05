"""
Belt Tracking Swing Arm Analysis - Main Entry Point

Collects data for all configured tests and generates an interactive HTML report.
Uses local caching - only new tests are fetched from InfluxDB.
"""

from data_collection import DataCollector
from visualization import Visualizer


def main(force_refresh: bool = False):
    """
    Run the analysis pipeline.
    
    Args:
        force_refresh: If True, ignore cache and fetch fresh data for all tests
    """
    print("=" * 60)
    print("Belt Tracking Swing Arm Analysis")
    print("=" * 60)
    
    # Initialize components
    collector = DataCollector()
    visualizer = Visualizer()
    
    # Get list of configured tests
    test_names = collector.test_config.list()
    cached_tests = collector.cache.list_cached()
    
    if not test_names:
        print("\nNo tests configured in tests_config.json")
        print("Add a test using:")
        print("  collector.test_config.add('test_name', 'YYYY-MM-DDTHH:MM:SS.sssZ', 'YYYY-MM-DDTHH:MM:SS.sssZ')")
        return
    
    print(f"\nConfigured tests: {test_names}")
    print(f"Cached tests: {cached_tests}")
    
    # Identify new tests that need fetching
    new_tests = [t for t in test_names if t not in cached_tests]
    if new_tests and not force_refresh:
        print(f"New tests to fetch: {new_tests}")
    elif force_refresh:
        print("Force refresh enabled - fetching all tests")
    else:
        print("All tests cached - no InfluxDB queries needed")
    
    # Collect data for all tests (uses cache where available)
    print("\n" + "-" * 60)
    print("Collecting data...")
    print("-" * 60)
    
    all_data = collector.collect_all_tests(force_refresh=force_refresh)
    time_ranges = collector.test_config.all()

    # Generate multi-test report
    print("\n" + "-" * 60)
    print("Generating report...")
    print("-" * 60)

    report_path = visualizer.generate_multi_test_report(all_data, time_ranges=time_ranges)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Report: {report_path}")
    print("=" * 60)
    
    return report_path


if __name__ == "__main__":
    # Run with caching (default)
    main()
    
    # To force refresh all data from InfluxDB:
    # main(force_refresh=True)
