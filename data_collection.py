"""
Data Collection and Processing Module for Belt Tracking Swing Arm Analysis
"""

from influxdb import InfluxDBClient
import config_sandbox
from datetime import datetime, timedelta
import pytz
import pandas as pd
import json
import pickle
from pathlib import Path


class TestConfig:
    """Manages test configurations stored in JSON file."""
    
    def __init__(self, config_file: Path = None):
        """
        Initialize TestConfig manager.
        
        Args:
            config_file: Path to JSON config file. Defaults to tests_config.json
        """
        if config_file is None:
            config_file = Path(__file__).parent / "tests_config.json"
        self.config_file = config_file
        self._ensure_file_exists()
    
    def _ensure_file_exists(self) -> None:
        """Create config file if it doesn't exist."""
        if not self.config_file.exists():
            self._save({"tests": {}})
    
    def _load(self) -> dict:
        """Load configuration from file."""
        with open(self.config_file, 'r') as f:
            return json.load(f)
    
    def _save(self, config: dict) -> None:
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)
    
    def add(self, test_name: str, from_time: str, to_time: str) -> None:
        """
        Add a new test timeframe.
        
        Args:
            test_name: Unique identifier for the test
            from_time: Start time in ISO 8601 format 'YYYY-MM-DDTHH:MM:SS.sssZ'
            to_time: End time in ISO 8601 format 'YYYY-MM-DDTHH:MM:SS.sssZ'
        """
        config = self._load()
        config["tests"][test_name] = {
            "from": from_time,
            "to": to_time
        }
        self._save(config)
        print(f"Added test '{test_name}': {from_time} to {to_time}")
    
    def remove(self, test_name: str) -> None:
        """Remove a test timeframe."""
        config = self._load()
        if test_name in config["tests"]:
            del config["tests"][test_name]
            self._save(config)
            print(f"Removed test '{test_name}'")
        else:
            print(f"Test '{test_name}' not found")
    
    def get(self, test_name: str) -> dict:
        """Get a specific test's time range."""
        config = self._load()
        if test_name not in config["tests"]:
            raise ValueError(f"Test '{test_name}' not found. Available: {self.list()}")
        return config["tests"][test_name]
    
    def list(self) -> list[str]:
        """List all saved test names."""
        config = self._load()
        return list(config.get("tests", {}).keys())
    
    def all(self) -> dict:
        """Get all tests with their time ranges."""
        config = self._load()
        return config.get("tests", {})


class DataCache:
    """Manages local caching of collected test data."""
    
    def __init__(self, cache_dir: Path = None):
        """
        Initialize DataCache manager.
        
        Args:
            cache_dir: Directory for cache files. Defaults to 'cache/' folder.
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent / "cache"
        self.cache_dir = cache_dir
        self._ensure_dir_exists()
    
    def _ensure_dir_exists(self) -> None:
        """Create cache directory if it doesn't exist."""
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_path(self, test_name: str) -> Path:
        """Get the cache file path for a test."""
        return self.cache_dir / f"{test_name}.pkl"
    
    def exists(self, test_name: str) -> bool:
        """Check if cached data exists for a test."""
        return self._get_cache_path(test_name).exists()
    
    def save(self, test_name: str, data: dict[str, pd.DataFrame]) -> None:
        """
        Save test data to cache.
        
        Args:
            test_name: Name of the test
            data: Dictionary mapping designator names to DataFrames
        """
        cache_path = self._get_cache_path(test_name)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Cached data for '{test_name}'")
    
    def load(self, test_name: str) -> dict[str, pd.DataFrame]:
        """
        Load test data from cache.
        
        Args:
            test_name: Name of the test
        
        Returns:
            Dictionary mapping designator names to DataFrames
        """
        cache_path = self._get_cache_path(test_name)
        if not cache_path.exists():
            raise FileNotFoundError(f"No cached data for '{test_name}'")
        
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded cached data for '{test_name}'")
        return data
    
    def delete(self, test_name: str) -> None:
        """Delete cached data for a test."""
        cache_path = self._get_cache_path(test_name)
        if cache_path.exists():
            cache_path.unlink()
            print(f"Deleted cache for '{test_name}'")
        else:
            print(f"No cache found for '{test_name}'")
    
    def list_cached(self) -> list[str]:
        """List all cached test names."""
        return [p.stem for p in self.cache_dir.glob("*.pkl")]
    
    def clear_all(self) -> None:
        """Delete all cached data."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        print("Cleared all cached data")


class DataCollector:
    """Handles InfluxDB data collection for belt tracking sensors."""
    
    # Preset designators for belt tracking swing arm sensors
    DESIGNATORS = [
        "R1_LEFT_ANGLE",
        "R1_RIGHT_ANGLE",
        "R3_LEFT_ANGLE",
        "R3_RIGHT_ANGLE",
        "R5_LEFT_ANGLE",
        "R5_RIGHT_ANGLE",
        "R7_LEFT_ANGLE",
        "R7_RIGHT_ANGLE",
    ]
    
    # Local timezone for display
    LOCAL_TIMEZONE = "US/Central"
    
    def __init__(
        self,
        vin: str = "Sandbox1",
        aggregator: str = "*",
        group_by_time: str = "1s",
        use_cache: bool = True
    ):
        """
        Initialize DataCollector.
        
        Args:
            vin: Vehicle identification number
            aggregator: Default aggregation function ('*' for raw data, 'mean', 'max', 'min', etc.)
            group_by_time: Time grouping interval when using aggregation (e.g., '1s', '5s', '1m')
            use_cache: Whether to use local caching (default True)
        """
        self.vin = vin
        self.aggregator = aggregator
        self.group_by_time = group_by_time
        self.use_cache = use_cache
        self._client = None
        self._test_config = TestConfig()
        self._cache = DataCache()
    
    @property
    def client(self) -> InfluxDBClient:
        """Lazy-load InfluxDB client."""
        if self._client is None:
            self._client = InfluxDBClient(
                host=config_sandbox.HOST,
                port=config_sandbox.PORT,
                username=config_sandbox.USERNAME,
                password=config_sandbox.PASSWORD,
                database=config_sandbox.DATABASE
            )
        return self._client
    
    @property
    def test_config(self) -> TestConfig:
        """Access to test configuration manager."""
        return self._test_config
    
    @property
    def cache(self) -> DataCache:
        """Access to data cache manager."""
        return self._cache
    
    def _parse_iso_time(self, time_str: str) -> datetime:
        """
        Parse ISO 8601 time string to datetime.
        
        Args:
            time_str: Time in format 'YYYY-MM-DDTHH:MM:SS.sssZ'
        
        Returns:
            datetime object in UTC
        """
        # Remove Z suffix and parse, then set UTC timezone
        time_str_clean = time_str.rstrip('Z')
        try:
            dt = datetime.fromisoformat(time_str_clean)
        except ValueError:
            # Fallback for formats without milliseconds
            dt = datetime.strptime(time_str_clean, '%Y-%m-%dT%H:%M:%S')
        return dt.replace(tzinfo=pytz.utc)
    
    def _convert_time_to_local(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the 'time' column from UTC to local timezone.
        
        Args:
            df: DataFrame with a 'time' column in UTC
        
        Returns:
            DataFrame with 'time' column converted to local timezone
        """
        if df.empty or 'time' not in df.columns:
            return df
        
        local_tz = pytz.timezone(self.LOCAL_TIMEZONE)
        
        # Parse times to datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Check if already timezone-aware
        if df['time'].dt.tz is not None:
            # Already tz-aware, just convert
            df['time'] = df['time'].dt.tz_convert(local_tz)
        else:
            # Not tz-aware, localize to UTC first then convert
            df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert(local_tz)
        
        return df
    
    def _get_measurement(
        self,
        start_time: str,
        end_time: str,
        designator: str,
        aggregator: str = None,
        group_by_time: str = None
    ) -> list:
        """
        Query InfluxDB for measurement data.
        
        Args:
            start_time: Start time in ISO 8601 format 'YYYY-MM-DDTHH:MM:SS.sssZ'
            end_time: End time in ISO 8601 format 'YYYY-MM-DDTHH:MM:SS.sssZ'
            designator: Sensor designator name
            aggregator: Aggregation function (uses instance default if None)
            group_by_time: Time grouping interval (uses instance default if None)
        
        Returns:
            List of measurement points
        """
        aggregator = aggregator or self.aggregator
        group_by_time = group_by_time or self.group_by_time
        
        # Parse ISO 8601 time format (already in UTC)
        start_utc = self._parse_iso_time(start_time)
        end_utc = self._parse_iso_time(end_time)
        now = datetime.now(pytz.utc)

        # Determine retention policy
        if (now - start_utc) > timedelta(weeks=2) or (now - end_utc) > timedelta(weeks=2):
            policy = 'tenyear'
        else:
            policy = 'autogen'

        # Format UTC times for query
        start = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        end = end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Build aggregation strings
        agg_verbose = ''
        group_by_verbose = ''
        group_by_str = ''
        agg_query = aggregator
        
        if aggregator != '*':
            agg_verbose = f' {aggregator}'
            group_by_verbose = f', aggregated every {group_by_time}'
            group_by_str = f'GROUP BY time({group_by_time}) fill(null)'
            agg_query = f' {aggregator}("value")'

        print(f"  Fetching{agg_verbose} '{designator}' from {start} to {end} ({policy}){group_by_verbose}")

        # Execute query
        query = f'SELECT {agg_query} FROM "{policy}"."/IOList/{designator}/Value" WHERE time > \'{start}\' AND time < \'{end}\' {group_by_str}'
        result_set = self.client.query(query).get_points(
            tags={'designator': f'{designator}'}
        ) if aggregator == '*' else self.client.query(query).get_points()

        return list(result_set)
    
    def _collect_from_influx(
        self,
        time_range: dict,
        designators: list[str] = None,
        aggregator: str = None,
        group_by_time: str = None
    ) -> dict[str, pd.DataFrame]:
        """
        Collect data directly from InfluxDB (no caching).
        
        Args:
            time_range: Dictionary with 'from' and 'to' keys
            designators: List of designators (uses all preset if None)
            aggregator: Aggregation function (uses instance default if None)
            group_by_time: Time grouping interval (uses instance default if None)
        
        Returns:
            Dictionary mapping designator names to DataFrames (times in local timezone)
        """
        designators = designators or self.DESIGNATORS
        results = {}
        
        for designator in designators:
            data = self._get_measurement(
                start_time=time_range["from"],
                end_time=time_range["to"],
                designator=designator,
                aggregator=aggregator,
                group_by_time=group_by_time
            )
            df = pd.DataFrame(data)
            df = self._convert_time_to_local(df)
            results[designator] = df
        
        return results
    
    def collect_test(
        self,
        test_name: str,
        aggregator: str = None,
        group_by_time: str = None,
        force_refresh: bool = False
    ) -> dict[str, pd.DataFrame]:
        """
        Collect data for a saved test configuration.
        Uses cached data if available, otherwise fetches from InfluxDB and caches.
        
        Args:
            test_name: Name of the test from tests_config.json
            aggregator: Aggregation function (uses instance default if None)
            group_by_time: Time grouping interval (uses instance default if None)
            force_refresh: If True, ignore cache and fetch fresh data
        
        Returns:
            Dictionary mapping designator names to DataFrames
        """
        time_range = self._test_config.get(test_name)
        
        # Check cache first (unless force_refresh)
        if self.use_cache and not force_refresh and self._cache.exists(test_name):
            print(f"\n=== Loading cached data for test: {test_name} ===")
            return self._cache.load(test_name)
        
        # Fetch from InfluxDB
        print(f"\n=== Fetching data for test: {test_name} ===")
        print(f"Time range: {time_range['from']} to {time_range['to']}")
        
        data = self._collect_from_influx(
            time_range,
            aggregator=aggregator,
            group_by_time=group_by_time
        )
        
        # Save to cache
        if self.use_cache:
            self._cache.save(test_name, data)
        
        return data
    
    def collect_all_tests(
        self,
        aggregator: str = None,
        group_by_time: str = None,
        force_refresh: bool = False
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Collect data for all configured tests.
        Uses cached data where available.
        
        Args:
            aggregator: Aggregation function (uses instance default if None)
            group_by_time: Time grouping interval (uses instance default if None)
            force_refresh: If True, ignore cache and fetch fresh data for all tests
        
        Returns:
            Dictionary mapping test names to their data dictionaries
        """
        all_data = {}
        test_names = self._test_config.list()
        
        for test_name in test_names:
            all_data[test_name] = self.collect_test(
                test_name,
                aggregator=aggregator,
                group_by_time=group_by_time,
                force_refresh=force_refresh
            )
        
        return all_data


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Belt Tracking Swing Arm - Data Collection")
    print("=" * 50)
    
    # Initialize collector
    collector = DataCollector()
    
    # List available tests
    print(f"\nConfigured tests: {collector.test_config.list()}")
    print(f"Cached tests: {collector.cache.list_cached()}")
    
    # Example: Add a new test timeframe
    # collector.test_config.add("my_test", "2026-02-01T00:00:00.000Z", "2026-02-01T20:00:00.000Z")
    
    # Example: Collect data for a test (uses cache if available)
    # data = collector.collect_test("example_test_1")
    # for designator, df in data.items():
    #     print(f"{designator}: {len(df)} rows")
    
    # Example: Force refresh (ignore cache)
    # data = collector.collect_test("example_test_1", force_refresh=True)
    
    # Example: Clear cache
    # collector.cache.clear_all()
