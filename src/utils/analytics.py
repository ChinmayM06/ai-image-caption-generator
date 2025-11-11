"""
Analytics Module

Tracks usage statistics and performance metrics for the caption generator.
Provides insights into model usage, processing times, and popular styles.
"""

import json
import threading
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from config import analytics_config, style_config


@dataclass
class AnalyticsData:
    """Container for analytics data"""
    total_captions: int = 0
    style_usage: Dict[str, int] = None
    avg_processing_time: float = 0.0
    total_processing_time: float = 0.0
    model_usage: Dict[str, int] = None
    error_count: int = 0
    last_updated: Optional[str] = None
    
    def __post_init__(self):
        if self.style_usage is None:
            self.style_usage = {style: 0 for style in style_config.STYLES.keys()}
        if self.model_usage is None:
            self.model_usage = {"blip": 0, "git": 0}
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


class AnalyticsManager:
    """
    Thread-safe analytics manager for tracking usage metrics
    
    Features:
    - Real-time metric tracking
    - Persistent storage
    - Thread-safe operations
    - Automatic calculations
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize analytics manager
        
        Args:
            storage_path: Path to analytics JSON file
        """
        self.storage_path = storage_path or analytics_config.ANALYTICS_FILE
        self._lock = threading.RLock()
        
        # Load existing data or initialize new
        self.data = self._load_data()
    
    def _load_data(self) -> AnalyticsData:
        """
        Load analytics data from file
        
        Returns:
            AnalyticsData: Loaded or initialized data
        """
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data_dict = json.load(f)
                    return AnalyticsData(**data_dict)
            except Exception as e:
                print(f"Warning: Failed to load analytics: {e}")
                return AnalyticsData()
        else:
            return AnalyticsData()
    
    def _save_data(self) -> bool:
        """
        Save analytics data to file
        
        Returns:
            bool: True if successful
        """
        try:
            # Ensure directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Update timestamp
            self.data.last_updated = datetime.now().isoformat()
            
            # Write to file
            with open(self.storage_path, 'w') as f:
                json.dump(self.data.to_dict(), f, indent=4)
            
            return True
        except Exception as e:
            print(f"Error saving analytics: {e}")
            return False
    
    def record_caption_generation(
        self,
        model_name: str,
        style: str,
        processing_time: float,
        success: bool = True
    ) -> None:
        """
        Record a caption generation event
        
        Args:
            model_name: Name of the model used (blip/git)
            style: Style applied
            processing_time: Time taken in seconds
            success: Whether generation was successful
        """
        with self._lock:
            if success:
                # Increment counters
                self.data.total_captions += 1
                
                # Update style usage
                if style in self.data.style_usage:
                    self.data.style_usage[style] += 1
                
                # Update model usage
                model_key = model_name.lower()
                if model_key in self.data.model_usage:
                    self.data.model_usage[model_key] += 1
                
                # Update processing time
                self.data.total_processing_time += processing_time
                self.data.avg_processing_time = (
                    self.data.total_processing_time / self.data.total_captions
                )
            else:
                self.data.error_count += 1
            
            # Save to disk
            self._save_data()
    
    def record_batch_generation(
        self,
        generations: list[dict]
    ) -> None:
        """
        Record multiple caption generations at once
        
        Args:
            generations: List of generation records
                Each record: {model_name, style, processing_time, success}
        """
        with self._lock:
            for gen in generations:
                self.record_caption_generation(
                    model_name=gen.get("model_name", "unknown"),
                    style=gen.get("style", "None"),
                    processing_time=gen.get("processing_time", 0.0),
                    success=gen.get("success", True)
                )
    
    def get_stats(self) -> dict:
        """
        Get current statistics
        
        Returns:
            dict: Current analytics data
        """
        with self._lock:
            return self.data.to_dict()
    
    def get_summary(self) -> dict:
        """
        Get formatted summary of analytics
        
        Returns:
            dict: Human-readable summary
        """
        with self._lock:
            total = self.data.total_captions
            
            # Calculate percentages for styles
            style_percentages = {}
            if total > 0:
                for style, count in self.data.style_usage.items():
                    style_percentages[style] = round((count / total) * 100, 1)
            
            # Calculate percentages for models
            model_percentages = {}
            if total > 0:
                for model, count in self.data.model_usage.items():
                    model_percentages[model] = round((count / total) * 100, 1)
            
            # Find most popular style
            popular_style = max(
                self.data.style_usage.items(),
                key=lambda x: x[1]
            )[0] if self.data.style_usage else "None"
            
            return {
                "total_captions": total,
                "avg_processing_time": round(self.data.avg_processing_time, 2),
                "error_rate": round(
                    (self.data.error_count / (total + self.data.error_count) * 100)
                    if (total + self.data.error_count) > 0 else 0,
                    2
                ),
                "most_popular_style": popular_style,
                "style_distribution": style_percentages,
                "model_distribution": model_percentages,
                "last_updated": self.data.last_updated
            }
    
    def get_display_stats(self) -> str:
        """
        Get formatted stats for UI display
        
        Returns:
            str: Formatted statistics string
        """
        with self._lock:
            summary = self.get_summary()
            
            stats_text = (
                f"📊 Total Captions: {summary['total_captions']} | "
                f"⚡ Avg Time: {summary['avg_processing_time']}s | "
                f"🎨 Popular Style: {summary['most_popular_style']}"
            )
            
            return stats_text
    
    def reset_stats(self) -> bool:
        """
        Reset all statistics
        
        Returns:
            bool: True if successful
        """
        with self._lock:
            self.data = AnalyticsData()
            return self._save_data()
    
    def export_stats(self, export_path: Optional[Path] = None) -> bool:
        """
        Export statistics to a file
        
        Args:
            export_path: Path to export file (default: timestamped file)
            
        Returns:
            bool: True if successful
        """
        with self._lock:
            if export_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = self.storage_path.parent / f"analytics_export_{timestamp}.json"
            
            try:
                with open(export_path, 'w') as f:
                    export_data = {
                        "exported_at": datetime.now().isoformat(),
                        "statistics": self.data.to_dict(),
                        "summary": self.get_summary()
                    }
                    json.dump(export_data, f, indent=4)
                return True
            except Exception as e:
                print(f"Error exporting analytics: {e}")
                return False


# Singleton instance
_analytics_manager = None
_manager_lock = threading.Lock()


def get_analytics_manager() -> AnalyticsManager:
    """Get singleton AnalyticsManager instance"""
    global _analytics_manager
    if _analytics_manager is None:
        with _manager_lock:
            if _analytics_manager is None:
                _analytics_manager = AnalyticsManager()
    return _analytics_manager


# Convenience functions
def record_generation(
    model_name: str,
    style: str,
    processing_time: float,
    success: bool = True
) -> None:
    """Record a caption generation (convenience function)"""
    get_analytics_manager().record_caption_generation(
        model_name, style, processing_time, success
    )


def get_stats() -> dict:
    """Get current statistics (convenience function)"""
    return get_analytics_manager().get_stats()


def get_summary() -> dict:
    """Get analytics summary (convenience function)"""
    return get_analytics_manager().get_summary()


def get_display_stats() -> str:
    """Get formatted display stats (convenience function)"""
    return get_analytics_manager().get_display_stats()


if __name__ == "__main__":
    # Test the analytics manager
    print("=" * 60)
    print("ANALYTICS MANAGER - TEST MODE")
    print("=" * 60)
    
    # Initialize manager with test path
    test_path = Path("cache/test_analytics.json")
    analytics = AnalyticsManager(storage_path=test_path)
    
    print("\n1. Initial state:")
    print(f"   {analytics.get_display_stats()}")
    
    print("\n2. Recording test generations:")
    analytics.record_caption_generation("blip", "Professional", 2.5, True)
    analytics.record_caption_generation("git", "Creative", 3.2, True)
    analytics.record_caption_generation("blip", "Professional", 2.1, True)
    analytics.record_caption_generation("git", "Social Media", 2.8, True)
    analytics.record_caption_generation("blip", "Technical", 2.3, False)
    print(f"   Recorded 5 generations (4 success, 1 error)")
    
    print("\n3. Current statistics:")
    stats = analytics.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"      {k}: {v}")
        else:
            print(f"   {key}: {value}")
    
    print("\n4. Summary:")
    summary = analytics.get_summary()
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"      {k}: {v}")
        else:
            print(f"   {key}: {value}")
    
    print("\n5. Display format:")
    print(f"   {analytics.get_display_stats()}")
    
    print("\n6. File saved to:")
    print(f"   {test_path}")
    
    print("\n" + "=" * 60)
    print("✓ Analytics manager tests complete")
    print("=" * 60)
    
    # Cleanup test file
    if test_path.exists():
        test_path.unlink()
        print("\n✓ Test file cleaned up")