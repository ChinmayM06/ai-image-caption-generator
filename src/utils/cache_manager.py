"""
Cache Management Module

Implements intelligent caching for caption generation results.
Uses LRU (Least Recently Used) eviction policy for memory efficiency.
"""

import time
import json
from typing import Optional, Any, Dict
from collections import OrderedDict
from dataclasses import dataclass, asdict
from datetime import datetime
import threading

from config import cache_config


@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata"""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.timestamp
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


class CacheManager:
    """
    Thread-safe LRU cache manager for caption results
    
    Features:
    - Automatic expiration based on TTL
    - LRU eviction when max size reached
    - Thread-safe operations
    - Access statistics
    - Memory-efficient storage
    """
    
    def __init__(
        self,
        max_size: int = cache_config.MAX_CACHE_SIZE,
        ttl_seconds: int = cache_config.CACHE_TTL_SECONDS
    ):
        """
        Initialize cache manager
        
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time to live for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
        # OrderedDict maintains insertion order and enables O(1) LRU
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
            "total_sets": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Any]: Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if self._is_expired(entry):
                self._remove_entry(key)
                self._stats["expirations"] += 1
                self._stats["misses"] += 1
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = time.time()
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            self._stats["hits"] += 1
            return entry.value
    
    def set(self, key: str, value: Any) -> bool:
        """
        Store value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            
        Returns:
            bool: True if successfully cached
        """
        with self._lock:
            current_time = time.time()
            
            # If key exists, update it
            if key in self._cache:
                entry = self._cache[key]
                entry.value = value
                entry.timestamp = current_time
                entry.last_accessed = current_time
                self._cache.move_to_end(key)
            else:
                # Check if we need to evict
                if len(self._cache) >= self.max_size:
                    self._evict_oldest()
                
                # Add new entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=current_time,
                    last_accessed=current_time
                )
                self._cache[key] = entry
            
            self._stats["total_sets"] += 1
            return True
    
    def delete(self, key: str) -> bool:
        """
        Remove entry from cache
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if entry was deleted
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry has expired"""
        return (time.time() - entry.timestamp) > self.ttl_seconds
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry without stats update"""
        if key in self._cache:
            del self._cache[key]
    
    def _evict_oldest(self) -> None:
        """Evict least recently used entry"""
        if self._cache:
            # OrderedDict: first item is least recently used
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._stats["evictions"] += 1
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries
        
        Returns:
            int: Number of entries removed
        """
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if (current_time - entry.timestamp) > self.ttl_seconds
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                self._stats["expirations"] += len(expired_keys)
            
            return len(expired_keys)
    
    def get_stats(self) -> dict:
        """
        Get cache statistics
        
        Returns:
            dict: Cache statistics including hit rate
        """
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                (self._stats["hits"] / total_requests * 100)
                if total_requests > 0 else 0
            )
            
            return {
                **self._stats,
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": round(hit_rate, 2),
                "total_requests": total_requests
            }
    
    def get_info(self) -> dict:
        """
        Get detailed cache information
        
        Returns:
            dict: Detailed cache state
        """
        with self._lock:
            entries_info = []
            for key, entry in self._cache.items():
                age_seconds = time.time() - entry.timestamp
                entries_info.append({
                    "key": key[:50] + "..." if len(key) > 50 else key,
                    "age_seconds": round(age_seconds, 2),
                    "access_count": entry.access_count,
                    "size_estimate": len(str(entry.value))
                })
            
            return {
                "stats": self.get_stats(),
                "entries": entries_info[:10],  # Show top 10
                "config": {
                    "max_size": self.max_size,
                    "ttl_seconds": self.ttl_seconds
                }
            }


class CaptionCache:
    """
    Specialized cache for image captions
    
    Manages caching of caption generation results with image hash keys
    """
    
    def __init__(self):
        """Initialize caption cache"""
        self.cache = CacheManager(
            max_size=cache_config.MAX_CACHE_SIZE,
            ttl_seconds=cache_config.CACHE_TTL_SECONDS
        )
        self.enabled = cache_config.ENABLE_CAPTION_CACHE
    
    def get_caption(
        self,
        image_hash: str,
        model_name: str,
        style: str
    ) -> Optional[str]:
        """
        Retrieve cached caption
        
        Args:
            image_hash: Hash of the image
            model_name: Name of the caption model
            style: Style applied
            
        Returns:
            Optional[str]: Cached caption or None
        """
        if not self.enabled:
            return None
        
        cache_key = self._generate_key(image_hash, model_name, style)
        return self.cache.get(cache_key)
    
    def set_caption(
        self,
        image_hash: str,
        model_name: str,
        style: str,
        caption: str
    ) -> bool:
        """
        Store caption in cache
        
        Args:
            image_hash: Hash of the image
            model_name: Name of the caption model
            style: Style applied
            caption: Generated caption
            
        Returns:
            bool: True if successfully cached
        """
        if not self.enabled:
            return False
        
        cache_key = self._generate_key(image_hash, model_name, style)
        return self.cache.set(cache_key, caption)
    
    def _generate_key(self, image_hash: str, model_name: str, style: str) -> str:
        """Generate cache key from components"""
        return f"{image_hash}:{model_name}:{style}"
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        return self.cache.get_stats()
    
    def clear(self) -> None:
        """Clear all cached captions"""
        self.cache.clear()
    
    def cleanup(self) -> int:
        """Clean up expired entries"""
        return self.cache.cleanup_expired()


# Singleton instances
_cache_manager = None
_caption_cache = None


def get_cache_manager() -> CacheManager:
    """Get singleton CacheManager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def get_caption_cache() -> CaptionCache:
    """Get singleton CaptionCache instance"""
    global _caption_cache
    if _caption_cache is None:
        _caption_cache = CaptionCache()
    return _caption_cache


if __name__ == "__main__":
    # Test the cache manager
    print("=" * 60)
    print("CACHE MANAGER - TEST MODE")
    print("=" * 60)
    
    # Test basic cache operations
    cache = CacheManager(max_size=3, ttl_seconds=5)
    
    print("\n1. Testing SET operations:")
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")
    print(f"   Added 3 items")
    print(f"   Cache size: {len(cache._cache)}")
    
    print("\n2. Testing GET operations:")
    result = cache.get("key1")
    print(f"   Get 'key1': {result}")
    print(f"   Stats: {cache.get_stats()}")
    
    print("\n3. Testing LRU eviction:")
    cache.set("key4", "value4")  # Should evict key2
    print(f"   Added 'key4'")
    print(f"   Cache size: {len(cache._cache)}")
    print(f"   Keys in cache: {list(cache._cache.keys())}")
    
    print("\n4. Testing TTL expiration:")
    print(f"   Waiting 6 seconds for expiration...")
    time.sleep(6)
    expired = cache.cleanup_expired()
    print(f"   Expired entries: {expired}")
    print(f"   Cache size: {len(cache._cache)}")
    
    print("\n5. Final stats:")
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✓ Cache manager tests complete")
    print("=" * 60)