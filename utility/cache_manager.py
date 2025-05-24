import hashlib
import json
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
from collections import OrderedDict

class LRUCache:
    """Simple LRU cache implementation"""
    def __init__(self, capacity: int = 1000):
        self.cache = OrderedDict()
        self.capacity = capacity
        
    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Remove least recently used
            self.cache.popitem(last=False)

class CacheManager:
    """Manages various caches for the cultural alignment system"""
    
    def __init__(self, ttl_seconds: int = 3600):  # 1 hour TTL
        self.ttl_seconds = ttl_seconds
        
        # Different caches for different purposes
        self.sensitivity_cache = LRUCache(capacity=5000)
        self.embedding_cache = LRUCache(capacity=10000)
        self.expert_response_cache = LRUCache(capacity=2000)
        
        # Track access patterns for optimization
        self.cache_hits = {"sensitivity": 0, "embedding": 0, "expert": 0}
        self.cache_misses = {"sensitivity": 0, "embedding": 0, "expert": 0}
        
    def _generate_question_key(self, question: str, user_profile: Optional[Dict] = None) -> str:
        """Generate a unique key for a question + optional user profile"""
        key_data = {"question": question.lower().strip()}
        if user_profile:
            # Include key demographic factors that might affect responses
            key_data["user"] = {
                "ancestry": user_profile.get("ancestry", ""),
                "country": user_profile.get("country", ""),
                "age_group": str(int(user_profile.get("age", 0)) // 10) + "0s"  # Group by decade
            }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_sensitivity_analysis(self, question: str) -> Optional[Dict]:
        """Get cached sensitivity analysis for a question"""
        key = self._generate_question_key(question)
        result = self.sensitivity_cache.get(key)
        
        if result:
            self.cache_hits["sensitivity"] += 1
            # Check if not expired
            if time.time() - result["timestamp"] < self.ttl_seconds:
                return result["data"]
        
        self.cache_misses["sensitivity"] += 1
        return None
    
    def put_sensitivity_analysis(self, question: str, analysis: Dict):
        """Cache sensitivity analysis results"""
        key = self._generate_question_key(question)
        self.sensitivity_cache.put(key, {
            "data": analysis,
            "timestamp": time.time()
        })
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding for text"""
        key = hashlib.md5(text.encode()).hexdigest()
        result = self.embedding_cache.get(key)
        
        if result:
            self.cache_hits["embedding"] += 1
            return result
        
        self.cache_misses["embedding"] += 1
        return None
    
    def put_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding"""
        key = hashlib.md5(text.encode()).hexdigest()
        self.embedding_cache.put(key, embedding)
    
    def get_expert_response(self, culture: str, question: str, user_profile: Dict) -> Optional[str]:
        """Get cached expert response"""
        key = f"{culture}:{self._generate_question_key(question, user_profile)}"
        result = self.expert_response_cache.get(key)
        
        if result:
            self.cache_hits["expert"] += 1
            # Check if not expired
            if time.time() - result["timestamp"] < self.ttl_seconds:
                return result["response"]
        
        self.cache_misses["expert"] += 1
        return None
    
    def put_expert_response(self, culture: str, question: str, user_profile: Dict, response: str):
        """Cache expert response"""
        key = f"{culture}:{self._generate_question_key(question, user_profile)}"
        self.expert_response_cache.put(key, {
            "response": response,
            "timestamp": time.time()
        })
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        stats = {}
        for cache_type in ["sensitivity", "embedding", "expert"]:
            hits = self.cache_hits[cache_type]
            misses = self.cache_misses[cache_type]
            total = hits + misses
            hit_rate = (hits / total * 100) if total > 0 else 0
            
            stats[cache_type] = {
                "hits": hits,
                "misses": misses,
                "hit_rate": f"{hit_rate:.1f}%",
                "total_requests": total
            }
        
        return stats
    
    def clear_expired(self):
        """Clear expired entries from all caches"""
        current_time = time.time()
        
        # Clear expired sensitivity entries
        for key in list(self.sensitivity_cache.cache.keys()):
            item = self.sensitivity_cache.cache[key]
            if current_time - item["timestamp"] > self.ttl_seconds:
                del self.sensitivity_cache.cache[key]
        
        # Clear expired expert responses
        for key in list(self.expert_response_cache.cache.keys()):
            item = self.expert_response_cache.cache[key]
            if current_time - item["timestamp"] > self.ttl_seconds:
                del self.expert_response_cache.cache[key]

# Global cache instance
_global_cache = None

def get_cache_manager() -> CacheManager:
    """Get or create global cache manager instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager()
    return _global_cache