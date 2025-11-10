from datetime import datetime

class DateAwareCache:
    def __init__(self):
        """Initialize an empty cache."""
        self.cache = {}  # Internal dictionary to store data
    
    def _parse_date(self, date_str):
        """Convert date string to datetime object for proper comparison."""
        return datetime.strptime(date_str, "%Y-%m-%d")
    
    def get(self, key_str, date_str):
        """
        Attempt to retrieve a value from the cache.
        
        Args:
            key_str: The primary key string
            date_str: The date string to compare (format: YYYY-MM-DD)
            
        Returns:
            tuple: (hit_status, value)
                - hit_status: Boolean indicating if it was a cache hit
                - value: The cached value if hit, None if miss
        """
        # Check if the key exists in our cache
        if key_str in self.cache:
            cached_date_str, cached_value = self.cache[key_str]
            
            # Convert string dates to datetime objects for proper comparison
            date = self._parse_date(date_str)
            cached_date = self._parse_date(cached_date_str)
            
            # Case 3: Exact match on key and date
            if date == cached_date:
                return True, cached_value
                
            # Case 4: Key matches and input date is earlier than cached date
            if date < cached_date:
                return True, cached_value
                
            # Case 5 (partial): Input date is later than cached date
            if date > cached_date:
                return False, None
        
        # Case 5 (partial): Key not found
        return False, None
    
    def put(self, key_str, date_str, value):
        """
        Store a value in the cache.
        
        Args:
            key_str: The primary key string
            date_str: The date string (format: YYYY-MM-DD)
            value: The value to cache
            
        Returns:
            None
        """
        # If key exists, check if we should update based on date
        if key_str in self.cache:
            cached_date_str, _ = self.cache[key_str]
            
            # Convert to datetime for comparison
            date = self._parse_date(date_str)
            cached_date = self._parse_date(cached_date_str)
            
            # If new date is earlier than cached date, don't update
            if date < cached_date:
                return
                
        # Store the new value with its date string
        self.cache[key_str] = (date_str, value)
        
    def __len__(self):
        """Return the number of entries in the cache."""
        return len(self.cache)
        
    def clear(self):
        """Clear all entries from the cache."""
        self.cache.clear()

# Create a global instance that can be imported
global_cache = DateAwareCache()