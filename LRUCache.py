from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def add(self, key, value):
        cache_hit = key in self.cache
        if cache_hit:
            # Move the key to the end to show that it was recently used
            self.cache.move_to_end(key)
            return True
        else:
            # Add the new key-value pair to the cache
            if len(self.cache) >= self.capacity:
                # Remove the first key-value pair
                self.cache.popitem(last=False)
            self.cache[key] = value
            return False
