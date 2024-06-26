class MemoryModule:
    def __init__(self, memory_capacity, memory_node = 'HBM2'):
        self.memory_capacity = memory_capacity
        self.memory_node = memory_node
        
    def __str__(self):
        return f"memory_capacity: {self.memory_capacity}, memory_node: {self.memory_node}"

memory_module_dict = {'A100_80GB': MemoryModule(80e9),'TPUv3': MemoryModule(float('inf')),'MI210': MemoryModule(64e9)}
