
process_node_mapper = {
    '7nm': {
        'SRAM_read': 0.3,
        'SRAM_write': 0.3,
        'SRAM_link_per_mm': 0.1,
        'l0_l1_distance': 1,
        'l1_l2_distance': 5,
        'l0_l1_overhead': 1,
        'l1_l2_overhead': 5,
        'energy_per_flop': 1
    },
}

memory_node_mapper = {
    'HBM2': {
        'DRAM_activate': 5,
        'DRAM_read': 5,
        'DRAM_write': 5,
        'DRAM_precharge': 5,
        'DRAM_link_per_mm': 10,
        'link_distance': 10,
        'overhead': 100,
    },
}


class EnergyModel:
    def __init__(self, process_node, memory_node) -> None:
        self.process_node = process_node
        self.memory_node = memory_node
        
    def compute(self, flop):
        return flop * process_node_mapper[self.process_node]['energy_per_flop']
        
    def transfer_memory_l2(self, size):
        return (
            memory_node_mapper[self.memory_node]['DRAM_activate'] + 
            memory_node_mapper[self.memory_node]['DRAM_read'] + 
            memory_node_mapper[self.memory_node]['DRAM_precharge'] + 
            memory_node_mapper[self.memory_node]['DRAM_link_per_mm'] * memory_node_mapper[self.memory_node]['link_distance']
        ) * size + memory_node_mapper[self.memory_node]['overhead']
    
    def transfer_l2_l1(self, size):
        return (
            process_node_mapper[self.process_node]['SRAM_read'] + 
            process_node_mapper[self.process_node]['SRAM_write'] + 
            process_node_mapper[self.process_node]['SRAM_link_per_mm'] * process_node_mapper[self.process_node]['l1_l2_distance']
        ) * size + process_node_mapper[self.process_node]['l1_l2_overhead']
    
    def transfer_l1_l0(self, size):
        return (
            process_node_mapper[self.process_node]['SRAM_read'] + 
            process_node_mapper[self.process_node]['SRAM_write'] + 
            process_node_mapper[self.process_node]['SRAM_link_per_mm'] * process_node_mapper[self.process_node]['l0_l1_distance']
        ) * size + process_node_mapper[self.process_node]['l0_l1_overhead']