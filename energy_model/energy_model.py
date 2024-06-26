
process_node_mapper = {
    '7nm': {
        'l2': 0.12,
        'l1': 0.04,
        'l0': 0.02,
        # 'SRAM_read': 0.3,
        # 'SRAM_write': 0.3,
        # 'SRAM_link_per_mm': 1,
        # 'l0_l1_distance': 0,
        # 'l1_l2_distance': 0,
        # 'l0_l1_overhead': 0,
        # 'l1_l2_overhead': 0,
        'energy_per_flop': 1.25
    },
}

memory_node_mapper = {
    'HBM2': {
        'Average_Device_Power': 3.97
        # 'DRAM_activate': 1.21,
        # 'DRAM_IO': 0.22,
        # 'DRAM_transfer': 2.54,
        # 'link_distance': 9.9,
        # 'overhead': 0,
    },
    'GDDR5': {
        'Average_Device_Power': 14.0
        # 'DRAM_activate': 14.0,
        # 'DRAM_IO': 0,
        # 'DRAM_transfer': 0,
        # 'link_distance': 10,
        # 'overhead': 0,
    },
    'GDDR5X': {
        'Average_Device_Power': 8.0,
    },
    'GDDR6': {
        'Average_Device_Power': 7.5,
    },
    'GDDR6X': {
        'Average_Device_Power': 7.25,  # https://www.micron.com/products/memory/hbm/gddr6x
        # 'Average_Device_Power': 6, # https://my.micron.com/products/memory/graphics-memory
    }
}


class EnergyModel:
    def __init__(self, process_node, memory_node) -> None:
        self.process_node = process_node
        self.memory_node = memory_node
        
    def compute(self, flop):
        return flop * process_node_mapper[self.process_node]['energy_per_flop']
        
    def transfer_memory_l2(self, size):
        return (
            memory_node_mapper[self.memory_node]['Average_Device_Power'] + 
            process_node_mapper[self.process_node]['l2']
        ) * size
        # return (
        #     memory_node_mapper[self.memory_node]['DRAM_activate'] + 
        #     memory_node_mapper[self.memory_node]['DRAM_IO'] + 
        #     memory_node_mapper[self.memory_node]['DRAM_transfer'] # * memory_node_mapper[self.memory_node]['link_distance']
        # ) * size + memory_node_mapper[self.memory_node]['overhead']
    
    def transfer_l2_l1(self, size):
        return (
            process_node_mapper[self.process_node]['l2'] + 
            process_node_mapper[self.process_node]['l1']
        ) * size
        # return (
        #     process_node_mapper[self.process_node]['SRAM_read'] + 
        #     process_node_mapper[self.process_node]['SRAM_write'] + 
        #     process_node_mapper[self.process_node]['SRAM_link_per_mm'] * process_node_mapper[self.process_node]['l1_l2_distance']
        # ) * size + process_node_mapper[self.process_node]['l1_l2_overhead']
    
    def transfer_l1_l0(self, size):
        return (
            process_node_mapper[self.process_node]['l1'] + 
            process_node_mapper[self.process_node]['l0']
        ) * size
        # return (
        #     process_node_mapper[self.process_node]['SRAM_read'] + 
        #     process_node_mapper[self.process_node]['SRAM_write'] + 
        #     process_node_mapper[self.process_node]['SRAM_link_per_mm'] * process_node_mapper[self.process_node]['l0_l1_distance']
        # ) * size + process_node_mapper[self.process_node]['l0_l1_overhead']