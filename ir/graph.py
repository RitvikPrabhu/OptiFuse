class OpNode:
    def __init__(self, name, op_type, shape, dtype, inputs=None):
        self.name = name
        self.op_type = op_type          # "GEMM", "ReLU", ...
        self.shape = shape              # tuple
        self.dtype = dtype              # "fp32" | "fp16" | "bf16"
        self.inputs = inputs or []      # list[OpNode]

class Graph:
    def __init__(self, ops):
        self.ops = ops                  
        self.device = None              
