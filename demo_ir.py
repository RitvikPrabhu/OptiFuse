from ir.graph import OpNode, Graph
import torch 

a = OpNode("gemm1", "GEMM", (512, 512, 512), "fp16")
b = OpNode("relu1", "ReLU", (512, 512), "fp16", inputs=[a])
g = Graph([a, b])
print([o.name for o in g.ops])

g.bind_device()
print(g.device)


