# %%
import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn import Gate

gate = Gate(
    irreps_scalars="2x0e",
    act_scalars=[torch.nn.functional.silu],
    irreps_gates="1x0e",
    act_gates=[torch.sigmoid],
    irreps_gated="1x1o",
)

print("Input irreps:", gate.irreps_in)    # what the gate expects
print("Output irreps:", gate.irreps_out)  # what comes out

# Feed it one node's features: 6 numbers
x = torch.randn(1, 6)  # [batch=1, 3 scalars + 3 vector components]
y = gate(x)
print("Input shape:", x.shape)   # [1, 6]
print("Output shape:", y.shape)  # [1, 5]
# %%
def _build_gate(out_irreps):
        irreps = o3.Irreps(out_irreps)
        scalars = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l == 0])
        non_scalars = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l > 0])
        num_gates = sum(mul for mul, _ in non_scalars)
        gates = o3.Irreps(f"{num_gates}x0e") if num_gates > 0 else o3.Irreps("")

        return Gate(
            irreps_scalars = scalars, 
            act_scalars = [F.silu] * len(scalars),
            irreps_gates = gates, 
            act_gates = [torch.sigmoid] * len(gates),
            irreps_gated = non_scalars
        )

# %%
gate = _build_gate("32x0e + 32x1o + 16x2e")
print("Input:", gate.irreps_in)   # should be 80x0e + 32x1o + 16x2e
print("Output:", gate.irreps_out) # should be 32x0e + 32x1o + 16x2e
# %%
