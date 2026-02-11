"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_vector_hash(self, v_val_addr, v_tmp1, v_tmp2, v_consts):
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_val1 = v_consts[f"h1_{hi}"]
            v_val3 = v_consts[f"h3_{hi}"]
            self.instrs.append({"valu": [
                (op1, v_tmp1, v_val_addr, v_val1),
                (op3, v_tmp2, v_val_addr, v_val3),
            ]})
            self.instrs.append({"valu": [
                (op2, v_val_addr, v_tmp1, v_tmp2),
            ]})
            
            

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized VLIW-packed vectorized kernel.
        Key optimizations:
        - Pack independent ops from different engines into same cycle
        - Parallel scatter-gather (8 ALU addr computations + paired loads)
        - Pack independent hash stage ops (op1/op3 in same cycle)
        - Use multiply_add + bitwise AND for branch logic
        - Hoist loop-invariant n_nodes broadcast
        - Overlap stores with next iteration's address computation
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_addr2 = self.alloc_scratch("tmp_addr2")

        v_consts = {}
        for val in [0, 1, 2]:
            addr = self.alloc_scratch(f"v_const_{val}", VLEN)
            self.add("valu", ("vbroadcast", addr, self.scratch_const(val)))
            v_consts[val] = addr

        for hi, (_, val1, _, _, val3) in enumerate(HASH_STAGES):
            a1 = self.alloc_scratch(f"vh1_{hi}", VLEN)
            a3 = self.alloc_scratch(f"vh3_{hi}", VLEN)
            self.add("valu", ("vbroadcast", a1, self.scratch_const(val1)))
            self.add("valu", ("vbroadcast", a3, self.scratch_const(val3)))
            v_consts[f"h1_{hi}"] = a1
            v_consts[f"h3_{hi}"] = a3

        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height","forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        self.add("flow", ("pause",))

        v_idx = self.alloc_scratch("v_idx", VLEN)
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node_val = self.alloc_scratch("v_node_val", VLEN)
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)

        gather_addrs = [self.alloc_scratch(f"ga_{vi}") for vi in range(VLEN)]

        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        self.instrs.append({"valu": [("vbroadcast", v_n_nodes, self.scratch["n_nodes"])]})

        i_consts = {}
        for i in range(0, batch_size, VLEN):
            i_consts[i] = self.scratch_const(i)

        fvp = self.scratch["forest_values_p"]
        iip = self.scratch["inp_indices_p"]
        ivp = self.scratch["inp_values_p"]

        all_offsets = []
        for r in range(rounds):
            for i in range(0, batch_size, VLEN):
                all_offsets.append(i)
        total_iters = len(all_offsets)

        self.instrs.append({"alu": [
            ("+", tmp_addr, iip, i_consts[all_offsets[0]]),
            ("+", tmp_addr2, ivp, i_consts[all_offsets[0]]),
        ]})

        for iter_idx in range(total_iters):
            self.instrs.append({"load": [
                ("vload", v_idx, tmp_addr),
                ("vload", v_val, tmp_addr2),
            ]})
            self.instrs.append({"alu": [
                ("+", gather_addrs[vi], fvp, v_idx + vi) for vi in range(VLEN)
            ]})

            for pair in range(0, VLEN, 2):
                self.instrs.append({"load": [
                    ("load", v_node_val + pair, gather_addrs[pair]),
                    ("load", v_node_val + pair + 1, gather_addrs[pair + 1]),
                ]})

            self.instrs.append({"valu": [("^", v_val, v_val, v_node_val)]})

            self.build_vector_hash(v_val, v_tmp1, v_tmp2, v_consts)

            self.instrs.append({"valu": [
                ("multiply_add", v_tmp1, v_idx, v_consts[2], v_consts[1]),  # 2*idx + 1
                ("&", v_tmp2, v_val, v_consts[1]),                         
            ]})

            self.instrs.append({"valu": [("+", v_idx, v_tmp1, v_tmp2)]})

            self.instrs.append({"valu": [("<", v_tmp1, v_idx, v_n_nodes)]})

            self.instrs.append({"flow": [("vselect", v_idx, v_tmp1, v_idx, v_consts[0])]})

            if iter_idx < total_iters - 1:
                next_ic = i_consts[all_offsets[iter_idx + 1]]
                self.instrs.append({
                    "store": [
                        ("vstore", tmp_addr, v_idx),
                        ("vstore", tmp_addr2, v_val),
                    ],
                    "alu": [
                        ("+", tmp_addr, iip, next_ic),
                        ("+", tmp_addr2, ivp, next_ic),
                    ],
                })
            else:
                self.instrs.append({"store": [
                    ("vstore", tmp_addr, v_idx),
                    ("vstore", tmp_addr2, v_val),
                ]})

        self.add("flow", ("pause",))
                

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
