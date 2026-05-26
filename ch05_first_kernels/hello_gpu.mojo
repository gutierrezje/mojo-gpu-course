from std.gpu.host import DeviceContext
from std.gpu import block_idx, thread_idx
from std.sys import has_accelerator

def print_threads():
    print(block_idx.x, thread_idx.x)

def main() raises:
    comptime if not has_accelerator():
        print("No GPU accelerator found on this system.")
    else:
        var ctx = DeviceContext()
        print("block\tthread")
        ctx.enqueue_function[print_threads, print_threads](
            grid_dim=2,
            block_dim=64
        )
        ctx.synchronize()
        print("done")