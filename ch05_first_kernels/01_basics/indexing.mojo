from std.gpu import thread_idx, block_idx, block_dim, grid_dim
from std.gpu.host import DeviceContext

def who_am_i():
    var block_id = 
        block_idx.x +
        block_idx.y * grid_dim.x +
        block_idx.z * grid_dim.x * grid_dim.y
    
    var block_offset =
        block_id *
        block_dim.x * block_dim.y * block_dim.z

    var thread_offset =
        thread_idx.x +
        thread_idx.y * block_dim.x +
        thread_idx.z * block_dim.x * block_dim.y

    var id = block_offset + thread_offset

    print("{} | Block({} {} {}) = {} | Thread({} {} {}) = {}".format(id, 
        block_idx.x, block_idx.y, block_idx.z, block_id, 
        thread_idx.x, thread_idx.y, thread_idx.z, thread_offset))

    

def main() raises:
    var bx, by, bz = 2, 3, 4
    var tx, ty, tz = 4, 4, 4

    var blocks_per_grid = bx * by * bz
    var threads_per_block = tx * ty * tz

    print("{} blocks/grid".format(blocks_per_grid))
    print("{} threads/block".format(threads_per_block))
    print("{} total threads".format(threads_per_block * blocks_per_grid))

    ctx = DeviceContext()
    ctx.enqueue_function[who_am_i, who_am_i](
        grid_dim=(bx, by, bz),
        block_dim=(tx, ty, tz)
    )
    ctx.synchronize()
