
import tensorflow as tf


def get_reference_grid(grid_size):
    # grid_size: [batch_size, height, width]
    grid = tf.cast(tf.stack(tf.meshgrid(
                        tf.range(grid_size[1]),
                        tf.range(grid_size[2]),
                        indexing='ij'), axis=2), dtype=tf.float32)
    return tf.tile(tf.expand_dims(grid, axis=0), [grid_size[0],1,1,1])


def warp_grid(grid, transform):
    # grid: [batch, height, width, 2]
    # transform: [batch, 3, 3]
    batch_size, height, width = grid.shape[0:3]
    grid = tf.concat([tf.reshape(grid,[batch_size,height*width,2]), 
                    tf.ones([batch_size,height*width,1])], axis=2)
    grid_warped = tf.matmul(grid, transform)
    return tf.reshape(grid_warped[...,:2], [batch_size,height,width,2])


def resample_linear(grid_data, sample_grids):
    # grid_data: [batch, height, width]
    # sample_grids: [batch, height, width, 2]    
    batch_size, height, width = (grid_data.shape[:])
    sample_coords = tf.reshape(sample_grids, [batch_size,-1,2])
    # pad to replicate the boundaries 1-ceiling, 2-floor
    sample_coords = tf.stack([tf.clip_by_value(sample_coords[...,0],0,height-1),
                            tf.clip_by_value(sample_coords[...,1],0,width-1)], axis=2)
    i1 = tf.cast(tf.math.ceil(sample_coords[...,0]), dtype=tf.int32)
    j1 = tf.cast(tf.math.ceil(sample_coords[...,1]), dtype=tf.int32)
    i0 = tf.maximum(i1-1, 0)
    j0 = tf.maximum(j1-1, 0)
    # four data points q_ij
    q00 = tf.gather_nd(grid_data,tf.stack([i0,j0],axis=2), batch_dims=1)
    q01 = tf.gather_nd(grid_data,tf.stack([i0,j1],axis=2), batch_dims=1)
    q11 = tf.gather_nd(grid_data,tf.stack([i1,j1],axis=2), batch_dims=1)
    q10 = tf.gather_nd(grid_data,tf.stack([i1,j0],axis=2), batch_dims=1)    
    # weights with normalised local coordinates
    wi1 = sample_coords[...,0] - tf.cast(i0,dtype=tf.float32)
    wi0 = 1 - wi1
    wj1 = sample_coords[...,1] - tf.cast(j0,dtype=tf.float32)
    wj0 = 1 - wj1
    return tf.reshape(q00*wi0*wj0 + q01*wi0*wj1 + q11*wi1*wj1 + q10*wi1*wj0, [batch_size,height,width])


def random_transform_generator(batch_size, corner_scale=.1):
    # righ-multiplication affine
    ori_corners = tf.tile([[[1.,1.], [1.,-1.], [-1.,1.], [-1.,-1.]]], [batch_size,1,1])
    new_corners = ori_corners + tf.random.uniform([batch_size,4,2], -corner_scale, corner_scale)    
    ori_corners = tf.concat([ori_corners,tf.ones([batch_size,4,1])], axis=2)
    new_corners = tf.concat([new_corners,tf.ones([batch_size,4,1])], axis=2)
    return tf.stack([tf.linalg.lstsq(ori_corners[n],new_corners[n]) for n in range(batch_size)], axis=0)
