import torch
import igl

import numpy as np

from DualMeshUDF_core import Octree, triangulate_faces

def extract_mesh(
        udf_grid,
        udf_grad,
        max_depth=7,
        udf_threshold=1e-7,
        boxmin=-1.,
        boxmax=1.,
        sample_udf_threshold=1e-16 # min bound for UDF values
):
    """
    Extract the mesh from a UDF
    Parameters
    ------------
    udf_func : udf function
    udf_grad_func : udf with gradient
    batch_size: batch size for inferring the UDF network
    max_depth: the max depth of the octree, e.g., max_depth=7 stands for resolution of 128^3
    """

    octree = Octree(max_depth=max_depth,
                    min_corner=np.array([[boxmin], [boxmin], [boxmin]]),
                    max_corner=np.array([[boxmax], [boxmax], [boxmax]]),
                    sampling_depth=1)

    cur_depth = 0
    voxel_res = np.exp2(max_depth)
    voxel_size = (boxmax - boxmin) / voxel_res

    while cur_depth <= max_depth:

        # get centroids of the nodes in the current depth
        centroids_coords = octree.centroids_of_new_nodes()
        # print(centroids_coords)

        # query udf values and gradients for the centroids
        centroid_udfs, centroid_grads = query_udf_and_grad(udf_grid, udf_grad, centroids_coords, voxel_res, boxmin, boxmax)
        print(centroid_udfs.shape, centroid_grads.shape)

        # adaptively subdivide the cells
        octree.adaptive_subdivide(centroid_udfs, centroid_grads, udf_threshold)

        cur_depth += 1

    new_grid_indices, new_grid_coords = octree.get_samples_of_new_nodes()

    # query udf values and gradients for the samples
    new_grid_udfs, new_grid_grads = query_udf_and_grad(udf_grid, udf_grad, new_grid_coords, voxel_res, boxmin, boxmax)
    print(new_grid_udfs.shape, new_grid_grads.shape)
    
    octree.set_new_grid_data(new_grid_indices, new_grid_udfs, new_grid_grads)

    # check if projections are reliable
    indices, projections = octree.get_projections_for_checking_validity()
    projection_udfs = query_udf(udf_grid, projections, voxel_res, boxmin, boxmax)
    print(projection_udfs.shape, np.sum(projection_udfs < udf_threshold), np.sum(projection_udfs > 1e-16))
    octree.set_grid_validity(indices, projection_udfs < udf_threshold)

    # (reliable udf threshold, corner factor, edge factor, face factor, singular value threshold)
    # octree.batch_solve(sample_udf_threshold, 1.0, 1.0, 0.15, 0.08)
    octree.batch_solve(sample_udf_threshold, 0.0, 0.0, 0.0, 0.1)

    octree.generate_mesh()

    tri_faces = triangulate_faces(octree.mesh_v, octree.mesh_f, octree.v_type, octree.mesh_v_dir)
    print(np.array(octree.mesh_v).shape, np.array(octree.mesh_v).dtype, tri_faces.shape, tri_faces.dtype)

    v, _, _, f = igl.remove_duplicate_vertices(np.array(octree.mesh_v), tri_faces, 1e-7)
    print(v.shape, f.shape)

    v, f, _, _ = igl.remove_unreferenced(v, f)
    print(v.shape, f.shape)
    # print(v, f)

    return v, f


def query_udf(udf_grid, coords, voxel_res, boxmin, boxmax):
    '''
    udf_grid should be a NxNxN grid
    udf_grid should be a NxNxN grid
    voxel_res should be the number of UDF cells
    boxmin should be the normalized min bound of the UDF grid
    coords should be N*3, where N is the number of query points, note that the input
        coords are float numbers in range [-1, 1], which should be converted into 
        integer in range [0, N-1]
    '''
    
    input_shape = list(coords.shape)
    query_points = (coords.reshape(-1, 3) - boxmin) / (boxmax - boxmin) * voxel_res # N, 3
    query_points = query_points.astype(int)
    # print(coords)
    # print(query_points)
    
    query_i, query_j, query_k = query_points.T
    d = udf_grid[query_i, query_j, query_k]

    input_shape[-1] = 1
    d = d.reshape(input_shape)
    
    return d


def query_udf_and_grad(udf_grid, udf_grad, coords, voxel_res, boxmin, boxmax):
    '''
    udf_grid should be a NxNxN grid
    udf_grad should be a NxNxNx3 grid
    voxel_res should be the number of UDF cells
    boxmin should be the normalized min bound of the UDF grid
    coords should be N*3, where N is the number of query points, note that the input
        coords are float numbers in range [-1, 1], which should be converted into 
        integer in range [0, N-1]
    '''

    input_shape = list(coords.shape)
    query_points = (coords.reshape(-1, 3) - boxmin) / (boxmax - boxmin) * voxel_res # N, 3
    query_points = query_points.astype(int)
    # print(coords)
    # print(query_points)

    query_i, query_j, query_k = query_points.T
    d = udf_grid[query_i, query_j, query_k]
    g = udf_grad[query_i, query_j, query_k]
    
    g = g.reshape(input_shape)
    input_shape[-1] = 1
    d = d.reshape(input_shape)
    
    return d, g

