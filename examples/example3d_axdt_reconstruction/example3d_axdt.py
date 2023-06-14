import pyelsa as elsa
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def read_geometry(vol_desc, size):
    indexed_angles = np.loadtxt("angles.csv", comments="#", delimiter=" ")

    angles = indexed_angles[:, [2, 1, 3]].copy()
    geos = []
    dst_src_center = 1000000.0
    dst_detec_center = 1.0
    range_desc = elsa.VolumeDescriptor([size, size, angles.shape[0]])

    for i in range(angles.shape[0]):
        rot_mat = R.from_euler("YZY", angles[i], degrees=True).as_matrix()
        geos.append(elsa.Geometry(dst_src_center, dst_detec_center,
                                  vol_desc,
                                  range_desc,
                                  rot_mat
                                  )
                    )
    return range_desc, geos


def read_weighted_recon_dirs():
    recon_dirs_weights_comb = np.loadtxt("vecs.csv", comments="#", delimiter=" ")
    assert (recon_dirs_weights_comb.shape[1] == 4)
    recon_dirs = recon_dirs_weights_comb[:, :3].copy()
    recon_weights = recon_dirs_weights_comb[:, -1:].reshape(-1).copy()
    return recon_dirs, recon_weights


def example3d():
    size = 32  # size on one dimension of the volume to be reconstructed

    vol_sz = [size, size, size]  # volume size

    phantom = elsa.phantoms.modifiedSheppLogan(vol_sz)  # one 3d-slice

    num_slice = 15  # spherical harmonics of order 4 will result into 15 3d-slice

    vol_desc = elsa.IdenticalBlocksDescriptor(num_slice, phantom.getDataDescriptor())
    vol_dc_raw = np.repeat(np.array(phantom)[np.newaxis, :, :], num_slice, 0).swapaxes(1, 3).flatten()
    vol_dc = elsa.DataContainer(vol_desc, vol_dc_raw)
    # repeat the 3d-slice for 15 times to mimic the actual data to be reconstructed

    range_desc, geos = read_geometry(vol_desc, size)
    # get info about the detector (range descriptor, geometries)
    sense_dir = [1., 0., 0.]
    # sensitivity vector of the grating
    xgi_desc = elsa.XGIDetectorDescriptor(range_desc.getNumberOfCoefficientsPerDimension(),
                                          range_desc.getSpacingPerDimension(),
                                          geos,
                                          sense_dir, True)
    # create the detector descriptor

    recon_dirs, recon_weights = read_weighted_recon_dirs()
    # get the reconstruction pattern

    sph_degree = 4
    # spherical harmonics of degree 4 should be enough
    projector = elsa.JosephsMethodf(phantom.getDataDescriptor(), xgi_desc)
    # create the projector matrix
    axdt_op = elsa.AXDTOperator(phantom.getDataDescriptor(), xgi_desc, projector,
                                recon_dirs, recon_weights,
                                elsa.AXDTOperator.Symmetry.Even, sph_degree)
    # create the actual dark-field operator using the projector matrix

    projection = axdt_op.apply(vol_dc)
    # projection data used for reconstruction

    eps = 1e-2
    iters = 20
    solver = elsa.CGLS(axdt_op, projection, eps * eps)
    result = solver.solve(iters)
    # do the actual reconstruction

    diff = axdt_op.apply(result) - projection

    pick_a_projection = np.random.randint(range_desc.getNumberOfCoefficientsPerDimension()[2])
    plt.imshow(np.array(axdt_op.apply(result))[:, :, pick_a_projection])
    plt.show()
    # display a picked reconstructed projection
    plt.imshow(np.array(projection)[:, :, pick_a_projection])
    plt.show()
    # display the corresponding real projection
    plt.imshow(np.array(diff)[:, :, pick_a_projection])
    plt.show()
    # display the difference

    pick_a_layer = np.random.randint(size)
    plt.imshow(np.array(result)[:, :, pick_a_layer, 0])
    plt.show()
    # display a picked 2d layer of reconstructed coefficients of the 3d volume
    # which 3d slice does not matter since we handcrafted the ground truth to be the same on all slices

    plt.imshow(np.array(phantom)[:, :, pick_a_layer])
    plt.show()
    # display the corresponding real 2d layer


if __name__ == '__main__':
    example3d()
