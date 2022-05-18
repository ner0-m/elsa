#include "JosephsMethodCUDA.h"
#include "Intersection.h"
#include "PlanarDetectorDescriptor.h"
#include "LogGuard.h"
#include "Timer.h"
#include "TypeCasts.hpp"

namespace elsa
{
    template <typename data_t>
    struct CUDAVariablesJosephsForward : public CUDAVariablesForward {
        CUDAVariablesJosephsForward(TextureWrapper&& tex, CudaArrayWrapper&& arr,
                                    PitchedPtrWrapper<data_t>&& ptr)
            : CUDAVariablesForward(),
              volumeTex(std::move(tex)),
              dvolumeArr(std::move(arr)),
              dsinoPtr(std::move(ptr))
        {
        }

        CUDAVariablesJosephsForward(CUDAVariablesJosephsForward<data_t>&& other)
            : CUDAVariablesForward(),
              volumeTex(std::move(other.volumeTex)),
              dvolumeArr(std::move(other.dvolumeArr)),
              dsinoPtr(std::move(other.dsinoPtr))
        {
        }

        TextureWrapper volumeTex;
        CudaArrayWrapper dvolumeArr;
        PitchedPtrWrapper<data_t> dsinoPtr;
    };

    template <typename data_t>
    struct CUDAVariablesJosephsForwardConstrained : public CUDAVariablesJosephsForward<data_t> {
        CUDAVariablesJosephsForwardConstrained(std::unique_ptr<CUDAVariablesForward>&& base,
                                               PinnedArray<data_t>&& volume,
                                               PinnedArray<data_t>&& sinogram)
            : CUDAVariablesJosephsForward<data_t>(
                dynamic_cast<CUDAVariablesJosephsForward<data_t>&&>(*base.release())),
              pvolume(std::move(volume)),
              psino(std::move(sinogram))
        {
        }

        PinnedArray<data_t> pvolume;
        PinnedArray<data_t> psino;
    };

    template <typename data_t>
    JosephsMethodCUDA<data_t>::JosephsMethodCUDA(const VolumeDescriptor& domainDescriptor,
                                                 const DetectorDescriptor& rangeDescriptor,
                                                 bool fast, int device)
        : CUDAProjector<data_t>(domainDescriptor, rangeDescriptor, device),
          _traverse2D{},
          _traverse3D{},
          _fast{fast}
    {
        auto dim = static_cast<std::size_t>(_volumeDescriptor.getNumberOfDimensions());
        if (dim != static_cast<std::size_t>(_detectorDescriptor.getNumberOfDimensions())) {
            throw LogicError(
                std::string("JosephsMethodCUDA: domain and range dimension need to match"));
        }

        if (dim != 2 && dim != 3) {
            throw LogicError("JosephsMethodCUDA: only supporting 2d/3d operations");
        }

        if (_detectorDescriptor.getNumberOfGeometryPoses() == 0) {
            throw LogicError("JosephsMethodCUDA: geometry list was empty");
        }

        const index_t numGeometry = _detectorDescriptor.getNumberOfGeometryPoses();

        std::scoped_lock lock(deviceLock);
        gpuErrchk(cudaSetDevice(_device));

        // allocate device memory and copy ray origins and the inverse of projection matrices to
        // device
        cudaPitchedPtr rayOrigins;
        if (cudaMallocPitch(&rayOrigins.ptr, &rayOrigins.pitch, dim * sizeof(real_t),
                            asUnsigned(numGeometry))
            != cudaSuccess)
            throw std::bad_alloc();

        rayOrigins.xsize = dim;
        rayOrigins.ysize = asUnsigned(numGeometry);

        cudaExtent extent = make_cudaExtent(dim * sizeof(real_t), dim, asUnsigned(numGeometry));
        cudaPitchedPtr projInvMatrices;
        cudaPitchedPtr projMatrices;
        gpuErrchk(cudaMalloc3D(&projInvMatrices, extent));

        if (_fast)
            gpuErrchk(cudaMalloc3D(&projMatrices, extent));

        auto projPitch = projInvMatrices.pitch;
        auto* rayBasePtr = (int8_t*) rayOrigins.ptr;
        auto rayPitch = rayOrigins.pitch;

        for (unsigned i = 0; i < numGeometry; i++) {
            auto geometry = _detectorDescriptor.getGeometryAt(i);

            if (!geometry)
                throw LogicError("JosephsMethodCUDA: Access not existing geometry pose");

            RealMatrix_t P = geometry->getInverseProjectionMatrix().block(0, 0, dim, dim);
            auto* slice = (int8_t*) projInvMatrices.ptr + i * projPitch * dim;

            // transfer inverse of projection matrix
            if (cudaMemcpy2D(slice, projPitch, P.data(), dim * sizeof(real_t), dim * sizeof(real_t),
                             dim, cudaMemcpyDefault)
                != cudaSuccess)
                throw LogicError(
                    "JosephsMethodCUDA: Could not transfer inverse projection matrices to GPU.");

            // transfer projection matrix if _fast flag is set
            if (_fast) {
                P = geometry->getProjectionMatrix().block(0, 0, dim, dim);
                slice = (int8_t*) projMatrices.ptr + i * projPitch * dim;
                if (cudaMemcpy2D(slice, projPitch, P.data(), dim * sizeof(real_t),
                                 dim * sizeof(real_t), dim, cudaMemcpyDefault)
                    != cudaSuccess)
                    throw LogicError("JosephsMethodCUDA: Could not transfer "
                                     "projection matrices to GPU.");
            }

            int8_t* rayPtr = rayBasePtr + i * rayPitch;
            // get ray origin using direct inverse
            RealVector_t ro =
                -geometry->getInverseProjectionMatrix().block(0, 0, dim, dim)
                * geometry->getProjectionMatrix().block(0, static_cast<index_t>(dim), dim, 1);
            // transfer ray origin
            if (cudaMemcpyAsync(rayPtr, ro.data(), dim * sizeof(real_t), cudaMemcpyDefault)
                != cudaSuccess)
                throw LogicError("JosephsMethodCUDA: Could not transfer ray origins to GPU.");
        }

        if (dim == 2) {
            if (fast) {
                _traverse2D = std::make_unique<TraverseJosephsCUDA<data_t, 2>>(
                    _volumeDescriptor.getNumberOfCoefficientsPerDimension(),
                    _detectorDescriptor.getNumberOfCoefficientsPerDimension(), rayOrigins,
                    projInvMatrices, projMatrices);
            } else {
                _traverse2D = std::make_unique<TraverseJosephsCUDA<data_t, 2>>(
                    _volumeDescriptor.getNumberOfCoefficientsPerDimension(),
                    _detectorDescriptor.getNumberOfCoefficientsPerDimension(), rayOrigins,
                    projInvMatrices);
            }
        } else if (dim == 3) {
            if (fast) {
                _traverse3D = std::make_unique<TraverseJosephsCUDA<data_t, 3>>(
                    _volumeDescriptor.getNumberOfCoefficientsPerDimension(),
                    _detectorDescriptor.getNumberOfCoefficientsPerDimension(), rayOrigins,
                    projInvMatrices, projMatrices);
            } else {
                _traverse3D = std::make_unique<TraverseJosephsCUDA<data_t, 3>>(
                    _volumeDescriptor.getNumberOfCoefficientsPerDimension(),
                    _detectorDescriptor.getNumberOfCoefficientsPerDimension(), rayOrigins,
                    projInvMatrices);
            }
        }
    }

    template <typename data_t>
    BoundingBox JosephsMethodCUDA<data_t>::constrainProjectionSpace2(
        const BoundingBox& imagePatch, const std::vector<Interval>& poses) const
    {
        const Eigen::Matrix<real_t, 3, 1> boxDims =
            _volumeDescriptor.getNumberOfCoefficientsPerDimension().template cast<real_t>();
        RealMatrix_t V(12, 3);
        RealMatrix_t e(12, 3);

        V << 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1,
            0, 1, 0, 0, 1, 1, 0;
        e << 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
            1, 0, 0, 1, 0, 0, 1;
        V = V.array().rowwise() * boxDims.transpose().array();
        e = e.array().rowwise() * boxDims.transpose().array();

        const auto intersectionWithPlane =
            [&V, &e](const Eigen::Hyperplane<real_t, 3>& plane) -> std::vector<RealVector_t> {
            RealVector_t lambda =
                (-plane.offset() - (V * plane.normal()).array()) / (e * plane.normal()).array();

            std::vector<RealVector_t> vertices;
            if ((lambda.array() < static_cast<real_t>(0) || lambda.array() > static_cast<real_t>(1))
                    .all()) {
                // plane does not intersect volume, return position of plane wrt. AABB
                RealVector_t sign =
                    Eigen::Map<Eigen::Array<real_t, 3, 1>, 0, Eigen::Stride<4, 0>>(lambda.data())
                        .sign();
                vertices.push_back(sign);
            } else {
                for (int i = 0; i < lambda.rows(); i++) {
                    if (lambda[i] >= static_cast<real_t>(0) && lambda[i] <= static_cast<real_t>(1))
                        vertices.emplace_back(V.row(i) + lambda[i] * e.row(i));
                }
            }

            return vertices;
        };

        const auto planeFromRays = [](const DetectorDescriptor::Ray& a,
                                      const DetectorDescriptor::Ray& b) {
            Eigen::Vector3f ad = a.direction();
            Eigen::Vector3f bd = b.direction();
            const RealVector_t normal = ad.cross(bd).normalized();
            return Eigen::Hyperplane<real_t, 3>(normal, a.origin());
        };

        RealVector_t imageSlice = imagePatch._max - imagePatch._min;
        index_t sliceDir;
        auto sliceSize = imageSlice.minCoeff(&sliceDir);
        index_t planeDir = 1 - sliceDir;

        RealVector_t boxMin = boxDims;
        RealVector_t boxMax = RealVector_t::Zero(3);

        for (const auto& [iStart, iEnd] : poses) {
            for (index_t pose = iStart; pose < iEnd; pose++) {

                if (sliceSize == 1) {
                    // dealing with a single row of pixels, only one plane to intersect with
                    IndexVector_t p1(3);
                    p1.topRows(2) = imagePatch._min.template cast<index_t>();
                    p1[2] = pose;
                    const auto a = _detectorDescriptor.computeRayFromDetectorCoord(p1);
                    IndexVector_t p2(3);
                    p2.topRows(2) = imagePatch._max.template cast<index_t>().array() - 1;
                    p2[2] = pose;
                    const auto b = _detectorDescriptor.computeRayFromDetectorCoord(p2);

                    const auto verts = intersectionWithPlane(planeFromRays(a, b));
                    if (verts.size() > 1) {
                        for (const auto& vertex : verts) {
                            boxMin = (vertex.array() < boxMin.array()).select(vertex, boxMin);
                            boxMax = (vertex.array() > boxMax.array()).select(vertex, boxMax);
                        }
                    }
                } else {
                    // many rows of pixels, compute intersection with top and bottom planes
                    std::vector<RealVector_t> verts[2]{{}, {}};

                    // bottom plane
                    IndexVector_t p1(3);
                    p1.topRows(2) = imagePatch._min.template cast<index_t>();
                    p1[2] = pose;
                    auto a = _detectorDescriptor.computeRayFromDetectorCoord(p1);
                    IndexVector_t p2(3);
                    p2.topRows(2) = imagePatch._min.template cast<index_t>();
                    p2[planeDir] = static_cast<index_t>(imagePatch._max[planeDir]) - 1;
                    p2[2] = pose;
                    auto b = _detectorDescriptor.computeRayFromDetectorCoord(p2);
                    verts[0] = intersectionWithPlane(planeFromRays(a, b));

                    // top plane
                    p1[sliceDir] = static_cast<index_t>(imagePatch._max[sliceDir]) - 1;
                    a = _detectorDescriptor.computeRayFromDetectorCoord(p1);
                    p2[sliceDir] = static_cast<index_t>(imagePatch._max[sliceDir]) - 1;
                    b = _detectorDescriptor.computeRayFromDetectorCoord(p2);
                    verts[1] = intersectionWithPlane(planeFromRays(a, b));

                    if (verts[0].size() == 1 && verts[1].size() == 1) {
                        // neither plane intersects volume
                        if (verts[0][0] != verts[1][0]) {
                            // planes lie on different sides of volume, entire volume required
                            return {IndexVector_t(IndexVector_t::Zero(3)),
                                    _volumeDescriptor.getNumberOfCoefficientsPerDimension()};
                        } // else planes lie on the same side of the volume, no part is required
                    } else {
                        // at least one plane intersects the volume
                        index_t intersecting = verts[0].size() > 1 ? 0 : 1;
                        index_t other = 1 - intersecting;

                        // adjust AABB for interesecting plane
                        for (const auto& vertex : verts[intersecting]) {
                            boxMin = (vertex.array() < boxMin.array()).select(vertex, boxMin);
                            boxMax = (vertex.array() > boxMax.array()).select(vertex, boxMax);
                        }

                        if (verts[other].size() == 1) {
                            // other plane does not intersect volume
                            // expanding AABB to max in direction of plane
                            const auto& planeDir = verts[other][0];
                            boxMin = (planeDir.array() < 0).select(RealVector_t::Zero(3), boxMin);
                            boxMax = (planeDir.array() > 0).select(boxDims, boxMax);
                        } else {
                            // other plane also intersects volume
                            // adjust AABB
                            for (const auto& vertex : verts[other]) {
                                boxMin = (vertex.array() < boxMin.array()).select(vertex, boxMin);
                                boxMax = (vertex.array() > boxMax.array()).select(vertex, boxMax);
                            }
                        }
                    }
                }
            }
        }

        // nothing hit
        if ((boxMax.array() == 0).all())
            return {RealVector_t(RealVector_t::Zero(3)), RealVector_t(RealVector_t::Ones(3))};

        boxMin = boxMin.array().floor();
        boxMin = (boxMin.array() > 0).select(boxMin.array() - 1, boxMin);
        boxMax = boxMax.array().ceil();
        boxMax = (boxMax.array() < boxDims.array()).select(boxMax.array() + 1, boxMax);

        return {boxMin, boxMax};
    }

    template <typename data_t>
    BoundingBox
        JosephsMethodCUDA<data_t>::constrainProjectionSpace(const BoundingBox& sinogramAABB) const
    {
        const auto numDims = _volumeDescriptor.getNumberOfDimensions();
        if (numDims != 3)
            throw std::logic_error(
                "JosephsMethodCUDA::constrainProjectionSpace: only works with 3d projectors");

        if (sinogramAABB._min.size() != numDims || sinogramAABB._max.size() != numDims)
            throw std::invalid_argument(
                "JosephsMethodCUDA::constrainProjectionSpace: wrong number of dimensions");

        if ((sinogramAABB._min.array() > sinogramAABB._max.array()).any())
            throw std::invalid_argument("JosephsMethodCUDA::constrainProjectionSpace: start "
                                        "coordinate cannot be larger than end coordinate");

        const auto volumeCoeffsPerDim = _volumeDescriptor.getNumberOfCoefficientsPerDimension();
        RealVector_t startCoordinate = RealVector_t::Zero(numDims).array() + 0.5;
        RealVector_t endCoordinate = volumeCoeffsPerDim.array().template cast<real_t>() - 0.5;
        const BoundingBox volumeAABB(startCoordinate, endCoordinate);

        // corners of the image, pose coordinate kept for convenience
        std::vector<IndexVector_t> corners;
        corners.push_back(sinogramAABB._min.template cast<index_t>());
        corners.push_back(sinogramAABB._max.array().template cast<index_t>() - 1);
        corners.push_back(sinogramAABB._min.template cast<index_t>());
        corners.back()[1] = sinogramAABB._max[1] - 1;
        corners.push_back(sinogramAABB._max.array().template cast<index_t>() - 1);
        corners.back()[1] = sinogramAABB._min[1];

        RealVector_t minSlice = volumeCoeffsPerDim.template cast<real_t>();
        RealVector_t maxSlice = RealVector_t::Zero(3);

        for (index_t i = sinogramAABB._min[2]; i < sinogramAABB._max[2]; i++) {
            for (auto& corner : corners) {
                corner[2] = i;
                const auto ray = _detectorDescriptor.computeRayFromDetectorCoord(corner);

                // Logger::get("JosephsMethodCUDA")
                //     ->debug("\tray origin: {} {} {}", ray.origin()[0], ray.origin()[1],
                //             ray.origin()[2]);
                // Logger::get("JosephsMethodCUDA")
                //     ->debug("\tray direction: {} {} {}", ray.direction()[0], ray.direction()[1],
                //             ray.direction()[2]);

                auto result = Intersection::withRay(volumeAABB, ray);
                if (result) {
                    auto entryPoint = ray.pointAt(result->_tmin);
                    auto exitPoint = ray.pointAt(result->_tmax);

                    minSlice = (entryPoint.array() < minSlice.array()).select(entryPoint, minSlice);
                    minSlice = (exitPoint.array() < minSlice.array()).select(exitPoint, minSlice);
                    maxSlice = (entryPoint.array() > maxSlice.array()).select(entryPoint, maxSlice);
                    maxSlice = (exitPoint.array() > maxSlice.array()).select(exitPoint, maxSlice);
                }
            }
        }

        minSlice = (minSlice.array() - 0.5).floor();
        minSlice = (minSlice.array() < 0).select(RealVector_t::Zero(numDims), minSlice);
        maxSlice = maxSlice.array().floor() + 1;
        maxSlice = (maxSlice.array() > volumeCoeffsPerDim.array().template cast<real_t>())
                       .select(volumeCoeffsPerDim.array().template cast<real_t>(), maxSlice);

        Logger::get("JosephsMethodCUDA")
            ->debug("Volume ({}, {}, {}) to ({}, {}, {}) is responsible for sino ({}, {}, {}) to "
                    "({}, {}, {})",
                    minSlice[0], minSlice[1], minSlice[2], maxSlice[0], maxSlice[1], maxSlice[2],
                    sinogramAABB._min[0], sinogramAABB._min[1], sinogramAABB._min[2],
                    sinogramAABB._max[0], sinogramAABB._max[1], sinogramAABB._max[2]);

        return {minSlice, maxSlice};
    }

    template <typename data_t>
    std::unique_ptr<CUDAVariablesForward>
        JosephsMethodCUDA<data_t>::setupCUDAVariablesForward(IndexVector_t chunkSizeDomain,
                                                             IndexVector_t chunkSizeRange) const
    {
        cudaChannelFormatDesc channelDesc;

        if constexpr (sizeof(data_t) == 4)
            channelDesc =
                cudaCreateChannelDesc(sizeof(data_t) * 8, 0, 0, 0, cudaChannelFormatKindFloat);
        else if (sizeof(data_t) == 8)
            channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindSigned);
        else
            throw std::invalid_argument("JosephsMethodCUDA::copyTextureToGPU: only supports "
                                        "DataContainer<data_t> with data_t of length 4 or 8 bytes");

        cudaTextureDesc texDesc;
        std::memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.addressMode[2] = cudaAddressModeClamp;
        texDesc.filterMode = sizeof(data_t) == 4 ? cudaFilterModeLinear : cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        std::scoped_lock lock(deviceLock);
        cudaSetDevice(_device);

        CudaArrayWrapper arr(channelDesc, chunkSizeDomain);

        return std::make_unique<CUDAVariablesJosephsForward<data_t>>(
            TextureWrapper(arr, texDesc), std::move(arr),
            PitchedPtrWrapper<data_t>(chunkSizeRange));
    }

    template <typename data_t>
    std::unique_ptr<CUDAVariablesForward>
        JosephsMethodCUDA<data_t>::setupCUDAVariablesForwardConstrained(
            IndexVector_t chunkSizeDomain, IndexVector_t chunkSizeRange) const
    {
        return std::make_unique<CUDAVariablesJosephsForwardConstrained<data_t>>(
            setupCUDAVariablesForward(chunkSizeDomain, chunkSizeRange),
            PinnedArray<data_t>(chunkSizeDomain), PinnedArray<data_t>(chunkSizeRange));
    }

    template <typename data_t>
    void JosephsMethodCUDA<data_t>::applyConstrained(const DataContainer<data_t>& x,
                                                     DataContainer<data_t>& Ax,
                                                     const ForwardProjectionTask& task,
                                                     CUDAVariablesForward& cudaVars) const
    {
        auto& cuVars = dynamic_cast<CUDAVariablesJosephsForwardConstrained<data_t>&>(cudaVars);

        containerChunkToPinned(x, cuVars.pvolume, task._volumeBox);

        copyDataForward(cuVars.pvolume.get(), BoundingBox(cuVars.pvolume._dims), cuVars);

        std::unique_lock lock(deviceLock);
        gpuErrchk(cudaSetDevice(_device));

        // assumes 3d
        _traverse3D->traverseForwardConstrained(
            cuVars.volumeTex, cuVars.dsinoPtr,
            BoundingBoxCUDA<3>(task._volumeBox._min, task._volumeBox._max),
            BoundingBoxCUDA<2>(task._imagePatch._min, task._imagePatch._max), task._numPoses,
            task._poses, cuVars.stream);

        lock.unlock();

        RealVector_t sinoMin = RealVector_t::Zero(3);
        sinoMin.topRows(2) = task._imagePatch._min;
        RealVector_t sinoMax = RealVector_t::Constant(3, task._numPoses);
        sinoMax.topRows(2) = task._imagePatch._max;

        retrieveResults(cuVars.psino.get(), cuVars.dsinoPtr, BoundingBox(sinoMin, sinoMax),
                        cuVars.stream);
        pinnedToContainerChunks(Ax, cuVars.psino, task._imagePatch, task._poses);
    }

    template <typename data_t>
    JosephsMethodCUDA<data_t>* JosephsMethodCUDA<data_t>::cloneImpl() const
    {
        return new JosephsMethodCUDA(*this);
    }

    template <typename data_t>
    bool JosephsMethodCUDA<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        // TODO: only need Geometry vector stored internally for comparisons, use kernels instead
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherJM = downcast_safe<JosephsMethodCUDA>(&other);
        if (!otherJM)
            return false;

        if (_fast != otherJM->_fast)
            return false;

        return true;
    }

    template <typename data_t>
    void JosephsMethodCUDA<data_t>::applyImpl(const DataContainer<data_t>& x,
                                              DataContainer<data_t>& Ax) const
    {
        Timer timeGuard("JosephsMethodCUDA", "apply");
        auto cudaVars =
            setupCUDAVariablesForward(_volumeDescriptor.getNumberOfCoefficientsPerDimension(),
                                      _detectorDescriptor.getNumberOfCoefficientsPerDimension());

        auto& cuVars = static_cast<CUDAVariablesJosephsForward<data_t>&>(*cudaVars);

        copyDataForward(&x[0], BoundingBox(_volumeDescriptor.getNumberOfCoefficientsPerDimension()),
                        cuVars);

        // synchronize because we are using multiple streams
        cudaStreamSynchronize(cuVars.stream);
        // perform projection
        if (_volumeDescriptor.getNumberOfDimensions() == 3) {
            _traverse3D->traverseForward(cuVars.volumeTex, cuVars.dsinoPtr);

        } else {
            _traverse2D->traverseForward(cuVars.volumeTex, cuVars.dsinoPtr);
        }
        cudaDeviceSynchronize();

        retrieveResults(&Ax[0], cuVars.dsinoPtr,
                        BoundingBox(_detectorDescriptor.getNumberOfCoefficientsPerDimension()),
                        (cudaStream_t) 0);
    }

    template <typename data_t>
    void JosephsMethodCUDA<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                                     DataContainer<data_t>& Aty) const
    {
        Timer timeguard("JosephsMethodCUDA", "applyAdjoint");

        // allocate memory for volume
        auto domainDims = _volumeDescriptor.getNumberOfCoefficientsPerDimension();
        auto domainDimsui = domainDims.template cast<unsigned int>();

        auto rangeDims = _detectorDescriptor.getNumberOfCoefficientsPerDimension();
        auto rangeDimsui = rangeDims.template cast<unsigned int>();

        index_t dim = _volumeDescriptor.getNumberOfDimensions();

        cudaPitchedPtr dvolumePtr;
        if (dim == 3) {
            cudaExtent volExt =
                make_cudaExtent(domainDimsui[0] * sizeof(data_t), domainDimsui[1], domainDimsui[2]);
            if (cudaMalloc3D(&dvolumePtr, volExt) != cudaSuccess)
                throw std::bad_alloc();

            // transfer projections
            if (_fast) {
                auto [sinoTex, sino] = copyTextureToGPU<cudaArrayLayered>(
                    (const void*) &y[0],
                    BoundingBox(_detectorDescriptor.getNumberOfCoefficientsPerDimension()),
                    (cudaStream_t) 0);

                cudaDeviceSynchronize();
                _traverse3D->traverseAdjointFast(dvolumePtr, sinoTex);
                cudaDeviceSynchronize();

                if (cudaDestroyTextureObject(sinoTex) != cudaSuccess)
                    Logger::get("JosephsMethodCUDA")
                        ->error("Couldn't destroy texture object; This may cause problems later.");

                if (cudaFreeArray(sino) != cudaSuccess)
                    Logger::get("JosephsMethodCUDA")
                        ->error("Couldn't free GPU memory; This may cause problems later.");
            } else {
                if (cudaMemset3DAsync(dvolumePtr, 0, volExt) != cudaSuccess)
                    throw LogicError("JosephsMethodCUDA::applyAdjoint: Could not "
                                     "zero-initialize volume on GPU.");

                cudaPitchedPtr dsinoPtr;
                cudaExtent sinoExt = make_cudaExtent(rangeDimsui[0] * sizeof(data_t),
                                                     rangeDimsui[1], rangeDimsui[2]);

                if (cudaMalloc3D(&dsinoPtr, sinoExt) != cudaSuccess)
                    throw std::bad_alloc();

                copy3DDataContainer<ContainerCpyKind::cpyContainerToRawGPU>(
                    (void*) &y[0], dsinoPtr, sinoExt, (cudaStream_t) 0);

                // perform projection

                // synchronize because we are using multiple streams
                cudaDeviceSynchronize();
                _traverse3D->traverseAdjoint(dvolumePtr, dsinoPtr);
                // synchonize because we are using multiple streams
                cudaDeviceSynchronize();

                // free allocated memory
                if (cudaFree(dsinoPtr.ptr) != cudaSuccess)
                    Logger::get("JosephsMethodCUDA")
                        ->error("Couldn't free GPU memory; This may cause problems later.");
            }

            // retrieve results from GPU
            copy3DDataContainer<ContainerCpyKind::cpyRawGPUToContainer, false>(
                (void*) &Aty[0], dvolumePtr, volExt, (cudaStream_t) 0);
        } else {
            // 2D case
            if (cudaMallocPitch(&dvolumePtr.ptr, &dvolumePtr.pitch,
                                domainDimsui[0] * sizeof(data_t), domainDimsui[1])
                != cudaSuccess)
                throw std::bad_alloc();

            if (_fast) {
                // transfer projections
                auto [sinoTex, sino] = copyTextureToGPU<cudaArrayLayered>(
                    (const void*) &y[0],
                    BoundingBox(_detectorDescriptor.getNumberOfCoefficientsPerDimension()),
                    (cudaStream_t) 0);

                cudaDeviceSynchronize();
                _traverse2D->traverseAdjointFast(dvolumePtr, sinoTex);
                cudaDeviceSynchronize();

                if (cudaDestroyTextureObject(sinoTex) != cudaSuccess)
                    Logger::get("JosephsMethodCUDA")
                        ->error("Couldn't destroy texture object; This may cause problems later.");

                if (cudaFreeArray(sino) != cudaSuccess)
                    Logger::get("JosephsMethodCUDA")
                        ->error("Couldn't free GPU memory; This may cause problems later.");
            } else {
                if (cudaMemset2DAsync(dvolumePtr.ptr, dvolumePtr.pitch, 0,
                                      domainDimsui[0] * sizeof(data_t), domainDimsui[1])
                    != cudaSuccess)
                    throw LogicError("JosephsMethodCUDA::applyAdjoint: Could not "
                                     "zero-initialize volume on GPU.");

                cudaPitchedPtr dsinoPtr;
                IndexVector_t rangeDims = _detectorDescriptor.getNumberOfCoefficientsPerDimension();
                if (cudaMallocPitch(&dsinoPtr.ptr, &dsinoPtr.pitch, rangeDimsui[0] * sizeof(data_t),
                                    rangeDimsui[1])
                    != cudaSuccess)
                    throw std::bad_alloc();

                if (cudaMemcpy2DAsync(dsinoPtr.ptr, dsinoPtr.pitch, (void*) &y[0],
                                      rangeDimsui[0] * sizeof(data_t),
                                      rangeDimsui[0] * sizeof(data_t), rangeDimsui[1],
                                      cudaMemcpyDefault)
                    != cudaSuccess)
                    throw LogicError(
                        "JosephsMethodCUDA::applyAdjoint: Couldn't transfer sinogram to GPU.");

                // synchronize because we are using multiple streams
                cudaDeviceSynchronize();
                _traverse2D->traverseAdjoint(dvolumePtr, dsinoPtr);
                // synchonize because we are using multiple streams
                cudaDeviceSynchronize();

                // free allocated memory
                if (cudaFree(dsinoPtr.ptr) != cudaSuccess)
                    Logger::get("JosephsMethodCUDA")
                        ->error("Couldn't free GPU memory; This may cause problems later.");
            }

            // retrieve results from GPU
            if (cudaMemcpy2D((void*) &Aty[0], domainDimsui[0] * sizeof(data_t), dvolumePtr.ptr,
                             dvolumePtr.pitch, sizeof(data_t) * domainDimsui[0], domainDimsui[1],
                             cudaMemcpyDefault)
                != cudaSuccess)
                throw LogicError(
                    "JosephsMethodCUDA::applyAdjoint: Couldn't retrieve results from GPU");
        }

        // free allocated memory
        if (cudaFree(dvolumePtr.ptr) != cudaSuccess)
            Logger::get("JosephsMethodCUDA")
                ->error("Couldn't free GPU memory; This may cause problems later.");
    }

    template <typename data_t>
    JosephsMethodCUDA<data_t>::JosephsMethodCUDA(const JosephsMethodCUDA<data_t>& other)
        : CUDAProjector<data_t>(other),
          _traverse2D{other._traverse2D},
          _traverse3D{other._traverse3D},
          _fast{other._fast}
    {
    }

    template <typename data_t>
    void JosephsMethodCUDA<data_t>::copyDataForward(const data_t* volumeData,
                                                    const BoundingBox& volumeBoundingBox,
                                                    const CUDAVariablesForward& cudaVars) const
    {
        const auto& cuVars = dynamic_cast<const CUDAVariablesJosephsForward<data_t>&>(cudaVars);

        // transfer volume as texture
        copyTextureToGPUForward((const void*) volumeData, volumeBoundingBox, cuVars.dvolumeArr,
                                cuVars.stream);
    }

    template <typename data_t>
    void JosephsMethodCUDA<data_t>::retrieveResults(data_t* hostData, const cudaPitchedPtr& gpuData,
                                                    const BoundingBox& aabb,
                                                    const cudaStream_t& stream) const
    {
        auto dataDims = (aabb._max - aabb._min).template cast<size_t>().eval();

        if (aabb._dim == 3) {
            auto dataExt = make_cudaExtent(dataDims[0] * sizeof(data_t), dataDims[1], dataDims[2]);
            auto gpuAdjusted =
                make_cudaPitchedPtr(gpuData.ptr, gpuData.pitch, dataExt.width, dataExt.height);
            copy3DDataContainer<ContainerCpyKind::cpyRawGPUToContainer, true>(
                (void*) hostData, gpuAdjusted, dataExt, stream);
        } else {
            std::scoped_lock lock(deviceLock);
            gpuErrchk(cudaSetDevice(_device));
            gpuErrchk(cudaMemcpy2DAsync((void*) hostData, dataDims[0] * sizeof(data_t), gpuData.ptr,
                                        gpuData.pitch, dataDims[0] * sizeof(data_t), dataDims[1],
                                        cudaMemcpyDeviceToHost, stream));
        }

        cudaStreamSynchronize(stream);
    }

    template <typename data_t>
    template <typename JosephsMethodCUDA<data_t>::ContainerCpyKind direction, bool async>
    void JosephsMethodCUDA<data_t>::copy3DDataContainer(void* hostData,
                                                        const cudaPitchedPtr& gpuData,
                                                        const cudaExtent& extent,
                                                        const cudaStream_t& stream) const
    {
        cudaMemcpy3DParms cpyParams = {};
        cpyParams.extent = extent;
        cpyParams.kind = cudaMemcpyDefault;
        cudaPitchedPtr tmp =
            make_cudaPitchedPtr(hostData, extent.width, extent.width, extent.height);

        if (direction == ContainerCpyKind::cpyContainerToRawGPU) {
            cpyParams.dstPtr = gpuData;
            cpyParams.srcPtr = tmp;
        } else if (direction == ContainerCpyKind::cpyRawGPUToContainer) {
            cpyParams.srcPtr = gpuData;
            cpyParams.dstPtr = tmp;
        }

        std::unique_lock lock(deviceLock);
        gpuErrchk(cudaSetDevice(_device));

        gpuErrchk(cudaMemcpy3DAsync(&cpyParams, stream));
        lock.unlock();

        if (!async)
            cudaStreamSynchronize(stream);
    }

    template <typename data_t>
    template <unsigned int flags>
    std::pair<cudaTextureObject_t, cudaArray*>
        JosephsMethodCUDA<data_t>::copyTextureToGPU(const void* hostData, const BoundingBox& aabb,
                                                    cudaStream_t stream) const
    {
        // transfer volume as texture
        auto domainDims = aabb._max - aabb._min;
        auto domainDimsui = domainDims.template cast<unsigned int>();

        cudaArray* volume;
        cudaTextureObject_t volumeTex = 0;

        cudaChannelFormatDesc channelDesc;

        if constexpr (sizeof(data_t) == 4)
            channelDesc =
                cudaCreateChannelDesc(sizeof(data_t) * 8, 0, 0, 0, cudaChannelFormatKindFloat);
        else if (sizeof(data_t) == 8)
            channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindSigned);
        else
            throw InvalidArgumentError("JosephsMethodCUDA::copyTextureToGPU: only supports "
                                       "DataContainer<data_t> with data_t of length 4 or 8 bytes");

        if (aabb._dim == 3) {
            cudaExtent volumeExtent =
                make_cudaExtent(domainDimsui[0], domainDimsui[1], domainDimsui[2]);
            if (cudaMalloc3DArray(&volume, &channelDesc, volumeExtent, flags) != cudaSuccess)
                throw std::bad_alloc();
            cudaMemcpy3DParms cpyParams = {};
            cpyParams.srcPtr.ptr = const_cast<void*>(hostData);
            cpyParams.srcPtr.pitch = domainDimsui[0] * sizeof(data_t);
            cpyParams.srcPtr.xsize = domainDimsui[0] * sizeof(data_t);
            cpyParams.srcPtr.ysize = domainDimsui[1];
            cpyParams.dstArray = volume;
            cpyParams.extent = volumeExtent;
            cpyParams.kind = cudaMemcpyDefault;

            if (cudaMemcpy3DAsync(&cpyParams, stream) != cudaSuccess)
                throw LogicError(
                    "JosephsMethodCUDA::copyTextureToGPU: Could not transfer data to GPU.");
        } else {
            // 2D case
            // CUDA has a very weird way of handling layered 1D arrays
            if (flags == cudaArrayLayered) {

                // must be allocated as a 3D Array of height 0
                cudaExtent volumeExtent = make_cudaExtent(domainDimsui[0], 0, domainDimsui[1]);
                gpuErrchk(cudaMalloc3DArray(&volume, &channelDesc, volumeExtent, flags));

                // adjust height to 1 for copy
                volumeExtent.height = 1;
                cudaMemcpy3DParms cpyParams = {};
                cpyParams.srcPos = make_cudaPos(0, 0, 0);
                cpyParams.dstPos = make_cudaPos(0, 0, 0);
                cpyParams.srcPtr =
                    make_cudaPitchedPtr(const_cast<void*>(hostData),
                                        domainDimsui[0] * sizeof(data_t), domainDimsui[0], 1);
                cpyParams.dstArray = volume;
                cpyParams.extent = volumeExtent;
                cpyParams.kind = cudaMemcpyDefault;

                if (cudaMemcpy3DAsync(&cpyParams, stream) != cudaSuccess)
                    throw LogicError(
                        "JosephsMethodCUDA::copyTextureToGPU: Could not transfer data to GPU.");
            } else {
                gpuErrchk(cudaMallocArray(&volume, &channelDesc, domainDimsui[0], domainDimsui[1],
                                          flags));

                if (cudaMemcpy2DToArrayAsync(volume, 0, 0, hostData,
                                             domainDimsui[0] * sizeof(data_t),
                                             domainDimsui[0] * sizeof(data_t), domainDimsui[1],
                                             cudaMemcpyDefault, stream)
                    != cudaSuccess)
                    throw LogicError(
                        "JosephsMethodCUDA::copyTextureToGPU: Could not transfer data to GPU.");
            }
        }

        cudaResourceDesc resDesc;
        std::memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = volume;

        cudaTextureDesc texDesc;
        std::memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = flags ? cudaAddressModeBorder : cudaAddressModeClamp;
        texDesc.addressMode[1] = flags ? cudaAddressModeBorder : cudaAddressModeClamp;
        texDesc.addressMode[2] = flags ? cudaAddressModeBorder : cudaAddressModeClamp;
        texDesc.filterMode = sizeof(data_t) == 4 ? cudaFilterModeLinear : cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        if (cudaCreateTextureObject(&volumeTex, &resDesc, &texDesc, nullptr) != cudaSuccess)
            throw LogicError("Couldn't create texture object");

        return {volumeTex, volume};
    }

    template <typename data_t>
    template <unsigned int flags>
    void JosephsMethodCUDA<data_t>::copyTextureToGPUForward(const void* hostData,
                                                            const BoundingBox& aabb,
                                                            cudaArray_t darray,
                                                            cudaStream_t stream) const
    {
        // transfer volume as texture
        Eigen::Matrix<uint, -1, 1> domainDimsui = (aabb._max - aabb._min).template cast<uint>();

        std::scoped_lock lock(deviceLock);
        gpuErrchk(cudaSetDevice(_device));
        if (aabb._dim == 3) {
            cudaExtent volumeExtent =
                make_cudaExtent(domainDimsui[0], domainDimsui[1], domainDimsui[2]);

            cudaMemcpy3DParms cpyParams = {0};
            cpyParams.srcPtr =
                make_cudaPitchedPtr(const_cast<void*>(hostData), domainDimsui[0] * sizeof(data_t),
                                    domainDimsui[0] * sizeof(data_t), domainDimsui[1]);
            cpyParams.dstArray = darray;
            cpyParams.extent = volumeExtent;
            cpyParams.kind = cudaMemcpyDefault;

            gpuErrchk(cudaMemcpy3DAsync(&cpyParams, stream));

        } else {
            // CUDA has a very weird way of handling layered 1D arrays
            if (flags == cudaArrayLayered) {

                // must be allocated as a 3D Array of height 0
                cudaExtent volumeExtent = make_cudaExtent(domainDimsui[0], 0, domainDimsui[1]);

                // adjust height to 1 for copy
                volumeExtent.height = 1;
                cudaMemcpy3DParms cpyParams = {};
                cpyParams.srcPos = make_cudaPos(0, 0, 0);
                cpyParams.dstPos = make_cudaPos(0, 0, 0);
                cpyParams.srcPtr =
                    make_cudaPitchedPtr(const_cast<void*>(hostData),
                                        domainDimsui[0] * sizeof(data_t), domainDimsui[0], 1);
                cpyParams.dstArray = darray;
                cpyParams.extent = volumeExtent;
                cpyParams.kind = cudaMemcpyHostToDevice;

                gpuErrchk(cudaMemcpy3DAsync(&cpyParams, stream));
            } else {
                gpuErrchk(cudaMemcpy2DToArrayAsync(
                    darray, 0, 0, hostData, domainDimsui[0] * sizeof(data_t),
                    domainDimsui[0] * sizeof(data_t), domainDimsui[1], cudaMemcpyHostToDevice,
                    stream));
            }
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class JosephsMethodCUDA<float>;
    template class JosephsMethodCUDA<double>;
} // namespace elsa
