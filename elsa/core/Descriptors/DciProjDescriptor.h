#pragma once

#include "IdenticalBlocksDescriptor.h"
#include "Geometry.h"

namespace elsa
{
/**
 * \brief Class for metadata of Dark-field projection data
 *
 * \author Matthias Wieczorek (wieczore@cs.tum.edu)
 *
 * \tparam data_t data type
 */
class DciProjDescriptor: public IdenticalBlocksDescriptor
{
public:
    typedef Eigen::Matrix<real_t, 3, 1> DirVec;

    /// desc is supposed to be a Descriptor describing a single projection
    DciProjDescriptor(const DataDescriptor& desc, const std::vector<Geometry>& geom, const DirVec& sensDir = DirVec(1, 0, 0), bool parallelBeamFlag = true);

    ~DciProjDescriptor(){}

    std::vector<Geometry>& getGeometryList();

    const std::vector<Geometry>& getGeometryList() const;

    bool isParallelBeam() const;

    const DirVec &getSensDir() const;

    virtual DciProjDescriptor* cloneImpl() const override;

private:
    std::vector<Geometry> _geom;

    const DirVec _sensDir;  ///< sensitivity direction (inplane orthogonal vector to the gratings)
    bool _parallelBeamFlag; ///< indicating if the geometry is assumed to represent an approximation to a parallel beam geometry
};
} // namespace elsa