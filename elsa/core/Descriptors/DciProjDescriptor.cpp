#include "DciProjDescriptor.h"

namespace elsa
{

    DciProjDescriptor::DciProjDescriptor(const DataDescriptor& desc,
                                         const std::vector<Geometry>& geom, const DirVec& sensDir,
                                         bool parallelBeamFlag)
        : IdenticalBlocksDescriptor(geom.size(), desc),
          _geom(geom),
          _sensDir(sensDir.normalized()),
          _parallelBeamFlag(parallelBeamFlag)
    {
    }

    std::vector<Geometry>& DciProjDescriptor::getGeometryList()
    {
        return _geom;
    }

    const std::vector<Geometry>& DciProjDescriptor::getGeometryList() const
    {
        return _geom;
    }

    const DciProjDescriptor::DirVec& DciProjDescriptor::getSensDir() const
    {
        return _sensDir;
    }

    bool DciProjDescriptor::isParallelBeam() const
    {
        return _parallelBeamFlag;
    }

    DciProjDescriptor* DciProjDescriptor::cloneImpl() const
    {
        return new DciProjDescriptor(IdenticalBlocksDescriptor::getDescriptorOfBlock(0), _geom,
                                     _sensDir, _parallelBeamFlag);
    }

} // namespace elsa
