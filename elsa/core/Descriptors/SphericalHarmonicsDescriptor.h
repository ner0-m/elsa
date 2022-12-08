#pragma once

#include "DataDescriptor.h"

namespace elsa {

/**
 * \brief  Class for metadata of spherical harmonics
 *
 * \author Max Endrass (endrass@cs.tum.edu), most boilerplate code courtesy of Matthias Wieczorek
 * \author Matthias Wieczore (wieczore@cs.tum.edu), logic merge with XTTSphericalHarmonicsDescriptor and fixes for symmetry cases
 * \author Nikola Dinev (nikola.dinev@tum.de), port to elsa
 *
 */
class SphericalHarmonicsDescriptor : public DataDescriptor{
public:
    // odd for odd functions, even for even functions and regular if the function is neither of those
    // for regular functions, no degrees will be skipped
    enum SYMMETRY { even, odd, regular };

protected:

    SYMMETRY _symmetry;
    size_t _maxDegree;

public:
    /**
    * \brief Constructor for SphericalHarmonicsDescriptor
    *
    * \param[in] maxDegree maximum degree of spherical harmonics to be used
    *
    * Automatically sets the number of coefficients according to the max degree
    */
    SphericalHarmonicsDescriptor(size_t maxDegree, SYMMETRY symmetry);

    ~SphericalHarmonicsDescriptor() {}

    virtual SphericalHarmonicsDescriptor* cloneImpl() const override;

    index_t getMaxDegree() const;

    SYMMETRY getSymmetry() const;

};
} //namespace elsa
