#include "catch2/catch.hpp"

#include "SiddonsMethod.h"
#include "Geometry.h"

using namespace elsa;

bool isApprox(DataContainer<real_t> x, DataContainer<real_t> y, real_t prec = Eigen::NumTraits<real_t>::dummy_precision()) {
    DataContainer<real_t> z = x;
    z -= y;
    return sqrt(z.squaredL2Norm()) <= prec*sqrt(std::min(x.squaredL2Norm(),y.squaredL2Norm()));
}

SCENARIO("Testing SiddonsMethod projector with only one ray")
{
    IndexVector_t sizeDomain(2);
    sizeDomain << 5, 5;

    IndexVector_t sizeRange(2);
    sizeRange << 1, 1;

    auto domain = DataDescriptor(sizeDomain);
    auto range = DataDescriptor(sizeRange);

    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");

    GIVEN("A SiddonsMethod for 2D and a domain data with all ones")
    {
        std::vector<Geometry> geom;

        auto dataDomain = DataContainer(domain);
        dataDomain = 1;

        auto dataRange = DataContainer(range);
        dataRange = 0;

        WHEN("We have a single ray with 0 degrees")
        {
            geom.emplace_back(100, 5, 0, domain, range);
            auto op = SiddonsMethod(domain, range, geom);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 5, 0, 0,
                        0, 0, 5, 0, 0,
                        0, 0, 5, 0, 0,
                        0, 0, 5, 0, 0,
                        0, 0, 5, 0, 0;

                REQUIRE(DataContainer(domain,cmp) == AtAx);
                REQUIRE(dataRange[0] == Approx(5));
            }
        }

        WHEN("We have a single ray with 180 degrees")
        {
            geom.emplace_back(100, 5, 180 * M_PI / 180., domain, range);
            auto op = SiddonsMethod(domain, range, geom);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 5, 0, 0,
                        0, 0, 5, 0, 0,
                        0, 0, 5, 0, 0,
                        0, 0, 5, 0, 0,
                        0, 0, 5, 0, 0;

                REQUIRE(DataContainer(domain,cmp) == AtAx);
                REQUIRE(dataRange[0] == Approx(5));
            }
        }

        WHEN("We have a single ray with 90 degrees")
        {
            geom.emplace_back(100, 5, 90 * M_PI / 180., domain, range);
            auto op = SiddonsMethod(domain, range, geom);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        5, 5, 5, 5, 5,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0;

                REQUIRE(DataContainer(domain,cmp) == AtAx);
                REQUIRE(dataRange[0] == Approx(5));
            }
        }

        WHEN("We have a single ray with 270 degrees")
        {
            geom.emplace_back(100, 5, 270 * M_PI / 180., domain, range);
            auto op = SiddonsMethod(domain, range, geom);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        5, 5, 5, 5, 5,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0;

                REQUIRE(DataContainer(domain,cmp) == AtAx);
                REQUIRE(dataRange[0] == Approx(5));
            }
        }

// FIXME This does not yield the desired result/if fixed the overall results in a reconstruction is bad
        /*WHEN("We have a single ray with 45 degrees")
        {
            geom.emplace_back(100, 5, 45 * M_PI / 180., domain, range);
            auto op = SiddonsMethod(domain, range, geom);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                std::cout << dataRange.getData().format(CommaInitFmt) << "\n";
                std::cout << AtAx.getData().format(CommaInitFmt) << "\n";

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 10, 0, 0, 0, 0,
                        0, 10, 0, 0, 0,
                        0, 0, 10, 0, 0,
                        0, 0, 0, 10, 0,
                        0, 0, 0, 0, 10;

                REQUIRE(cmp.isApprox(AtAx.getData(), 1e-2));
                REQUIRE(dataRange[0] == Approx(7.071));
            }
        }

        WHEN("We have a single ray with 225 degrees")
        {
            geom.emplace_back(100, 5, 225 * M_PI / 180., domain, range);
            auto op = SiddonsMethod(domain, range, geom);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                std::cout << dataRange.getData().format(CommaInitFmt) << "\n";
                std::cout << AtAx.getData().format(CommaInitFmt) << "\n";

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 10, 0, 0, 0, 0,
                        0, 10, 0, 0, 0,
                        0, 0, 10, 0, 0,
                        0, 0, 0, 10, 0,
                        0, 0, 0, 0, 10;

                REQUIRE(cmp.isApprox(AtAx.getData()));
                REQUIRE(dataRange[0] == Approx(7.071));
            }
        }*/

        // TODO fix this direction, currently it does not work correctly. Consider changing geometry, to mirror stuff
        /*WHEN("We have a single ray with 135 degrees")
        {
            geom.emplace_back(100, 5, 135 * M_PI / 180., domain, range);
            auto op = SiddonsMethod(domain, range, geom);

            THEN("A^t A x should be close to the original data")
            {
                auto AtAx = DataContainer(domain);

                op.apply(dataDomain, dataRange);
                op.applyAdjoint(dataRange, AtAx);

                std::cout << dataRange.getData().format(CommaInitFmt) << "\n";
                std::cout << AtAx.getData().format(CommaInitFmt) << "\n";

                auto cmp = RealVector_t(sizeDomain.prod());
                cmp << 0, 0, 0, 0, 10,
                        0, 0, 0, 10, 0,
                        0, 0, 10, 0, 0,
                        0, 10, 0, 0, 0,
                        10, 0, 0, 0, 0;

                REQUIRE(cmp.isApprox(AtAx.getData()));
                REQUIRE(dataRange[0] == Approx(7.071));
            }
        }*/
    }
}

SCENARIO("Calls to functions of super class") {
    GIVEN("A projector") {
        IndexVector_t volumeDims(2), sinoDims(2);
        const index_t volSize = 50;
        const index_t detectorSize = 50;
        const index_t numImgs = 50;
        volumeDims<<volSize,volSize;
        sinoDims<<detectorSize,numImgs;
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);

        RealVector_t randomStuff(volumeDescriptor.getNumberOfCoefficients());
        randomStuff.setRandom();
        DataContainer volume(volumeDescriptor, randomStuff);
        DataContainer sino(sinoDescriptor);
        std::vector<Geometry> geom;
        for (std::size_t i = 0;i<numImgs;i++) {
            real_t angle = i*2*M_PI/50;
            geom.emplace_back(20*volSize,volSize,angle,volumeDescriptor,sinoDescriptor);
        }
        SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);

        // WHEN("Checking whether projector is a spd operator") {
        //     THEN("Returns false") {
        //         REQUIRE_FALSE(op.isSpd());
        //     }
        // }

        WHEN("Projector is cloned") {
            auto opClone = op.clone();
            auto sinoClone = DataContainer(sino.getDataDescriptor());
            auto volumeClone = DataContainer(volume.getDataDescriptor());

            THEN("Results do not change (may still be slightly different due to summation being performed in a different order)") {
                op.apply(volume,sino);
                opClone->apply(volume,sinoClone);
                REQUIRE( isApprox(sino,sinoClone) );

                op.applyAdjoint(sino,volume);
                opClone->applyAdjoint(sino,volumeClone);
                REQUIRE(isApprox(volume,volumeClone));
            }
        }
    }
}

SCENARIO("Output DataContainer is not zero initialized") {
    GIVEN("A 2D setting") {
        IndexVector_t volumeDims(2), sinoDims(2);
        const index_t volSize = 5;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims<<volSize,volSize;
        sinoDims<<detectorSize,numImgs;
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);
        std::vector<Geometry> geom;
        geom.emplace_back(20*volSize,volSize,0.0,volumeDescriptor,sinoDescriptor);
        SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);

        WHEN("Sinogram conatainer is not zero initialized and we project through an empty volume") {
            volume = 0;
            sino = 1;

            THEN("Result is zero") {
                op.apply(volume,sino);
                DataContainer zero(sinoDescriptor);
                REQUIRE(sino == zero);
            }
        }

        WHEN("Volume container is not zero initialized and we backproject from an empty sinogram") {
            sino = 0;
            volume = 1;

            THEN("Result is zero") {
                op.applyAdjoint(sino,volume);
                DataContainer zero(volumeDescriptor);
                REQUIRE(volume == zero);
            }
        }
    }

    GIVEN("A 3D setting") {
        IndexVector_t volumeDims(3), sinoDims(3);
        const index_t volSize = 3;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims<<volSize,volSize,volSize;
        sinoDims<<detectorSize,detectorSize,numImgs;
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);
        std::vector<Geometry> geom;

        geom.emplace_back(volSize*20,volSize,volumeDescriptor,sinoDescriptor,0);
        SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);

        WHEN("Sinogram conatainer is not zero initialized and we project through an empty volume") {
            volume = 0;
            sino = 1;

            THEN("Result is zero") {
                op.apply(volume,sino);
                DataContainer zero(sinoDescriptor);
                REQUIRE(sino == zero);
            }
        }

        WHEN("Volume container is not zero initialized and we backproject from an empty sinogram") {
            sino = 0;
            volume = 1;

            THEN("Result is zero") {
                op.applyAdjoint(sino,volume);
                DataContainer zero(volumeDescriptor);
                REQUIRE(volume == zero);
            }
        }
    }
}

SCENARIO("Rays not intersecting the bounding box are present") {
    GIVEN("A 2D setting") {
        IndexVector_t volumeDims(2), sinoDims(2);
        const index_t volSize = 5;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims<<volSize,volSize;
        sinoDims<<detectorSize,numImgs;
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);
        volume = 1;
        sino = 1;
        std::vector<Geometry> geom;

        WHEN("Tracing along a y-axis-aligned ray with a negative x-coordinate of origin") {
            geom.emplace_back(20*volSize,volSize,0.0,volumeDescriptor,sinoDescriptor,0.0,-volSize);

            SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);

            THEN("Result of forward projection is zero") {
                op.apply(volume,sino);
                DataContainer zero(sinoDescriptor);
                REQUIRE(sino == zero);

                AND_THEN("Result of backprojection is zero") {
                    op.applyAdjoint(sino,volume);
                    DataContainer zero(volumeDescriptor);
                    REQUIRE(volume == zero);
                }
            }

        }
        
        WHEN("Tracing along a y-axis-aligned ray with a x-coordinate of origin beyond the bounding box") {
            geom.emplace_back(20*volSize,volSize,0.0,volumeDescriptor,sinoDescriptor,0.0,volSize);

            SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);

            THEN("Result of forward projection is zero") {
                op.apply(volume,sino);
                DataContainer zero(sinoDescriptor); REQUIRE(sino==zero);

                AND_THEN("Result of backprojection is zero") {
                    op.applyAdjoint(sino,volume);
                    DataContainer zero(volumeDescriptor); REQUIRE(volume==zero);
                }
            }

            
        }

        WHEN("Tracing along a x-axis-aligned ray with a negative y-coordinate of origin") {
            geom.emplace_back(20*volSize,volSize, M_PI_2,volumeDescriptor,sinoDescriptor,0.0, 0.0, -volSize);

            SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);

            THEN("Result of forward projection is zero") {
                op.apply(volume,sino);
                DataContainer zero(sinoDescriptor); REQUIRE(sino==zero);

                AND_THEN("Result of backprojection is zero") {
                    op.applyAdjoint(sino,volume);
                    DataContainer zero(volumeDescriptor); REQUIRE(volume==zero);
                }
            }

            
        }

        WHEN("Tracing along a x-axis-aligned ray with a y-coordinate of origin beyond the bounding box") {
            geom.emplace_back(20*volSize,volSize, M_PI_2 ,volumeDescriptor,sinoDescriptor,0.0, 0.0, volSize);

            SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);

            THEN("Result of forward projection is zero") {
                op.apply(volume,sino);
                DataContainer zero(sinoDescriptor); REQUIRE(sino==zero);

                AND_THEN("Result of backprojection is zero") {
                    op.applyAdjoint(sino,volume);
                    DataContainer zero(volumeDescriptor); REQUIRE(volume==zero);
                }
            }


        }
    }

    GIVEN("A 3D setting") {
        IndexVector_t volumeDims(3), sinoDims(3);
        const index_t volSize = 5;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims<<volSize,volSize,volSize;
        sinoDims<<detectorSize,detectorSize,numImgs;
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);
        volume = 1;
        sino = 1;
        std::vector<Geometry> geom;

        const index_t numCases = 9;
        real_t alpha[numCases] = {0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0};
        real_t beta[numCases] = {0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0,
                          M_PI_2, M_PI_2, M_PI_2};
        real_t gamma[numCases] = {0.0, 0.0, 0.0,
                                M_PI_2, M_PI_2, M_PI_2,
                                M_PI_2, M_PI_2, M_PI_2};
        real_t offsetx[numCases] = {volSize, 0.0 , volSize,
                                    0.0, 0.0 , 0.0,
                                    volSize, 0.0 , volSize};
        real_t offsety[numCases] = {0.0, volSize, volSize,
                                    volSize, 0.0 , volSize,
                                    0.0, 0.0, 0.0};
        real_t offsetz[numCases] = {0.0, 0.0, 0.0,
                                    0.0, volSize , volSize,
                                    0.0, volSize ,volSize};
        std::string neg[numCases] = {"x","y","x and y",
                                    "y","z","y and z",
                                    "x","z","x and z"};
        std::string ali[numCases] = {"z", "z", "z",
                                     "x", "x", "x",
                                     "y", "y", "y"};

        for (int i=0;i<numCases;i++) {
            WHEN("Tracing along a " + ali[i] +"-axis-aligned ray with negative " + neg[i] + "-coodinate of origin") {
                geom.emplace_back(20*volSize,volSize,volumeDescriptor,sinoDescriptor,gamma[i],beta[i],alpha[i],0.0,0.0,-offsetx[i],-offsety[i],-offsetz[i]);

                SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);

                THEN("Result of forward projection is zero") {
                    op.apply(volume,sino);
                    DataContainer zero(sinoDescriptor); REQUIRE(sino==zero);

                    AND_THEN("Result of backprojection is zero") {
                        op.applyAdjoint(sino,volume);
                        DataContainer zero(volumeDescriptor); REQUIRE(volume==zero);
                    }
                }

                
            }
        }
                
    }
}

SCENARIO("Axis-aligned rays are present")
{
    GIVEN("A 2D setting with a single ray") {
        IndexVector_t volumeDims(2), sinoDims(2);
        const index_t volSize = 5;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims<<volSize,volSize;
        sinoDims<<detectorSize,numImgs;
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);
        std::vector<Geometry> geom;

        const index_t numCases = 4;
        const real_t angles[numCases] = {0.0, M_PI_2, M_PI, 3*M_PI_2};
        Eigen::Matrix<real_t,volSize*volSize,1> backProj[2]; 
        backProj[1] << 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    1, 1, 1, 1, 1,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0;

        backProj[0] << 0, 0, 1, 0, 0,
                    0, 0, 1, 0, 0,
                    0, 0, 1, 0, 0,
                    0, 0, 1, 0, 0,
                    0, 0, 1, 0, 0;   

        for (index_t i=0; i<numCases; i++) {
            WHEN("An axis-aligned ray with an angle of " + std::to_string(angles[i]) + " radians passes through the center of a pixel") {
                geom.emplace_back(volSize*20,volSize,angles[i],volumeDescriptor,sinoDescriptor);
                SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);
                THEN("The result of projecting through a pixel is exactly the pixel value") {       
                    for (index_t j=0; j<volSize;j++) {
                        volume = 0;
                        if (i%2==0)
                            volume(volSize/2,j) = 1;
                        else
                            volume(j,volSize/2) = 1;
                        
                        op.apply(volume,sino);
                        REQUIRE(sino[0]==1);
                    }

                    AND_THEN("The backprojection sets the values of all hit pixels to the detector value") {
                        op.applyAdjoint(sino,volume);

                        REQUIRE(isApprox(volume, DataContainer(volumeDescriptor,backProj[i%2])));
                    }
                }
            }
        }

        WHEN("A y-axis-aligned ray runs along a voxel boundary") {
            geom.emplace_back(volSize*2000,volSize,0.0,volumeDescriptor,sinoDescriptor,0.0,-0.5);
            SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);
            THEN("The result of projecting through a pixel is the value of the pixel with the higher index") {       
                for (index_t j=0; j<volSize;j++) {
                    volume = 0;
                    volume(volSize/2,j) = 1;
                    
                    op.apply(volume,sino);
                    REQUIRE(sino[0] == Approx(1.0));
                }

                AND_THEN("The backprojection yields the exact adjoint") {
                    sino[0] = 1;
                    op.applyAdjoint(sino,volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor,backProj[0])));
                }
            }
        }
        
        WHEN("A y-axis-aligned ray runs along the right volume boundary") {
            //For Siddon's values in the range [0,boxMax) are considered, a ray running along boxMax should be ignored
            geom.emplace_back(volSize*2000,volSize,0.0,volumeDescriptor,sinoDescriptor,0.0, (volSize*0.5) );
            SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);

            THEN("The result of projecting is zero") {       
                volume = 0;
                op.apply(volume,sino);
                REQUIRE(sino[0]==0.0);

                AND_THEN("The result of backprojection is also zero") {
                    sino[0] = 1;

                    op.applyAdjoint(sino,volume);
                    DataContainer zero(volumeDescriptor); REQUIRE(volume==zero);
                }
            }
        }
        
        /**
         * CPU version exhibits slightly different behaviour, ignoring all ray running along a bounding box border
         */
        // backProj[0] <<  1, 0, 0, 0, 0,
        //             1, 0, 0, 0, 0,
        //             1, 0, 0, 0, 0,
        //             1, 0, 0, 0, 0,
        //             1, 0, 0, 0, 0;
        
        // WHEN("A y-axis-aligned ray runs along the left volume boundary") {
        //     geom.emplace_back(volSize*2000,volSize,0.0,volumeDescriptor,sinoDescriptor,0.0,volSize/2.0);
        //     SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);
        //     THEN("The result of projecting through a pixel is exactly the pixel's value") {       
        //         for (index_t j=0; j<volSize;j++) {
        //             volume = 0;
        //             volume(0,j) = 1;
                    
        //             op.apply(volume,sino);
        //             REQUIRE(sino[0] == 1);
        //         }

        //         AND_THEN("The backprojection yields the exact adjoint") {
        //             sino[0] = 1;
        //             op.applyAdjoint(sino,volume);
        //             REQUIRE(volume.getData().isApprox(backProj[0]));
        //         }
        //     }
        // }
    }

    GIVEN("A 3D setting with a single ray") {
        IndexVector_t volumeDims(3), sinoDims(3);
        const index_t volSize = 3;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims<<volSize,volSize,volSize;
        sinoDims<<detectorSize,detectorSize,numImgs;
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);
        std::vector<Geometry> geom;

        const index_t numCases = 6;
        real_t beta[numCases] = {0.0, 0.0, 0.0,
                        0.0,M_PI_2,3*M_PI_2};
        real_t gamma[numCases] = {0.0, M_PI, M_PI_2, 3*M_PI_2,M_PI_2,3*M_PI_2};
        std::string al[numCases] = {"z","-z","x","-x","y","-y"}; 

        Eigen::Matrix<real_t,volSize*volSize*volSize,1> backProj[numCases]; 

        backProj[2] << 0, 0, 0,
                    0, 0, 0,
                    0, 0, 0,
                    
                    0, 1, 0,
                    0, 1, 0,
                    0, 1, 0,
                    
                    0, 0, 0,
                    0, 0, 0,
                    0, 0, 0;  

        backProj[1] << 0, 0, 0,
                    0, 0, 0,
                    0, 0, 0,

                    0, 0, 0,
                    1, 1, 1,
                    0, 0, 0,

                    0, 0, 0,
                    0, 0, 0,
                    0, 0, 0;
        
        backProj[0] << 0, 0, 0,
                    0, 1, 0,
                    0, 0, 0,

                    0, 0, 0,
                    0, 1, 0,
                    0, 0, 0,

                    0, 0, 0,
                    0, 1, 0,
                    0, 0, 0;

        for (index_t i=0; i<numCases; i++) {
            WHEN("A " + al[i] +"-axis-aligned ray passes through the center of a pixel") {
                geom.emplace_back(volSize*20,volSize,volumeDescriptor,sinoDescriptor,gamma[i],beta[i]);
                SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);
                THEN("The result of projecting through a voxel is exactly the voxel value") {       
                    for (index_t j=0; j<volSize;j++) {
                        volume = 0;
                        if (i/2==0)
                            volume(volSize/2,volSize/2,j) = 1;
                        else if(i/2==1)
                            volume(j,volSize/2,volSize/2) = 1;
                        else if(i/2==2)
                            volume(volSize/2,j,volSize/2) = 1;
                        
                        op.apply(volume,sino);
                        REQUIRE(sino[0]==1);
                    }

                    AND_THEN("The backprojection sets the values of all hit voxels to the detector value") {
                        op.applyAdjoint(sino,volume);
                        REQUIRE(isApprox(volume, DataContainer(volumeDescriptor,backProj[i/2])));
                    }
                }
            }
        }

        real_t offsetx[numCases];
        real_t offsety[numCases];

        offsetx[0] = volSize/2.0;
        offsetx[3] = -(volSize/2.0);
        offsetx[1] = 0.0;
        offsetx[4] = 0.0;
        offsetx[5] = -(volSize/2.0);
        offsetx[2] = (volSize/2.0);

        offsety[0] = 0.0;
        offsety[3] = 0.0;
        offsety[1] = volSize/2.0;
        offsety[4] = -(volSize/2.0);
        offsety[5] = -(volSize/2.0);
        offsety[2] = (volSize/2.0);

        backProj[0] << 0, 0, 0,
                    1, 0, 0,
                    0, 0, 0,

                    0, 0, 0,
                    1, 0, 0,
                    0, 0, 0,

                    0, 0, 0,
                    1, 0, 0,
                    0, 0, 0;
        
        backProj[1] << 0, 1, 0,
                    0, 0, 0,
                    0, 0, 0,

                    0, 1, 0,
                    0, 0, 0,
                    0, 0, 0,

                    0, 1, 0,
                    0, 0, 0,
                    0, 0, 0;
        
        backProj[2] << 1, 0, 0,
                    0, 0, 0,
                    0, 0, 0,

                    1, 0, 0,
                    0, 0, 0,
                    0, 0, 0,

                    1, 0, 0,
                    0, 0, 0,
                    0, 0, 0;

        al[0] = "left border";
        al[1] = "right border";
        al[2] = "top border";
        al[3] = "bottom border";
        al[4] = "top right edge";
        al[5] = "bottom left edge";
        
        /**
         * Slightly different bahaviour: all rays running along a border are ignored
         * 
         */
        // for (index_t i=0; i<numCases/2; i++) {
        //     WHEN("A z-axis-aligned ray runs along the " + al[i] + " of the volume") {
        //         // x-ray source must be very far from the volume center to make testing of the op backprojection simpler
        //         geom.emplace_back(volSize*2000,volSize,volumeDescriptor,sinoDescriptor,0.0,0.0,0.0,0.0,0.0,offsetx[i],offsety[i]);
        //         SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);
        //         THEN("The result of projecting through a voxel is exactly the voxel's value") {       
        //             for (index_t j=0; j<volSize;j++) {
        //                 volume = 0;
        //                 switch (i)
        //                 {
        //                 case 0:
        //                     volume(0,volSize/2,j) = 1;
        //                     break;
        //                 case 1:
        //                     volume(volSize/2,0,j) = 1;
        //                     break;
        //                     break;
        //                 case 2:
        //                     volume(0,0,j) = 1;
        //                     break;
        //                 default:
        //                     break;
        //                 }
                        
        //                 op.apply(volume,sino);
        //                 REQUIRE(sino[0]==1);
        //             }

        //             AND_THEN("The backprojection yields the exact adjoints") {
        //                 sino[0]=1;
        //                 op.applyAdjoint(sino,volume);

        //                 REQUIRE(volume.getData().isApprox(backProj[i]));
        //             }
        //         }
        //     }
        // }

        for (index_t i=0; i<numCases; i++) {
            WHEN("A z-axis-aligned ray runs along the " + al[i] + " of the volume") {
                // x-ray source must be very far from the volume center to make testing of the op backprojection simpler
                geom.emplace_back(volSize*2000,volSize,volumeDescriptor,sinoDescriptor,0.0,0.0,0.0,0.0,0.0,-offsetx[i],-offsety[i]);
                SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);
                THEN("The result of projecting is zero") {       
                    volume = 1;
                    
                    op.apply(volume,sino);
                    REQUIRE(sino[0]==0);

                    AND_THEN("The result of backprojection is also zero") {
                    sino[0]=1;
                    op.applyAdjoint(sino,volume);

                    DataContainer zero(volumeDescriptor); REQUIRE(volume==zero);
                    }
                }
            }
        }
        

    }

    GIVEN("A 2D setting with multiple projection angles") {
        IndexVector_t volumeDims(2), sinoDims(2);
        const index_t volSize = 5;
        const index_t detectorSize = 1;
        const index_t numImgs = 4;
        volumeDims<<volSize,volSize;
        sinoDims<<detectorSize,numImgs;
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);
        std::vector<Geometry> geom;

        WHEN("Both x- and y-axis-aligned rays are present") {
            geom.emplace_back(20*volSize, volSize, 0, volumeDescriptor, sinoDescriptor);
            geom.emplace_back(20*volSize, volSize, 90  * M_PI / 180., volumeDescriptor, sinoDescriptor);
            geom.emplace_back(20*volSize, volSize, 180 * M_PI / 180., volumeDescriptor, sinoDescriptor);
            geom.emplace_back(20*volSize, volSize, 270 * M_PI / 180., volumeDescriptor, sinoDescriptor);

            SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);

            THEN("Values are accumulated correctly along each ray's path") {
                volume = 0;

                //set only values along the rays' path to one to make sure interpolation is dones correctly
                for (index_t i=0;i<volSize;i++) {
                    volume(i,volSize/2) = 1;
                    volume(volSize/2,i) = 1;
                }
                
                op.apply(volume,sino);
                for (index_t i=0; i<numImgs; i++)
                    REQUIRE(sino[i]==Approx(5.0));
                
                AND_THEN("Backprojection yields the exact adjoint") {
                    RealVector_t cmp(volSize*volSize);

                    cmp << 0, 0, 10, 0, 0,
                           0, 0, 10, 0, 0,
                           10, 10, 20, 10, 10,
                           0, 0, 10, 0, 0,
                           0, 0, 10, 0, 0;

                    op.applyAdjoint(sino,volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor,cmp)));
                }
            }
        }
    }

    GIVEN("A 3D setting with multiple projection angles") {
        IndexVector_t volumeDims(3), sinoDims(3);
        const index_t volSize = 3;
        const index_t detectorSize = 1;
        const index_t numImgs = 6;
        volumeDims<<volSize,volSize,volSize;
        sinoDims<<detectorSize,detectorSize,numImgs;
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);
        std::vector<Geometry> geom;

        WHEN("x-, y and z-axis-aligned rays are present") {
            real_t beta[numImgs] = {0.0, 0.0, 0.0,0.0,M_PI_2,3*M_PI_2};
            real_t gamma[numImgs] = {0.0, M_PI, M_PI_2, 3*M_PI_2,M_PI_2,3*M_PI_2};

            for (index_t i = 0;i<numImgs;i++)
                geom.emplace_back(volSize*20,volSize,volumeDescriptor,sinoDescriptor,gamma[i],beta[i]);

            SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);

            THEN("Values are accumulated correctly along each ray's path") {
                volume = 0;

                //set only values along the rays' path to one to make sure interpolation is dones correctly
                for (index_t i=0;i<volSize;i++) {
                    volume(i,volSize/2,volSize/2) = 1;
                    volume(volSize/2,i,volSize/2) = 1;
                    volume(volSize/2,volSize/2,i) = 1;
                }
                
                op.apply(volume,sino);
                for (index_t i=0; i<numImgs; i++)
                    REQUIRE(sino[i]==Approx(3.0));
                
                AND_THEN("Backprojection yields the exact adjoint") {
                    RealVector_t cmp(volSize*volSize*volSize);

                    cmp << 0, 0, 0,
                           0, 6, 0,
                           0, 0, 0,

                           0, 6, 0,
                           6, 18, 6,
                           0, 6, 0,

                           0, 0, 0,
                           0, 6, 0,
                           0, 0, 0;

                    op.applyAdjoint(sino,volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor,cmp)));
                }
            }
        }
    }
}

SCENARIO("Projection under an angle") {
    GIVEN("A 2D setting with a single ray") {
        IndexVector_t volumeDims(2), sinoDims(2);
        const index_t volSize = 4;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims<<volSize,volSize;
        sinoDims<<detectorSize,numImgs;
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);
        std::vector<Geometry> geom;

        WHEN("Projecting under an angle of 30 degrees and ray goes through center of volume") {
            // In this case the ray enters and exits the volume through the borders along the main direction
            geom.emplace_back(volSize*20,volSize,-pi/6,volumeDescriptor,sinoDescriptor);
            SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);

            THEN("Ray intersects the correct pixels") {
                volume = 1;
                volume(3,0) = 0;
                volume(2,0) = 0;
                volume(2,1) = 0;

                volume(1,2) = 0;
                volume(1,3) = 0;
                volume(0,3) = 0;
                // volume(2,2 also hit due to numerical errors)
                volume(2,2) = 0;                

                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor);
                REQUIRE(sino==zero);

                AND_THEN("The correct weighting is applied") {
                    volume(3,0) = 1;
                    volume(2,0) = 2;
                    volume(2,1) = 3;

                    op.apply(volume,sino);
                    REQUIRE( sino[0] == Approx( 2*sqrt(3) + 2 ) ); 

                    //on the other side of the center
                    volume = 0;
                    volume(1,2) = 3;
                    volume(1,3) = 2;
                    volume(0,3) = 1;

                    op.apply(volume,sino);
                    REQUIRE( sino[0] == Approx(2*sqrt(3) + 2) );
                    
                    sino[0] = 1;

                    RealVector_t expected(volSize*volSize);
                    expected << 0, 0, 2-2/sqrt(3), 4/sqrt(3)-2,
                                    0, 0, 2/sqrt(3), 0,
                                    0, 2/sqrt(3), 0, 0,
                                    4/sqrt(3)-2, 2-2/sqrt(3), 0, 0;

                    op.applyAdjoint(sino,volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor,expected)));                        
                }
                
            }
            
        }

        WHEN("Projecting under an angle of 30 degrees and ray enters through the right border") {
            // In this case the ray exits through a border along the main ray direction, but enters through a border
            // not along the main direction
            geom.emplace_back(volSize*20,volSize,-M_PI/6,volumeDescriptor,sinoDescriptor,0.0,sqrt(3));
            SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);

            THEN("Ray intersects the correct pixels") {
                volume = 1;
                volume(3,1) = 0;
                volume(3,2) = 0;
                volume(3,3) = 0;
                volume(2,3) = 0;
                
                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor); REQUIRE(sino==zero);

                AND_THEN("The correct weighting is applied") {
                    volume(3,1) = 4;
                    volume(3,2) = 3;
                    volume(3,3) = 2;
                    volume(2,3) = 1;

                    op.apply(volume,sino);
                    REQUIRE( sino[0] == Approx(14-4*sqrt(3)) ); 
                    
                    sino[0] = 1;

                    RealVector_t expected(volSize*volSize);
                    expected << 0, 0, 0, 0,
                                    0, 0, 0, 4-2*sqrt(3),
                                    0, 0, 0, 2/sqrt(3),
                                    0, 0, 2-2/sqrt(3), 4/sqrt(3)-2;

                    op.applyAdjoint(sino,volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor,expected)));                       
                }
                
            }
            
        }

        WHEN("Projecting under an angle of 30 degrees and ray exits through the left border") {
            // In this case the ray enters through a border along the main ray direction, but exits through a border
            // not along the main direction
            geom.emplace_back(volSize*20,volSize,-M_PI/6,volumeDescriptor,sinoDescriptor,0.0,-sqrt(3));
            SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);

            THEN("Ray intersects the correct pixels") {
                volume = 1;
                volume(0,0) = 0;
                volume(1,0) = 0;
                volume(0,1) = 0;
                volume(0,2) = 0;
                
                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor); REQUIRE(sino==zero);

                AND_THEN("The correct weighting is applied") {
                    volume(1,0) = 1;
                    volume(0,0) = 2;
                    volume(0,1) = 3;
                    volume(0,2) = 4;

                    op.apply(volume,sino);
                    REQUIRE( sino[0] == Approx( 14-4*sqrt(3) )); 
                    
                    sino[0] = 1;

                    RealVector_t expected(volSize*volSize);
                    expected << 4/sqrt(3)-2, 2-2/sqrt(3), 0, 0,
                                    2/sqrt(3) , 0, 0, 0,
                                    4-2*sqrt(3), 0, 0, 0,
                                    0, 0, 0, 0;

                    op.applyAdjoint(sino,volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor,expected)));                       
                }
                
            }
            
        }

        WHEN("Projecting under an angle of 30 degrees and ray only intersects a single pixel") {
            geom.emplace_back(volSize*20,volSize,-M_PI/6,volumeDescriptor,sinoDescriptor,0.0,-2-sqrt(3)/2);
            SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);

            THEN("Ray intersects the correct pixels") {
                volume = 1;
                volume(0,0) = 0;
                
                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor); REQUIRE(sino==zero);

                AND_THEN("The correct weighting is applied") {
                    volume(0,0) = 1;

                    op.apply(volume,sino);
                    REQUIRE( sino[0] == Approx( 1/sqrt(3) ) ); 
                    
                    sino[0] = 1;

                    RealVector_t expected(volSize*volSize);
                    expected << 1/sqrt(3), 0, 0, 0,
                                    0, 0, 0, 0,
                                    0, 0, 0, 0,
                                    0, 0, 0, 0;

                    op.applyAdjoint(sino,volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor,expected)));
                }
                
            }
            
        }

        WHEN("Projecting under an angle of 120 degrees and ray goes through center of volume") {
            // In this case the ray enters and exits the volume through the borders along the main direction
            geom.emplace_back(volSize*20,volSize,-2*M_PI/3,volumeDescriptor,sinoDescriptor);
            SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);

            THEN("Ray intersects the correct pixels") {
                volume = 1;
                volume(0,0) = 0;
                volume(0,1) = 0;
                volume(1,1) = 0;
                volume(2,2) = 0;
                volume(3,2) = 0;
                volume(3,3) = 0;
                // volume(1,2) hit due to numerical error
                volume(1,2) = 0;

                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor); REQUIRE(sino==zero);

                AND_THEN("The correct weighting is applied") {
                    volume(0,0) = 1;
                    volume(0,1) = 2;
                    volume(1,1) = 3;

                    op.apply(volume,sino);
                    REQUIRE( sino[0] == Approx( 2*sqrt(3) + 2 ) ); 

                    //on the other side of the center
                    volume = 0;
                    volume(2,2) = 3;
                    volume(3,2) = 2;
                    volume(3,3) = 1;

                    op.apply(volume,sino);
                    REQUIRE( sino[0] == Approx(2*sqrt(3) + 2) );
                    
                    sino[0] = 1;

                    RealVector_t expected(volSize*volSize);

                    expected << 4/sqrt(3)-2, 0, 0, 0,
                                2-2/sqrt(3), 2/sqrt(3),0, 0,
                                0, 0, 2/sqrt(3), 2-2/sqrt(3),
                                0, 0, 0, 4/sqrt(3)-2;

                    op.applyAdjoint(sino,volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor,expected)));
                }
                
            }
            
        }
        
        WHEN("Projecting under an angle of 120 degrees and ray enters through the top border") {
            // In this case the ray exits through a border along the main ray direction, but enters through a border
            // not along the main direction
            geom.emplace_back(volSize*20,volSize,-2*M_PI/3,volumeDescriptor,sinoDescriptor,0.0,0.0,sqrt(3));
            SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);

            THEN("Ray intersects the correct pixels") {
                volume = 1;
                volume(0,2) = 0;
                volume(0,3) = 0;
                volume(1,3) = 0;
                volume(2,3) = 0;
                
                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor); REQUIRE(sino==zero);

                AND_THEN("The correct weighting is applied") {
                    volume(0,2) = 1;
                    volume(0,3) = 2;
                    volume(1,3) = 3;
                    volume(2,3) = 4;

                    op.apply(volume,sino);
                    REQUIRE( sino[0] == Approx( 14-4*sqrt(3) ) ) ; 
                    
                    sino[0] = 1;

                    RealVector_t expected(volSize*volSize);
                    
                    expected << 0, 0, 0, 0,
                                0, 0, 0, 0,
                                2-2/sqrt(3),0, 0, 0,
                                4/sqrt(3)-2, 2/sqrt(3), 4-2*sqrt(3), 0;

                    op.applyAdjoint(sino,volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor,expected)));
                }
                
            }
            
        }

        WHEN("Projecting under an angle of 120 degrees and ray exits through the bottom border") {
            // In this case the ray enters through a border along the main ray direction, but exits through a border
            // not along the main direction
            geom.emplace_back(volSize*20,volSize,-2*M_PI/3,volumeDescriptor,sinoDescriptor,0.0,0.0,-sqrt(3));
            SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);

            THEN("Ray intersects the correct pixels") {
                volume = 1;
                volume(1,0) = 0;
                volume(2,0) = 0;
                volume(3,0) = 0;
                volume(3,1) = 0;
                
                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor); REQUIRE(sino==zero);

                AND_THEN("The correct weighting is applied") {
                    volume(1,0) = 4;
                    volume(2,0) = 3;
                    volume(3,0) = 2;
                    volume(3,1) = 1;

                    op.apply(volume,sino);
                    REQUIRE( sino[0] == Approx( 14-4*sqrt(3) ) ); 
                    
                    sino[0] = 1;

                    RealVector_t expected(volSize*volSize);
                    
                    expected << 0,4-2*sqrt(3), 2/sqrt(3), 4/sqrt(3)-2,
                                0, 0, 0, 2-2/sqrt(3),
                                0, 0, 0, 0,
                                0, 0, 0, 0;

                    op.applyAdjoint(sino,volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor,expected)));
                }
                
            }
            
        }

        WHEN("Projecting under an angle of 120 degrees and ray only intersects a single pixel") {
            // This is a special case that is handled separately in both forward and backprojection
            geom.emplace_back(volSize*20,volSize,-2*M_PI/3,volumeDescriptor,sinoDescriptor,0.0,0.0,-2-sqrt(3)/2);
            SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);

            THEN("Ray intersects the correct pixels") {
                volume = 1;
                volume(3,0) = 0;
                
                op.apply(volume, sino);
                DataContainer zero(sinoDescriptor); REQUIRE(sino==zero);

                AND_THEN("The correct weighting is applied") {
                    volume(3,0) = 1;

                    op.apply(volume,sino);
                    REQUIRE( sino[0] == Approx( 1/sqrt(3) ).epsilon(0.005) ); 
                    
                    sino[0] = 1;

                    RealVector_t expected(volSize*volSize);
                    expected << 0, 0, 0, 1/sqrt(3),
                                    0, 0, 0, 0,
                                    0, 0, 0, 0,
                                    0, 0, 0, 0;

                    op.applyAdjoint(sino,volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor,expected)));
                }
                
            }
            
        }
        
    }
    
    GIVEN("A 3D setting with a single ray") {
        IndexVector_t volumeDims(3), sinoDims(3);
        const index_t volSize = 3;
        const index_t detectorSize = 1;
        const index_t numImgs = 1;
        volumeDims<<volSize,volSize,volSize;
        sinoDims<<detectorSize,detectorSize,numImgs;
        DataDescriptor volumeDescriptor(volumeDims);
        DataDescriptor sinoDescriptor(sinoDims);
        DataContainer volume(volumeDescriptor);
        DataContainer sino(sinoDescriptor);
        std::vector<Geometry> geom;

        Eigen::Matrix<real_t,volSize*volSize*volSize,1> backProj; 


        WHEN("A ray with an angle of 30 degrees goes through the center of the volume") {
            // In this case the ray enters and exits the volume along the main direction
            geom.emplace_back(volSize*20,volSize,volumeDescriptor,sinoDescriptor,M_PI/6);
            SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);
            
            THEN("The ray intersects the correct voxels") {       
                volume = 1;
                volume(1,1,1) = 0;
                volume(2,1,0) = 0;
                volume(1,1,0) = 0;
                volume(0,1,2) = 0;
                volume(1,1,2) = 0;

                op.apply(volume,sino);
                REQUIRE(sino[0]==Approx(0).margin(1e-5));

                AND_THEN("The correct weighting is applied") {
                    volume(1,1,1) = 1;
                    volume(2,1,0) = 3;
                    volume(1,1,2) = 2;

                    op.apply(volume,sino);
                    REQUIRE( sino[0]==Approx( 3*sqrt(3) - 1 ) );

                    sino[0] = 1;
                    backProj<< 0, 0, 0,
                               0, 1-1/sqrt(3),sqrt(3)-1,
                               0, 0, 0,

                               0, 0, 0,
                               0, 2/sqrt(3), 0,
                               0, 0, 0,

                               0, 0, 0,
                               sqrt(3)-1,1-1/sqrt(3), 0,
                               0, 0, 0;
                    
                    op.applyAdjoint(sino,volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor,backProj)));
                }
            }
        }

        WHEN("A ray with an angle of 30 degrees enters through the right border") {
            // In this case the ray enters through a border orthogonal to a non-main direction
            geom.emplace_back(volSize*20,volSize,volumeDescriptor,sinoDescriptor,M_PI/6,0.0,0.0,0.0,0.0,1);
            SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);
            
            THEN("The ray intersects the correct voxels") {       
                volume = 1;
                volume(2,1,1) = 0;
                volume(2,1,0) = 0;
                volume(2,1,2) = 0;
                volume(1,1,2) = 0;

                op.apply(volume,sino);
                REQUIRE(sino[0]==Approx(0).margin(1e-5));

                AND_THEN("The correct weighting is applied") {
                    volume(2,1,0) = 4;
                    volume(1,1,2) = 3;
                    volume(2,1,1) = 1;

                    op.apply(volume,sino);
                    REQUIRE( sino[0]==Approx( 1-2/sqrt(3) + 3*sqrt(3) ) );

                    sino[0] = 1;
                    backProj<< 0, 0, 0,
                               0, 0, 1-1/sqrt(3),
                               0, 0, 0,

                               0, 0, 0,
                               0, 0, 2/sqrt(3),
                               0, 0, 0,

                               0, 0, 0,
                               0, sqrt(3)-1,1-1/sqrt(3),
                               0, 0, 0;
                    
                    op.applyAdjoint(sino,volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor,backProj)));
                }
            }
        }

        WHEN("A ray with an angle of 30 degrees exits through the left border") {
            // In this case the ray exit through a border orthogonal to a non-main direction
            geom.emplace_back(volSize*20,volSize,volumeDescriptor,sinoDescriptor,M_PI/6,0.0,0.0,0.0,0.0,-1);
            SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);
            
            THEN("The ray intersects the correct voxels") {       
                volume = 1;
                volume(0,1,0) = 0;
                volume(1,1,0) = 0;
                volume(0,1,1) = 0;
                volume(0,1,2) = 0;

                op.apply(volume,sino);
                REQUIRE(sino[0]==Approx(0).margin(1e-5));

                AND_THEN("The correct weighting is applied") {
                    volume(0,1,2) = 4;
                    volume(1,1,0) = 3;
                    volume(0,1,1) = 1;

                    op.apply(volume,sino);
                    REQUIRE( sino[0]==Approx( 3*sqrt(3)+ 1 - 2/sqrt(3) ) );

                    sino[0] = 1;
                    backProj<< 0, 0, 0,
                               1-1/sqrt(3), sqrt(3)-1, 0,
                               0, 0, 0,

                               0, 0, 0,
                               2/sqrt(3), 0, 0,
                               0, 0, 0,

                               0, 0, 0,
                               1-1/sqrt(3), 0, 0,
                               0, 0, 0;
                    
                    op.applyAdjoint(sino,volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor,backProj)));
                }
            }
        }

        WHEN("A ray with an angle of 30 degrees only intersects a single voxel") {
            // special case - no interior voxels, entry and exit voxels are the same
            geom.emplace_back(volSize*20,volSize,volumeDescriptor,sinoDescriptor,M_PI/6,0.0,0.0,0.0,0.0,-2);
            SiddonsMethod op(volumeDescriptor,sinoDescriptor,geom);
            
            THEN("The ray intersects the correct voxels") {       
                volume = 1;
                volume(0,1,0) = 0;

                op.apply(volume,sino);
                REQUIRE(sino[0]==Approx(0).margin(1e-5));

                AND_THEN("The correct weighting is applied") {
                    volume(0,1,0) = 1;

                    op.apply(volume,sino);
                    REQUIRE( sino[0]==Approx( sqrt(3)-1) );

                    sino[0] = 1;
                    backProj<< 0, 0, 0,
                               sqrt(3)-1, 0, 0,
                               0, 0, 0,

                               0, 0, 0,
                               0, 0, 0,
                               0, 0, 0,

                               0, 0, 0,
                               0, 0, 0,
                               0, 0, 0;
                    
                    op.applyAdjoint(sino,volume);
                    REQUIRE(isApprox(volume, DataContainer(volumeDescriptor,backProj)));
                }
            }
        }
    }

}