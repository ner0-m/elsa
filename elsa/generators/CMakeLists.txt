# list all the headers of the module
set(MODULE_HEADERS
    EllipseGenerator.h
    Ellipsoid.h
    EllipCylinder.h
    EllipCylinderFree.h
    CylinderFree.h
    Cylinder.h
    Sphere.h
    Box.h
    PhantomDefines.h
    Phantoms.h
    ForbildPhantom.h
    ForbildData.h
    BaseCircleTrajectoryGenerator.h
    CircleTrajectoryGenerator.h
    CurvedCircleTrajectoryGenerator.h
    LimitedAngleTrajectoryGenerator.h
    NoiseGenerators.h
    SphereTrajectoryGenerator.h
    TrajectoryGenerator.h
    analytical/Image.h
)

# list all the code files of the module
set(MODULE_SOURCES
    EllipseGenerator.cpp
    Ellipsoid.cpp
    EllipCylinder.cpp
    EllipCylinderFree.cpp
    CylinderFree.cpp
    Cylinder.cpp
    Sphere.cpp
    Box.cpp
    PhantomDefines.cpp
    Phantoms.cpp
    ForbildPhantom.cpp
    BaseCircleTrajectoryGenerator.cpp
    CircleTrajectoryGenerator.cpp
    CurvedCircleTrajectoryGenerator.cpp
    LimitedAngleTrajectoryGenerator.cpp
    NoiseGenerators.cpp
    SphereTrajectoryGenerator.cpp
    TrajectoryGenerator.cpp
)

list(APPEND MODULE_PUBLIC_DEPS elsa_core elsa_logging elsa_projectors)
list(APPEND MODULE_PRIVATE_DEPS)

ADD_ELSA_MODULE(
    generators "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

write_module_config(${ELSA_MODULE_NAME} DEPENDENCIES elsa_core elsa_logging elsa_projectors)
