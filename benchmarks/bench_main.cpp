#define CATCH_CONFIG_RUNNER
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>

int main(int argc, char* argv[])
{
    Catch::Session session; // There must be exactly one instance

    // writing to session.configData() here sets defaults
    // this is the preferred way to set them

    // Get binary name
    std::string executable_path = argv[0];

    // Find name or executable (without filesystem as GCC 7.4 doesn't support it)
    // find last of / or \ (depending on linux or windows systems)
    auto found = executable_path.find_last_of("/\\");
    std::string filename = executable_path.substr(found + 1);

    // set reporter and filename to match binary
    session.configData().reporterName = "console";
    // session.configData().outputFilename = filename + ".xml";

    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) // Indicates a command line error
        return returnCode;

    // writing to session.configData() or session.Config() here
    // overrides command line args
    // only do this if you know you need to

    int numFailed = session.run();

    // numFailed is clamped to 255 as some unices only use the lower 8 bits.
    // This clamping has already been applied, so just return it here
    // You can also do any post run clean-up here
    return numFailed;
}
