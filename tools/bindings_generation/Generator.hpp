#pragma once

#include "Module.h"

#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>
#include <assert.h>

// Determine wether to use <filesystem> or <experimental/filesystem>. Adapted from
// https://stackoverflow.com/questions/53365538/how-to-determine-whether-to-use-filesystem-or-experimental-filesystem,
// simplified by removing MSVC specific code, as this is expected to be run with clang

// We haven't checked which filesystem to include yet
#ifndef INCLUDE_STD_FILESYSTEM_EXPERIMENTAL

// Check for feature test macro for <filesystem>
#if defined(__cpp_lib_filesystem)
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 0

// Check for feature test macro for <experimental/filesystem>
#elif defined(__cpp_lib_experimental_filesystem)
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1

// We can't check if headers exist...
// Let's assume experimental to be safe
#elif !defined(__has_include)
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1

// Check if the header "<filesystem>" exists
#elif __has_include(<filesystem>)
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 0

// Check if the header "<filesystem>" exists
#elif __has_include(<experimental/filesystem>)
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1

// Fail if neither header is available with a nice error message
#else
#error Could not find system header "<filesystem>" or "<experimental/filesystem>"
#endif

// We priously determined that we need the exprimental version
#if INCLUDE_STD_FILESYSTEM_EXPERIMENTAL
// Include it
#include <experimental/filesystem>

// We need the alias from std::experimental::filesystem to std::filesystem
namespace std
{
    namespace filesystem = experimental::filesystem;
}

// We have a decent compiler and can use the normal version
#else
// Include it
#include <filesystem>
#endif

#endif // #ifndef INCLUDE_STD_FILESYSTEM_EXPERIMENTAL

class Generator
{
public:
    static void generateBindingsForModule(const elsa::Module& m, std::string outputPath)
    {
        if (outputPath.empty())
            outputPath = "bind_" + m.name + ".cpp";

        std::filesystem::path p(outputPath);
        std::ofstream outputFile(outputPath);

        outputFile << "#include <pybind11/pybind11.h>\n";
        for (const auto& include : m.pybindIncludes)
            outputFile << "#include <" << include << ">\n";

        outputFile << "\n";
        for (const auto& include : m.includes)
            outputFile << "#include \"" << include.substr(m.path.size() + 1) << "\"\n";

        if (!m.moduleHints.includePath.empty())
            outputFile << "\n#include \""
                       << m.moduleHints.includePath.substr(m.moduleHints.includePath.rfind("/") + 1)
                       << "\"\n";

        outputFile << "\n"
                   << "namespace py = pybind11;\n\n"
                   << "void add_definitions_" << m.pythonName << "(py::module& m)\n"
                   << "{\n";

        for (const auto& e : m.enums) {
            generateBindingsForEnum(*e, outputFile);
        }

        for (const auto& r : m.records) {
            if (m.classHints.find(r->name) != m.classHints.end()) {
                const auto& hints = m.classHints.at(r->name);
                generateBindingsForRecord(*r, outputFile, &hints);
            } else {
                generateBindingsForRecord(*r, outputFile);
            }
        }

        if (m.moduleHints.definesGlobalCustomFunctions)
            outputFile << "\t" << m.moduleHints.moduleHintsName << "::addCustomFunctions(m);\n";

        outputFile << "}\n";

        if (!m.noPythonModule) {
            outputFile << "\n"
                       << "PYBIND11_MODULE(" << m.pythonName << ", m)\n"
                       << "{\n"
                       << "\tadd_definitions_" << m.pythonName << "(m);\n"
                       << "}";
        }
    }

    static void generateBindingsForRecord(const elsa::Module::Record& r, std::ofstream& outputFile,
                                          const elsa::Module::ClassHints* hints = nullptr)
    {
        std::string qualifiedName = r.name;

        std::string additionalProps = "";

        if (classSupportsBufferProtocol(r, hints))
            additionalProps = ", py::buffer_protocol()";

        auto pythonName = getPythonNameForTag(r.namespaceStrippedName);
        // py::class_<Class, Base1, Base2,...>(m, Class[, py::buffer_protocol()])
        outputFile << "\tpy::class_<" << r.name;
        for (auto& base : r.bases)
            outputFile << ", " << base;
        outputFile << "> " << pythonName << "(m, \"" << pythonName << "\"" << additionalProps
                   << ");\n\t" << pythonName;

        // define constructors and other methods
        for (const auto& [fId, f] : r.methods) {
            // skip function templates
            if (f.isTemplate)
                continue;

            // skip functions with an rvalue ref-qualifier
            if (f.refQualifier == elsa::Module::Function::RQ_RVAL)
                continue;

            // methods accepting rvalue references or unique_ptrs as parameters cannot be bound
            // TODO: add warning for the case when a non-abstract class is left without any
            // viable constructors
            bool viable = true;
            for (const auto& param : f.params) {
                if (param.type.rfind("&&") == param.type.length() - 2
                    || param.type.find("std::unique_ptr<") != std::string::npos
                    || param.type.find("std::__1::unique_ptr<") != std::string::npos) {
                    viable = false;
                    break;
                }
            }
            if (!viable)
                continue;

            // convert default parameters to overloads
            for (std::size_t i = 0; i <= f.params.size() - f.posFirstNonBindableDefaultArg; i++) {

                std::size_t numSelectedArgs = f.posFirstNonBindableDefaultArg + i;
                std::string types = makeParamTypesString(f, numSelectedArgs);
                std::string namesAndDefaults = makeParamNamesString(f, numSelectedArgs);

                if (f.isConstructor) {
                    // .def(py::init<ArgType1, ArgType2,...>(), py::arg("argName1") =
                    // defaultValue1,...)
                    if (!r.isAbstract)
                        outputFile << "\n\t\t.def(py::init<" << types << ">()" << namesAndDefaults
                                   << ")";
                } else {
                    // .def[_static]("methodName", [(methodType)&methodAddress[,
                    // methodParameters]])
                    std::string def = "";
                    if (methodExposesBufferInfo(r, f)) {
                        def = "\n\t\t.def_buffer(";
                        outputFile << def << "&" << qualifiedName << "::" << f.name << ")";
                    } else {
                        if (f.isStatic) {
                            def = "\n\t\t.def_static(";
                        } else {
                            def = "\n\t\t.def(";
                        }

                        additionalProps = getReturnValuePolicyAsString(f);

                        def += "\"" + getPythonNameForMethod(f.name) + "\", ";

                        outputFile << def << "(" << f.returnType << "("
                                   << (f.isStatic ? "" : qualifiedName + "::") << "*)"
                                   << "(" << types << ")" << (f.isConst ? " const" : "")
                                   << f.refQualifier << ")("
                                   << "&" << qualifiedName << "::" << f.name << ")"
                                   << namesAndDefaults << additionalProps << ")";
                    }
                }
            }
        }
        outputFile << ";\n";

        // add custom methods
        if (hints) {

            if (hints->definesCustomMethods) {
                outputFile << "\n\t" << hints->classHintsName << "::"
                           << "addCustomMethods(" << pythonName << ");\n";
            }
            if (hints->exposesBufferInfo) {
                outputFile << "\n\t" << hints->classHintsName << "::"
                           << "exposeBufferInfo(" << pythonName << ");\n";
            }
        }

        outputFile << "\n";

        // add alias
        if (r.alias != "")
            outputFile << "\tm.attr(\"" << r.alias << "\") = m.attr(\""
                       << getPythonNameForTag(r.namespaceStrippedName) << "\");\n\n";
    }

    static void generateBindingsForEnum(const elsa::Module::Enum& e, std::ofstream& outputFile)
    {
        outputFile << "\tpy::enum_<" << e.name << ">(m, \""
                   << getPythonNameForTag(e.namespaceStrippedName) << "\")";
        for (const auto& p : e.values) {
            outputFile << "\n\t\t.value(\"" << p.first << "\", " << e.name << "::" << p.first
                       << ")";
        }
        if (!e.isScoped)
            outputFile << "\n\t\t.export_values()";

        outputFile << ";\n\n";
    }

private:
    static bool methodExposesBufferInfo(const elsa::Module::Record& r,
                                        const elsa::Module::Function& f)
    {
        return f.returnType == "pybind11::buffer_info" && f.params.size() == 1
               && f.params[0].type == r.name + "&";
    }

    static std::string makeParamTypesString(const elsa::Module::Function& f, std::size_t numParams)
    {
        std::string types = "";
        for (int i = 0; i < numParams; i++) {
            types += f.params[i].type;
            if (i != numParams - 1)
                types += ", ";
        }
        return types;
    }

    static std::string makeParamNamesString(const elsa::Module::Function& f, std::size_t numParams)
    {
        std::string namesAndDefaults = "";
        for (int i = 0; i < numParams; i++) {
            if (f.params[i].name != "") {
                namesAndDefaults += ", py::arg(\"" + f.params[i].name + "\")";

                if (f.posFirstNonBindableDefaultArg >= numParams && f.params[i].defaultValue != "")
                    namesAndDefaults += " = static_cast<" + f.params[i].type + ">("
                                        + f.params[i].defaultValue + ")";
            } else {
                break;
            }
        }

        return namesAndDefaults;
    }

    static std::string getPythonNameForMethod(const std::string& methodName)
    {
        static std::map<std::string, std::string> specialMethodsMap = {
            {"operator+", "__add__"},     {"operator–", "__sub__"},   {"operator*", "__mul__"},
            {"operator/", "__truediv__"}, {"operator%", "__mod__"},   {"operator<", "__lt__"},
            {"operator>", "__gt__"},      {"operator<=", "__le__"},   {"operator>=", "__ge__"},
            {"operator==", "__eq__"},     {"operator!=", "__ne__"},   {"operator-=", "__isub__"},
            {"operator+=", "__iadd__"},   {"operator*=", "__imul__"}, {"operator/=", "__idiv__"},
            {"operator%=", "__imod__"},   {"operator=", "set"},       {"operator[]", "__getitem__"},
        };

        auto it = specialMethodsMap.find(methodName);
        if (it != specialMethodsMap.end()) {
            return it->second;
        } else {
            return methodName;
        }
    }

    static std::string getPythonNameForTag(const std::string& unqualifiedId)
    {
        auto pythonName = unqualifiedId;

        // should be parsed in a specific order, so do NOT use std::map
        static std::vector<std::pair<std::string, std::string>> simplifiedTypes = {
            {"complex<float>", "cf"},
            {"complex<double>", "cd"},
            {"complex<long double>", "clongd"},
            {"<float>", "f"},
            {"<long double>", "longd"},
            {"<double>", "d"},
            {"<bool>", "b"},
            {"<unsigned short>", "us"},
            {"<short>", "s"},
            {"<unsigned int>", "ui"},
            {"<int>", "i"},
            {"<unsigned long long>", "ulongl"},
            {"<long long>", "longl"},
            {"<unsigned long>", "ul"},
            {"<long>", "l"},
            {"<", ""},
            {">", ""},
            {"::", ""},
            {", ", ""},
            {"-", "_"}};

        std::for_each(simplifiedTypes.begin(), simplifiedTypes.end(),
                      [&pythonName](std::pair<std::string, std::string> simple) {
                          for (auto pos = pythonName.find(simple.first); pos != std::string::npos;
                               pos = pythonName.find(simple.first)) {
                              pythonName.replace(pos, simple.first.length(), simple.second);
                          }
                      });

        return pythonName;
    }

    static std::string getReturnValuePolicyAsString(const elsa::Module::Function& f)
    {
        if (f.returnType.find("std::unique_ptr<") == 0
            || f.returnType.find("std::shared_ptr<") == 0) {
            // smart pointers and are handled automatically by pybind
            return "";
        } else if (f.returnType.rfind("&") == f.returnType.length() - 1
                   && f.returnType.rfind("&&") != f.returnType.length() - 2) {
            // when returning a lvalue reference, assume that it is a reference to a
            // member variable
            return ", py::return_value_policy::reference_internal";
        } else if (f.returnType.rfind("*") == f.returnType.length() - 1) {
            return ", py::return_value_policy::take_ownership";
        } else if (f.returnType.find("::") == std::string::npos) {
            // no special return value policy for builtin types that are not
            // referenced
            return "";
        } else {
            // otherwise, we have a custom type temporary or rvalue
            return ", py::return_value_policy::move";
        }
    }

    static bool classSupportsBufferProtocol(const elsa::Module::Record& r,
                                            const elsa::Module::ClassHints* hints = nullptr)
    {
        if (hints) {
            if (hints->exposesBufferInfo)
                return true;
        }

        for (const auto& [fId, f] : r.methods) {
            if (methodExposesBufferInfo(r, f))
                return true;
        }

        return false;
    }
};