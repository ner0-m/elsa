#pragma once

#include <vector>
#include <set>
#include <string>
#include <map>
#include <memory>

namespace elsa
{
    struct Module {
        struct Function {
            struct Parameter {
                std::string name;
                std::string type;
                std::string defaultValue;

                Parameter(std::string name, std::string type)
                    : name{name}, type{type}, defaultValue{""}
                {
                }

                Parameter(std::string name, std::string type, std::string defaultValue)
                    : name{name}, type{type}, defaultValue{defaultValue}
                {
                }
            };

            std::string name;
            std::string returnType;
            std::vector<Parameter> params;
            std::string refQualifier;
            std::size_t numDefaultArgs{0};

            // Some default arguments may not represent an rvalue or have side effects, and,
            // therefore, may not be (at least not easily) bindable using the standard method for
            // binding of default args. Upon encountering such an argument, it and all following
            // arguments should be converted to overloads instead. Set to the number of
            // arguments if all are bindable.
            std::size_t posFirstNonBindableDefaultArg{0};
            bool isStatic;
            bool isConstructor;
            bool isConst;
            bool isTemplate;

            constexpr static auto RQ_NONE = "";
            constexpr static auto RQ_LVAL = "&";
            constexpr static auto RQ_RVAL = "&&";
        };

        struct UserDefinedTag {
            std::string name;
            std::string namespaceStrippedName;
            std::string alias;
        };

        // represents a class or struct
        struct Record : public UserDefinedTag {
            std::vector<std::string> bases;
            std::map<std::string, Function> methods;
            bool isAbstract;
        };

        struct Enum : public UserDefinedTag {
            /// maps enumerator name to value
            /// the value is stored as a string, as an enum type may be as large as an unsigned long
            /// long or a (signed) long long, and the signedness must be determined on a per-enum
            /// basis
            std::map<std::string, std::string> values;
            bool isScoped;
        };

        struct ClassHints {
            std::string recordName;
            std::string classHintsName;
            std::set<std::string> ignoredMethods;
            bool definesCustomMethods{false};
            bool exposesBufferInfo{false};
        };

        std::string name;
        std::string path;
        std::string pythonName;
        std::vector<std::unique_ptr<Enum>> enums;
        std::vector<std::unique_ptr<Record>> records;
        std::set<std::string> includes;
        std::set<std::string> pybindIncludes;
        bool noPythonModule;

        struct ModuleHints {
            // path to hints file, empty if none is specified
            std::string includePath;
            std::string moduleHintsName;
            bool definesGlobalCustomFunctions{false};
        };

        ModuleHints moduleHints;
        std::map<std::string, ClassHints> classHints;
    };
} // namespace elsa