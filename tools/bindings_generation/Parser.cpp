#include "Module.h"
#include "Generator.hpp"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/CommandLine.h"

#include <memory>
#include <set>
#include <sstream>

using namespace clang;
using namespace clang::tooling;

// Apply a custom category to all command-line options so that they are the only ones displayed.
static llvm::cl::OptionCategory bindingsOptions("Bindings generation options");
static llvm::cl::opt<std::string> hintsPath("hints", llvm::cl::cat(bindingsOptions),
                                            llvm::cl::desc("Path to hints file"));
static llvm::cl::opt<std::string>
    outputDir("o", llvm::cl::cat(bindingsOptions),
              llvm::cl::desc("Output directory, defaults to the current working directory."));
static llvm::cl::opt<std::string> moduleName(
    "name", llvm::cl::cat(bindingsOptions),
    llvm::cl::desc("Name of the module. This option or the pyname option must be specified."));
static llvm::cl::opt<std::string> pythonModuleName(
    "pyname", llvm::cl::cat(bindingsOptions),
    llvm::cl::desc("Name that should be given assigned to the generated python module. \"py\" "
                   "will be prepended to the value of the name option if not specified"));

static llvm::cl::extrahelp commonHelp(CommonOptionsParser::HelpMessage);

static elsa::Module m;

class ParserBase
{
public:
    // returns the lexically normal path similarly to std::filesystem::path::lexically_normal
    static std::string lexicallyNormalPathFallback(std::string path)
    {
        while (path.find("..") != std::string::npos) {
            auto posDots = path.find("..");

            std::size_t eraseEnd = posDots + 2;
            std::size_t numDots = 1;
            while (path.rfind("..", posDots) == posDots - 3) {
                posDots = path.rfind("..", posDots);
                numDots++;
            }

            std::size_t eraseStart = posDots - 1;
            for (int i = 0; i < numDots; i++)
                eraseStart = path.rfind("/", eraseStart - 1);

            path.erase(eraseStart, eraseEnd - eraseStart);
        }

        return path;
    }

protected:
    ParserBase(ASTContext* context) : context{context}, printingPolicy{context->getLangOpts()}
    {
        printingPolicy.PrintCanonicalTypes = 1;
    }

    std::string getNamespaceStrippedName(const TagDecl* declaration)
    {
        std::string namespaceStrippedName = declaration->getNameAsString();

        auto parent = declaration->getParent();
        while (parent->isRecord()) {
            auto rec = dyn_cast<CXXRecordDecl>(parent);
            namespaceStrippedName = getNamespaceStrippedName(rec) + "::" + namespaceStrippedName;
            parent = parent->getParent();
        }

        if (auto ctsd = dyn_cast<ClassTemplateSpecializationDecl>(declaration)) {
            namespaceStrippedName += "<";
            for (std::size_t i = 0; i < ctsd->getTemplateArgs().size(); i++) {

                const auto& tmpArg = ctsd->getTemplateArgs()[i];
                switch (tmpArg.getKind()) {
                    case TemplateArgument::ArgKind::Integral:
                        namespaceStrippedName += tmpArg.getAsIntegral().toString(10);
                        break;

                    case TemplateArgument::ArgKind::Type:
                        if (tmpArg.getAsType()->getAsCXXRecordDecl() != nullptr) {
                            namespaceStrippedName +=
                                getNamespaceStrippedName(tmpArg.getAsType()->getAsCXXRecordDecl());
                        } else {
                            namespaceStrippedName += tmpArg.getAsType().getAsString();
                        }
                        break;

                    default:
                        // TODO: add error message if another argkind is encountered
                        break;
                }

                if (i < ctsd->getTemplateArgs().size() - 1)
                    namespaceStrippedName += ", ";
            }
            namespaceStrippedName += ">";
        }

        return namespaceStrippedName;
    }

    std::string fullyQualifiedId(const QualType& qualType)
    {
        auto id = qualType.getAsString(printingPolicy);

        // remove spaces before references
        for (auto pos = id.find(" &"); pos != std::string::npos; pos = id.find(" &")) {
            id.erase(pos, 1);
        }

        // remove spaces before template closing brackets
        for (auto pos = id.find(" >"); pos != std::string::npos; pos = id.find(" >")) {
            id.erase(pos, 1);
        }

        return id;
    }

    std::string getHeaderLocation(CXXRecordDecl* declaration)
    {
        std::string location;
        auto ctsd = dyn_cast<ClassTemplateSpecializationDecl>(declaration);
        if (ctsd
            && (ctsd->getSpecializationKind() == TSK_ExplicitInstantiationDeclaration
                || ctsd->getSpecializationKind() == TSK_ExplicitInstantiationDefinition
                || ctsd->getSpecializationKind() == TSK_ImplicitInstantiation)) {
            // if this is a a template instantiation, include the ClassTemplateDecl instead
            ClassTemplateDecl* ctd = *ctsd->getInstantiatedFrom().getAddrOfPtr1();
            location = ctd->getBeginLoc().printToString(context->getSourceManager());
        } else {
            location = declaration->getBeginLoc().printToString(context->getSourceManager());
        }

        return location.substr(0, location.find(":"));
    }

    std::string getRefQualifierString(RefQualifierKind refQual)
    {
        switch (refQual) {
            case RQ_None:
                return elsa::Module::Function::RQ_NONE;
            case RQ_LValue:
                return elsa::Module::Function::RQ_LVAL;
            case RQ_RValue:
                return elsa::Module::Function::RQ_RVAL;
            default:
                assert(false && ("ref-qualifier kind not supported"));
                return elsa::Module::Function::RQ_NONE;
        }
    }

    // encountered tags' typenames are added to the map as soon as encountered, but the pointer to
    // the object remains a nullptr until the record is fully processed
    static std::map<std::string, elsa::Module::Record*> encounteredRecords;
    static std::map<std::string, elsa::Module::Enum*> encounteredEnums;
    ASTContext* context;
    PrintingPolicy printingPolicy;
};

std::map<std::string, elsa::Module::Record*> ParserBase::encounteredRecords = {};
std::map<std::string, elsa::Module::Enum*> ParserBase::encounteredEnums = {};

class ModuleParser : public RecursiveASTVisitor<ModuleParser>, public ParserBase
{
public:
    explicit ModuleParser(ASTContext* context) : ParserBase{context} {}

    bool shouldVisitTemplateInstantiations() { return true; };

    bool shouldVisitImplicitCode() { return true; }

    bool VisitCXXRecordDecl(CXXRecordDecl* declaration)
    {
        // ignore record declarations from includes
        if (!context->getSourceManager().isInMainFile(declaration->getLocation()))
            return true;

        // ignore lambdas (implicitly generated lambda classes may be visited if the
        // visitImplicitCode option is set)
        if (declaration->isLambda())
            return true;

        // ignore template declarations, only template instantiations carry the necessary
        // information for bindings generation
        if (declaration->isTemplated())
            return true;

        FullSourceLoc fullLocation = context->getFullLoc(declaration->getBeginLoc());
        if (!fullLocation.isValid())
            return true;

        if (declaration->isClass() || declaration->isStruct()) {

            m.includes.insert(getHeaderLocation(declaration));
            parseRecord(declaration);
        }
        return true;
    }

    bool VisitCXXMethodDecl(CXXMethodDecl* method)
    {
        // ignore method declarations from includes
        if (!context->getSourceManager().isInMainFile(method->getLocation()))
            return true;

        if (method->isTemplateInstantiation())
            appendMethodTemplateInstantiation(method);

        return true;
    }

    void parseDeclForQualType(QualType type)
    {
        type = type.getNonReferenceType();
        if (type->isRecordType()) {
            parseRecord(type->getAsCXXRecordDecl());
        } else if (type->isEnumeralType() || type->isScopedEnumeralType()) {
            parseEnum(dyn_cast<EnumDecl>(type->getAsTagDecl()));
        } else {
            return;
        }
    }

    void parseEnum(EnumDecl* declaration)
    {

        auto id = fullyQualifiedId(context->getEnumType(declaration));
        if (encounteredEnums.find(id) != encounteredEnums.end())
            return;

        auto e = std::make_unique<elsa::Module::Enum>();

        e->name = id;
        e->namespaceStrippedName = getNamespaceStrippedName(declaration);
        e->isScoped = declaration->isScoped();
        for (auto enumerator : declaration->enumerators()) {
            e->values.emplace(enumerator->getNameAsString(), enumerator->getInitVal().toString(10));
        }

        m.enums.push_back(std::move(e));
        encounteredEnums.emplace(m.enums.back()->name, m.enums.back().get());
    }

    void parseRecord(CXXRecordDecl* declaration)
    {
        if (!shouldBeRegisteredInCurrentModule(declaration)) {
            return;
        }

        auto r = std::make_unique<elsa::Module::Record>();
        auto id = fullyQualifiedId(context->getRecordType(declaration));

        // register tag as encountered, nullptr as second argument means it is still not fully
        // processed
        encounteredRecords.emplace(id, nullptr);
        r->name = id;
        r->namespaceStrippedName = getNamespaceStrippedName(declaration);

        if (declaration->hasDefinition()) {
            for (auto& base : declaration->bases()) {
                if (base.getAccessSpecifier() == AS_public) {
                    r->bases.push_back(fullyQualifiedId(base.getType()));
                    parseRecord(base.getType().getCanonicalType()->getAsCXXRecordDecl());
                }
            }
        }

        bool isAbstract = false;

        for (auto decl : declaration->decls()) {
            if (dyn_cast<UsingDecl>(decl)) {
                for (auto shadow : dyn_cast<UsingDecl>(decl)->shadows()) {
                    auto target = shadow->getTargetDecl();
                    // ignore non-public shadows as well as copy constructors
                    if (decl->getAccess() == AS_public
                        && (dyn_cast<CXXConstructorDecl>(target) == nullptr
                            || !dyn_cast<CXXConstructorDecl>(target)->isCopyConstructor())) {
                        handleMemberDeclaration(target, *r);
                    }
                }
            } else {
                handleMemberDeclaration(decl, *r);
            }
        }

        m.records.push_back(std::move(r));

        // update encountered tags pointer
        encounteredRecords[id] = m.records.back().get();

        // register unencountered template type parameters
        if (auto ctsd = dyn_cast<ClassTemplateSpecializationDecl>(declaration)) {

            // get associated class template declaration tparam list
            auto tparamList = ctsd->getSpecializedTemplate()->getTemplateParameters();

            // check if template args match default args
            bool tparamsMatchDefaults = true;
            for (unsigned int i = 0; i < tparamList->size(); i++) {
                if (auto ttpd = dyn_cast<TemplateTypeParmDecl>(tparamList->getParam(i))) {
                    // TODO: look for a better way to compare types
                    if (!ttpd->hasDefaultArgument()
                        || fullyQualifiedId(ttpd->getDefaultArgument())
                               != fullyQualifiedId(ctsd->getTemplateArgs()[i].getAsType())) {
                        tparamsMatchDefaults = false;
                        break;
                    }
                } else {
                    tparamsMatchDefaults = false;
                    break;
                }
            }

            if (tparamsMatchDefaults)
                m.records.back()->alias = ctsd->getSpecializedTemplate()->getNameAsString();

            for (int i = 0; i < ctsd->getTemplateArgs().size(); i++) {
                if (ctsd->getTemplateArgs()[i].getKind() != TemplateArgument::ArgKind::Type)
                    continue;

                const auto type = ctsd->getTemplateArgs()[i].getAsType();
                parseDeclForQualType(type);
            }
        }
    }

    void parseMethod(const CXXMethodDecl* method, elsa::Module::Record& r)
    {
        if (method->isPure())
            r.isAbstract = true;

        // ignore non-public methods, destructors, and conversion functions
        if (method->getAccess() == AS_public && !method->isDeleted()
            && !method->isDeletedAsWritten() && dyn_cast<CXXDestructorDecl>(method) == nullptr
            && dyn_cast<CXXConversionDecl>(method) == nullptr) {

            // skip methods that are ignored in class hints
            if (m.classHints.find(r.name) != m.classHints.end()) {
                const auto& hints = m.classHints[r.name];
                if (hints.ignoredMethods.find(method->getNameAsString())
                    != hints.ignoredMethods.end())
                    return;
            }

            elsa::Module::Function f;
            f.name = method->getNameAsString();
            f.isStatic = method->isStatic();
            f.returnType = fullyQualifiedId(method->getReturnType());
            f.isConst = method->isConst();
            f.refQualifier = getRefQualifierString(method->getRefQualifier());
            f.isConstructor = (dyn_cast<CXXConstructorDecl>(method) != nullptr);
            f.isTemplate = method->isTemplated();
            f.posFirstNonBindableDefaultArg = method->getNumParams();
            parseDeclForQualType(method->getReturnType());

            for (auto param : method->parameters()) {
                f.params.emplace_back(param->getQualifiedNameAsString(),
                                      fullyQualifiedId(param->getType()));

                if (param->hasDefaultArg()) {
                    f.numDefaultArgs++;

                    if (f.posFirstNonBindableDefaultArg == method->getNumParams()) {
                        auto defaultArg = param->getDefaultArg();

                        // uninstantiated default args must be retrieved using a different method
                        if (!defaultArg)
                            defaultArg = param->getUninstantiatedDefaultArg();

                        Expr::EvalResult result;
                        if (defaultArg && !defaultArg->isInstantiationDependent()
                            && !defaultArg->isValueDependent() && !defaultArg->isTypeDependent()
                            && defaultArg->EvaluateAsRValue(result, *context)
                            && !result.hasSideEffects()) {

                            f.params.back().defaultValue =
                                result.Val.getAsString(*context, param->getType());
                        } else {
                            f.posFirstNonBindableDefaultArg = f.params.size() - 1;
                        }
                    }
                }

                parseDeclForQualType(param->getType());
            }

            auto fType = method->getType()->getCanonicalTypeInternal().getAsString();
            r.methods.emplace(fType + " " + f.name, f);
        }
    }

protected:
    void appendMethodTemplateInstantiation(CXXMethodDecl* method)
    {
        auto parent = method->getParent();
        auto id = fullyQualifiedId(context->getRecordType(parent));

        auto it = encounteredRecords.find(id);
        if (it != encounteredRecords.end()) {
            // we know this is a Record
            auto r = it->second;
            if (r == nullptr)
                return;

            auto fType = method->getType()->getCanonicalTypeInternal().getAsString();
            if (r->methods.find(fType + " " + method->getNameAsString()) == r->methods.end())
                parseMethod(method, *r);
        } else {
            // we might encounter implicitly templated lambdas here
            if (!parent->isLambda()) {
                m.includes.insert(getHeaderLocation(parent));
                parseRecord(parent);
            }
        }
    };

    bool shouldBeRegisteredInCurrentModule(const CXXRecordDecl* declaration,
                                           bool considerEncountered = false)
    {
        auto id = fullyQualifiedId(context->getRecordType(declaration));

        auto type = context->getRecordType(declaration);
        type.removeLocalConst();
        type.removeLocalRestrict();
        type.removeLocalVolatile();
        auto noFastQualId = fullyQualifiedId(type);

        // include header for Eigen if necessary
        if (noFastQualId.find("Eigen::") == 0)
            m.pybindIncludes.insert("pybind11/eigen.h");

        // do not register STL types but add necessary pybind includes
        if (context->getFullLoc(declaration->getBeginLoc()).isInSystemHeader()) {

            auto inPybindSTL = {"std::vector<", "std::deque<",        "std::list<",
                                "std::array<",  "std::set<",          "std::unordered_set<",
                                "std::map<",    "std::unordered_map<"};

            if (noFastQualId.find("std::complex<") == 0) {
                m.pybindIncludes.insert("pybind11/complex.h");
            } else if (noFastQualId.find("std::function<") == 0) {
                m.pybindIncludes.insert("pybind11/functional.h");
            } else if (noFastQualId.find("std::chrono::duration<") == 0
                       || noFastQualId.find("std::chrono::time_point<") == 0) {
                m.pybindIncludes.insert("pybind11/chrono.h");
            } else if (m.pybindIncludes.find("pybind11/stl.h") == m.pybindIncludes.end()
                       && std::any_of(inPybindSTL.begin(), inPybindSTL.end(),
                                      [&noFastQualId](const std::string& comp) {
                                          return noFastQualId.find(comp) == 0;
                                      })) {
                m.pybindIncludes.insert("pybind11/stl.h");
            }

            // parse template arguments of STL types
            if (auto ctsd = dyn_cast<ClassTemplateSpecializationDecl>(declaration)) {
                for (int i = 0; i < ctsd->getTemplateArgs().size(); i++) {
                    if (ctsd->getTemplateArgs()[i].getKind() != TemplateArgument::ArgKind::Type)
                        continue;

                    const auto type = ctsd->getTemplateArgs()[i].getAsType();
                    parseDeclForQualType(type);
                }
            }

            return false;
        }

        if (encounteredRecords.find(id) != encounteredRecords.end())
            return considerEncountered ? true : false;

        // otherwise, register in this module if declaration is contained in the module path or a
        // template argument should be registered in this module
        if (fileLocation(declaration).find(m.path) != 0) {
            if (auto ctsd = dyn_cast<ClassTemplateSpecializationDecl>(declaration)) {
                for (int i = 0; i < ctsd->getTemplateArgs().size(); i++) {
                    if (ctsd->getTemplateArgs()[i].getKind() != TemplateArgument::ArgKind::Type)
                        continue;

                    const auto decl = ctsd->getTemplateArgs()[i].getAsType()->getAsCXXRecordDecl();

                    // builtin types (e.g. float) do not have an associated declaration
                    if (decl != nullptr && shouldBeRegisteredInCurrentModule(decl, true))
                        return true;
                }
            }
            return false;
        }

        // printStuff(declaration);
        return true;
    }

    void handleMemberDeclaration(Decl* decl, elsa::Module::Record& r)
    {
        if (dyn_cast<CXXMethodDecl>(decl)) {
            auto method = dyn_cast<CXXMethodDecl>(decl);
            parseMethod(method, r);
        } else if (dyn_cast<FunctionTemplateDecl>(decl)) {
            const auto ftd = dyn_cast<FunctionTemplateDecl>(decl);
            const auto method = dyn_cast<CXXMethodDecl>(ftd->getTemplatedDecl());
            parseMethod(method, r);
            for (const auto spec : ftd->specializations()) {
                const auto method = dyn_cast<CXXMethodDecl>(spec);
                parseMethod(method, r);
            }
        }
    }

    std::string fileLocation(const TagDecl* declaration)
    {
        const auto location = declaration->getBeginLoc().printToString(context->getSourceManager());

        // format of location is path:line:column
        auto path = tooling::getAbsolutePath(location.substr(0, location.find(":")));

        // normalize path so that it does not contain parent directory references (..)
        while (path.find("..") != std::string::npos) {
            auto posDots = path.find("..");

            std::size_t eraseEnd = posDots + 2;
            std::size_t numDots = 1;
            while (path.rfind("..", posDots) == posDots - 3) {
                posDots = path.rfind("..", posDots);
                numDots++;
            }

            std::size_t eraseStart = posDots - 1;
            for (int i = 0; i < numDots; i++)
                eraseStart = path.rfind("/", eraseStart - 1);

            path.erase(eraseStart, eraseEnd - eraseStart);
        }

        return path;
    }
};

class HintsParser : public clang::RecursiveASTVisitor<HintsParser>, public ParserBase
{
public:
    explicit HintsParser(ASTContext* context) : ParserBase{context} {}

    bool shouldVisitTemplateInstantiations() { return true; }

    bool shouldVisitImplicitCode() { return true; }

    bool VisitCXXRecordDecl(CXXRecordDecl* declaration)
    {
        // skip records from includes
        if (!context->getSourceManager().isInMainFile(declaration->getLocation()))
            return true;

        // ignore lambdas (implicitly generated lambda classes may be visited if the
        // visitImplicitCode option is set)
        if (declaration->isLambda())
            return true;

        // ignore template declarations, only template instantiations carry the necessary
        // information for bindings generation
        if (declaration->isTemplated())
            return true;

        // we only care about ClassHints and ModuleHints
        bool isModuleHints = false;
        if (declaration->getNameAsString() == "ModuleHints") {
            isModuleHints = true;
        }

        bool isClassHintsDescendant = false;
        CXXRecordDecl* hintsFor;
        if (declaration->hasDefinition()) {
            for (const auto& base : declaration->bases()) {
                if (base.getType()->getAsCXXRecordDecl() != nullptr) {
                    const auto baseDecl = base.getType()->getAsCXXRecordDecl();
                    if (baseDecl->getNameAsString() == "ClassHints") {
                        isClassHintsDescendant = true;
                        const auto ctsd = dyn_cast<ClassTemplateSpecializationDecl>(baseDecl);
                        hintsFor = ctsd->getTemplateArgs()[0].getAsType()->getAsCXXRecordDecl();
                        break;
                    }
                }
            }
        }

        if (isModuleHints || isClassHintsDescendant) {
            auto location = declaration->getBeginLoc().printToString(context->getSourceManager());
            m.moduleHints.includePath = location.substr(0, location.find(":"));

            if (isModuleHints) {
                for (auto method : declaration->methods()) {
                    if (method->getNameAsString() == "addCustomFunctions") {
                        m.moduleHints.definesGlobalCustomFunctions = true;
                    }
                }
                return true;
            }
        } else {
            return true;
        }

        elsa::Module::ClassHints classHints;
        classHints.recordName = fullyQualifiedId(context->getRecordType(hintsFor));
        classHints.classHintsName = fullyQualifiedId(context->getRecordType(declaration));

        for (const auto decl : declaration->decls()) {
            if (dyn_cast<VarDecl>(decl)) {
                auto varDecl = dyn_cast<VarDecl>(decl);

                if (varDecl->getNameAsString() == "ignoreMethods") {

                    auto initListExpr = dyn_cast<InitListExpr>(
                        dyn_cast<InitListExpr>(varDecl->getInit())->getInit(0)->IgnoreImplicit());

                    llvm::outs() << initListExpr->getNumInits() << "\n";
                    initListExpr->getInit(0)->dump();

                    if (initListExpr) {
                        for (auto arg : initListExpr->inits()) {

                            arg = arg->IgnoreImplicit();

                            if (!dyn_cast<StringLiteral>(arg))
                                assert(false
                                       && "ignoreMethods initializer contains non-string literals");

                            auto methodName = dyn_cast<StringLiteral>(arg)->getString().str();
                            classHints.ignoredMethods.insert(methodName);
                        }
                    }
                    // this may be just a single StringLiteral (and not packed in a
                    // CXXConstructExpr) when we are ignoring only a single method
                    else if (dyn_cast<StringLiteral>(varDecl->getInit()->IgnoreImplicit())) {
                        auto methodName =
                            dyn_cast<StringLiteral>(varDecl->getInit()->IgnoreImplicit())
                                ->getString()
                                .str();
                        classHints.ignoredMethods.insert(methodName);
                    } else {
                        assert(false && "ignoreMethods not inline initialized");
                    }
                }
            }

            else if (dyn_cast<FunctionTemplateDecl>(decl)) {
                auto ftd = dyn_cast<FunctionTemplateDecl>(decl);
                if (ftd->getNameAsString() == "addCustomMethods"
                    || ftd->getNameAsString() == "exposeBufferInfo") {
                    auto addCustomMethods = dyn_cast<CXXMethodDecl>(ftd->getTemplatedDecl());

                    assert(addCustomMethods->isStatic());

                    assert(addCustomMethods->getNumParams() == 1);

                    auto pyClass = *addCustomMethods->param_begin();

                    assert(pyClass->getType().getAsString().substr(0, 11) == "py::class_<");

                    if (ftd->getNameAsString() == "addCustomMethods") {
                        classHints.definesCustomMethods = true;
                    } else {
                        classHints.exposesBufferInfo = true;
                    }
                }
            }
        }

        m.classHints.emplace(classHints.recordName, classHints);

        return true;
    }
};

class ModuleParserConsumer : public clang::ASTConsumer
{
public:
    explicit ModuleParserConsumer(ASTContext* context) : visitor(context) {}

    void HandleTranslationUnit(clang::ASTContext& context) override
    {
        visitor.TraverseDecl(context.getTranslationUnitDecl());
    }

private:
    ModuleParser visitor;
};

class HintsParserConsumer : public clang::ASTConsumer
{
public:
    explicit HintsParserConsumer(ASTContext* context) : visitor(context) {}

    void HandleTranslationUnit(clang::ASTContext& context) override
    {
        visitor.TraverseDecl(context.getTranslationUnitDecl());
    }

private:
    HintsParser visitor;
};

class ModuleParserAction : public clang::ASTFrontendAction
{
public:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& Compiler,
                                                          llvm::StringRef InFile) override
    {
        return std::unique_ptr<clang::ASTConsumer>(
            new ModuleParserConsumer(&Compiler.getASTContext()));
    }
};

class HintsParserAction : public clang::ASTFrontendAction
{
public:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& Compiler,
                                                          llvm::StringRef InFile) override
    {
        return std::unique_ptr<clang::ASTConsumer>(
            new HintsParserConsumer(&Compiler.getASTContext()));
    }
};

int main(int argc, const char** argv)
{
    CommonOptionsParser optionsParser(argc, argv, bindingsOptions);

    assert((!moduleName.empty() || !pythonModuleName.empty())
           && "-name or -pyname should be specified");

    if (!moduleName.empty()) {
        m.name = moduleName;
        if (pythonModuleName.empty())
            m.pythonName = "py" + m.name;
    }

    if (!pythonModuleName.empty()) {
        m.pythonName = pythonModuleName;
        if (m.name.empty())
            m.name = m.pythonName;
    }

    if (!hintsPath.empty()) {
        const char* argv2[2];
        argv2[0] = argv[0];
        argv2[1] = &std::string_view(hintsPath).front();
        int argc2 = 2;

        CommonOptionsParser hints(argc2, argv2, bindingsOptions);

        ClangTool hintsParser(hints.getCompilations(), hints.getSourcePathList());
        hintsParser.appendArgumentsAdjuster(optionsParser.getArgumentsAdjuster());

        auto factory = newFrontendActionFactory<HintsParserAction>();
        int retCode = hintsParser.run(factory.get());

        // fail with the corresponding error code when the hints file can't be parsed
        if (retCode)
            return retCode;
    }

    ClangTool tool(optionsParser.getCompilations(), optionsParser.getSourcePathList());
    tool.appendArgumentsAdjuster(optionsParser.getArgumentsAdjuster());

    std::filesystem::path p(tooling::getAbsolutePath(optionsParser.getSourcePathList().front()));

// normalize path so that it does not contain parent directory references (..)
#if INCLUDE_STD_FILESYSTEM_EXPERIMENTAL
    m.path = ParserBase::lexicallyNormalPathFallback(p.generic_string());
#else
    m.path = p.parent_path().lexically_normal().c_str();
#endif
    // find the longest lexically normal common parent path for all specified source files
    for (std::size_t i = 1; i < optionsParser.getSourcePathList().size(); i++) {
        std::filesystem::path p(tooling::getAbsolutePath(optionsParser.getSourcePathList()[i]));
#if INCLUDE_STD_FILESYSTEM_EXPERIMENTAL
        auto path = ParserBase::lexicallyNormalPathFallback(p.c_str());
#else
        std::string path = p.parent_path().lexically_normal().c_str();
#endif
        while (path.find(m.path) != 0)
            m.path = m.path.substr(0, m.path.rfind("/"));
    }

    auto factory = newFrontendActionFactory<ModuleParserAction>();
    int retCode = tool.run(factory.get());

    // fail with the corresponding error code when a file can't be parsed
    if (retCode)
        return retCode;

    Generator::generateBindingsForModule(m, outputDir);

    return 0;
}