#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ASTContext.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "clang/AST/QualTypeNames.h"

#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <iterator>

#include "json.hpp"

using json = nlohmann::json;

using namespace clang;
using namespace clang::tooling;

// Helper functions:
void replaceAllInString(std::string &str, const std::string &from, const std::string &to) {
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
}

bool stringStartsWith(std::string const &fullString, std::string const &beginning) {
    if (fullString.length() >= beginning.length()) {
        return fullString.compare(0, beginning.length(), beginning) == 0;
    } else {
        return false;
    }
}

bool stringEndsWith(std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return fullString.compare(fullString.length() - ending.length(), ending.length(), ending) == 0;
    } else {
        return false;
    }
}

std::vector<std::string> split(std::stringstream &ss, char delim) {
    std::vector<std::string> resultElems;
    std::string currItem;
    auto resultInserter = std::back_inserter(resultElems);
    while (std::getline(ss, currItem, delim)) {
        *(resultInserter++) = currItem;
    }
    return resultElems;
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::stringstream ss(s);
    return split(ss, delim);
}

bool isTrue(std::string &string) {
    return string == "1" || string == "true";
}

/**
 * Class for building a json tree structure with information from a given AST. The json tree structure has approximately
 * the same shape than the given AST (see below for more information).
 */
class Ast2Json {

    ASTContext &Context;
    const std::string &projectPath;
    bool addEndNodes;
    bool addStaticKindInfo;

public:
    /**
     *
     * @param astContext The AST from which information should be extracted
     * @param projectPath Must be the empty string or an absolute path. If not empty, only AST nodes whose
     * expanded location's path starts with projectPath are considered for the resulting json tree structure
     * @param addEndNodes Whether to also add additional "end-of" nodes to the json tree structure. An end-of node will
     * be added as sibling for each node, that has children. Due to this, tree structure can be reconstructed after the
     * json tree has been flattened (e.g. after sequencing with a depth-search traversal)
     * @param addStaticKindInfo Whether information about possible Clang kind (Stmt, Expr, Type, ...) values should
     * be attached to the json tree root object. This information does not depend on the given AST.
     */
    Ast2Json(ASTContext &astContext, const std::string &projectPath, bool addEndNodes, bool addStaticKindInfo)
            : Context(astContext), projectPath(projectPath), addEndNodes(addEndNodes),
              addStaticKindInfo(addStaticKindInfo) {
        // Check if projectPath is empty or ends with a path separator:
        if (projectPath.length() > 0
            && !stringEndsWith(projectPath, std::string("/"))
            && !stringEndsWith(projectPath, std::string("\\"))
                ) {
            throw std::invalid_argument("projectPath must be empty or end with a path separator (either / or \\).");
        }
    }

    /**
     * Performs the information extraction and json tree structure generation
     * @return root of json tree structure
     */
    json produceJson() {
        return getAstRootJson(getNodeJson(dyn_cast<Decl>(Context.getTranslationUnitDecl())));
    }

private:

    typedef std::pair<nlohmann::basic_json<>, nlohmann::basic_json<> > NodeJsonResultPairType;

    template<typename NodeType>
    SourceLocation getSourceLoc(NodeType *N) {
        return getSourceRange(N).getBegin();
    }

    template<typename NodeType>
    SourceRange getSourceRange(NodeType *N) {
        // Use ExpansionLoc everywhere!
        // SpellingLoc: Location where the node's code was actually written
        // (i.e. for a node in a macro it returns the location of the macro definition)
        // ExpansionLoc: Location where the node is located after macro expansion and so on
        // (i.e. for a node in a macro it returns the location where the macro is used/expanded)
        return SourceRange(Context.getSourceManager().getExpansionLoc(N->getSourceRange().getBegin()),
                           Context.getSourceManager().getExpansionLoc(N->getSourceRange().getEnd()));
    }

    bool isInMainFile(SourceLocation loc) {
        return Context.getSourceManager().isWrittenInMainFile(loc);
    }

    json getSourceLocJson(SourceLocation loc, bool omitFile = false, bool useMainFilePlaceholder = true) {
        if (loc.isValid()) {
            json resultJson = {
                    // We can use the Spelling variants here because loc is already an expansionloc and therefore
                    // does not need to get expanded again
                    {"line", Context.getSourceManager().getSpellingLineNumber(loc)},
                    {"col",  Context.getSourceManager().getSpellingColumnNumber(loc)}
            };
            if (!omitFile) {
                // Avoid adding the same (long) file path too often by adding placeholder for locations in main file
                // if requested
                if (useMainFilePlaceholder && isInMainFile(loc)) {
                    resultJson["file"] = ":MN:";
                } else {
                    resultJson["file"] = Context.getSourceManager().getFilename(loc);
                }
            }
            return resultJson;
        }
        return nullptr;
    }

    json getSourceRangeJson(SourceRange range) {
        // e.g. ImplicitCastExpr has valid range but invalid start and end
        if (range.isValid()) {
            // Only add file name to start:
            return {
                    {"start", getSourceLocJson(range.getBegin(), false)},
                    {"end",   getSourceLocJson(range.getEnd(), true)},
            };
        } else {
            return nullptr;
        }
    }

    template<typename NodeType>
    bool isOutOfProject(NodeType *D) {
        SourceLocation loc = getSourceLoc(D);
        // std::cout << getLocationJson(D->getSourceRange().getBegin()) << std::endl;
        if (loc.isValid()) {
            // Always allow the main file itself
            if (isInMainFile(loc)) {
                return false;
            }
            if (!Context.getSourceManager().isInSystemHeader(loc)) {
                if (projectPath.length() > 0) {
                    // Check whether file path starts with project path (assuming projectPath is ending with os path
                    // separator):
                    std::string filePath = Context.getSourceManager().getFilename(loc);
                    if (filePath.length() > 0 && filePath.find(projectPath) == 0) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    void addChildToJson(json &jsonObj, NodeJsonResultPairType &&childPair) {
        if (childPair.first != nullptr) { // is_null
            // Add node itself
            jsonObj["children"].push_back(childPair.first);
            // Check for end-of node:
            if (childPair.second != nullptr) { // is_null
                // Add the node's end-of node:
                jsonObj["children"].push_back(childPair.second);
            }
        }
    }

    std::string getOperatorKindString(Stmt *node) {
        if (auto *uo = dyn_cast<UnaryOperator>(node)) {
            return UnaryOperator::getOpcodeStr(uo->getOpcode()).str();
        } else if (auto *bo = dyn_cast<BinaryOperator>(node)) {
            return BinaryOperator::getOpcodeStr(bo->getOpcode()).str();
        } else if (isa<ArraySubscriptExpr>(node)) {
            return "[]";
        } else if (isa<ConditionalOperator>(node)) {
            return "?:";
        } else {
            return "";
        }
    }

    json getOperatorKindJson(Stmt *node) {
        std::string operString = getOperatorKindString(node);
        if (operString.length() > 0) {
            return operString;
        } else {
            return nullptr;
        }
    }

    json getNodeValueJson(Stmt *S) {
        // TODO: Enum Values?
        json value = nullptr;
        if (auto *FL = dyn_cast<FloatingLiteral>(S)) {
            value = FL->getValueAsApproximateDouble();
        } else if (auto *IL = dyn_cast<IntegerLiteral>(S)) {
            value = IL->getValue().getSExtValue();
        } else if (auto *FPL = dyn_cast<FixedPointLiteral>(S)) {
            value = FPL->getValue().getSExtValue();
        } else if (auto *CL = dyn_cast<CharacterLiteral>(S)) {
            value = CL->getValue();
        } else if (auto *SL = dyn_cast<StringLiteral>(S)) {
            // C++ support for unicode/utf/wide strings is not good. Thus, do not try to convert unicode/utf/wide string
            // literals (which may not contain non-ascii chars at all) to an ascii or utf8 representation. Instead,
            // assume that the source file itself does not contain non-ascii chars and use the string literal as
            // written in source code as json value. The actual decoding/conversion is done in the calling application
            // then.
            std::string stringLiteralAsWritten;
            llvm::raw_string_ostream stringLiteralAsWrittenStream(stringLiteralAsWritten);
            SL->outputString(stringLiteralAsWrittenStream);
            value = stringLiteralAsWrittenStream.str();
        } else if (auto *BL = dyn_cast<CXXBoolLiteralExpr>(S)) {
            value = BL->getValue();
        } else if (isa<CXXNullPtrLiteralExpr>(S) || isa<GNUNullExpr>(S)) {
            value = 0.0;
        }
        return value;
    }

    json getKindJson(std::string &&coarseKindName, std::string &&fineKindName, int fineKindNumber) {
        return json::array({coarseKindName, fineKindName, fineKindNumber});
    }

    json getTypeJson(QualType type) {
        if (type.isNull()) {
            return nullptr;
        }
        // Look through elaborated type:
        if (auto *et = dyn_cast<ElaboratedType>(type)) {
            type = et->getNamedType();
        }
        // Look through typedefs, ...:
        type = type.getCanonicalType();
        // Remember const and volatile before removing it
        bool isConst = type.isConstQualified();
        bool isVolatile = type.isVolatileQualified();
        // Remove qualifiers:
        type = type.getUnqualifiedType();
        // Basic type json (more key-value-pairs may be added below):
        json typeJson = {
                {"kind",     getKindJson("Type", type->getTypeClassName(), type->getTypeClass())},
                {"id",       getIdentifierJson(type)},
                {"is_const", isConst},
                {"is_vola",  isVolatile},
                {"is_pod",   type.isPODType(Context)},
        };

        if (auto *at = dyn_cast<ArrayType>(type)) {
            typeJson["arr_elem_type"] = getTypeJson(at->getElementType());
        }
        if (auto *cat = dyn_cast<ConstantArrayType>(type)) {
            typeJson["array_size"] = cat->getSize().getSExtValue();
        }
        if (type->isAnyPointerType()) {
            typeJson["pointee_type"] = getTypeJson(type->getPointeeType());
        }
        if (auto *rt = dyn_cast<ReferenceType>(type)) {
            typeJson["reference_type"] = getTypeJson(rt->getPointeeType());
        }
        if (type->isFunctionType()) {
            typeJson["result_type"] = getTypeJson(dyn_cast<FunctionType>(type)->getReturnType());
        }
        return typeJson;
    }

    json getIdentifierJson(Decl *N) {
        json resultJson;
        if (auto *nd = dyn_cast<NamedDecl>(N)) {
            N = nd->getUnderlyingDecl();
        }
        resultJson["hash"] = reinterpret_cast<uint64_t>(N);
        // Whether this decl is outside the project.
        // This can happen if decl is referenced in a DeclRefExpr or MemberExpr
        bool outOfProject = isOutOfProject(N);
        if (outOfProject) {
            // This is quite uncommon and therefore it is inside an if to avoid lots of "in_project":true
            resultJson["in_project"] = false;
        }
        if (auto *nd = dyn_cast<NamedDecl>(N)) {
            // Use getNameForDiagnostic over getQualifiedNameAsString for template args
            std::string qualifiedName;
            llvm::raw_string_ostream qualifiedNameStream(qualifiedName);
            // nd->getNameForDiagnostic(qualifiedNameStream, Context.getPrintingPolicy(), true);
            printQualifiedNameWithFunctionsAndTemplates(nd, qualifiedNameStream, Context.getPrintingPolicy());

            // Generate spelling from file path together with fully qualified id:
            // Get file path (which may be not available (e.g. in case of implicit functions)):
            json locJson = getSourceLocJson(getSourceLoc(nd), /*omitFile=*/ false, /*useMainFilePlaceholder=*/ false);
            std::string filePath = "<NA>";
            if (locJson["file"] != nullptr) {
                filePath = locJson["file"].get<std::string>();
                // If filePath is in project, remove the project path prefix as it is the same for all in-project files:
                if (!outOfProject && projectPath.length() > 0 && stringStartsWith(filePath, projectPath)) {
                    filePath.replace(0, projectPath.length(), std::string());
                }
            }
            resultJson["spelling"] = filePath + std::string(":") + qualifiedNameStream.str();
        } else if (auto *tud = dyn_cast<TranslationUnitDecl>(N)) {
            // Use file name of TU as spelling:
            resultJson["spelling"] = Context.getSourceManager().getFileEntryForID(
                    Context.getSourceManager().getMainFileID())->getName().str();
        }
        return resultJson;
    }

    json getIdentifierJson(Stmt *N) {
        json resultJson;
        resultJson["hash"] = reinterpret_cast<uint64_t>(N);
        // resultJson["spelling"] = nullptr;
        return resultJson;
    }

    json getIdentifierJson(QualType N) {
        json resultJson;
        resultJson["hash"] = reinterpret_cast<uint64_t>(N.getTypePtrOrNull());
        resultJson["spelling"] = TypeName::getFullyQualifiedName(N, Context, Context.getPrintingPolicy(), true);
        return resultJson;
    }


    NodeJsonResultPairType getNodeJson(Decl *D, int depth = 0) {
        if (!D || D->getKind() == 0 || (isOutOfProject(D) && !isa<TranslationUnitDecl>(D))) {
            return NodeJsonResultPairType(nullptr, nullptr);
        }

        QualType type;
        Expr *init = nullptr;
        if (auto *nd = dyn_cast<NamedDecl>(D)) {
            D = nd->getUnderlyingDecl();
            //name = nd->getName().str();
        }
        if (auto *vd = dyn_cast<ValueDecl>(D)) {
            type = vd->getType();
        }
        if (auto *td = dyn_cast<TypeDecl>(D)) {
            type = QualType(td->getTypeForDecl(), 0);
        }
        if (auto *vd = dyn_cast<VarDecl>(D)) {
            if (vd->hasInit()) {
                init = vd->getInit();
            }
        }

        bool is_def = false;
        if (auto *fd = dyn_cast<FunctionDecl>(D)) {
            is_def = fd->isThisDeclarationADefinition();
        } else if (auto *ftd = dyn_cast<FunctionTemplateDecl>(D)) {
            is_def = ftd->isThisDeclarationADefinition();
        } else if (auto *td = dyn_cast<TagDecl>(D)) {
            is_def = td->isThisDeclarationADefinition();
        } else if (auto *ctd = dyn_cast<ClassTemplateDecl>(D)) {
            is_def = ctd->isThisDeclarationADefinition();
        } else if (auto *vd = dyn_cast<VarDecl>(D)) {
            is_def = vd->isThisDeclarationADefinition();
        } else if (auto *vtd = dyn_cast<VarTemplateDecl>(D)) {
            is_def = vtd->isThisDeclarationADefinition();
        }

        json resultJson = {
                {"kind",     getKindJson("Decl", D->getDeclKindName(), D->getKind())},
                {"id",       getIdentifierJson(D)},
                {"type",     getTypeJson(type)},
                {"is_def",   is_def},
                {"extent",   getSourceRangeJson(getSourceRange(D))},
                //{"hash", D->getID()}, // with clang 8 or 9?
                {"children", json::array()},
        };

        // Add template params and underlying declaration
        if (auto *td = dyn_cast<TemplateDecl>(D)) {
            if (auto *tpl = td->getTemplateParameters()) {
                for (NamedDecl *templateParamDecl : *tpl) {
                    addChildToJson(resultJson, getNodeJson(templateParamDecl, depth + 1));
                }
            }
            // Underlying declaration:
            addChildToJson(resultJson, getNodeJson(td->getTemplatedDecl(), depth + 1));
        }

        // Add parameters and init expression if any:
        if (auto *fd = dyn_cast<FunctionDecl>(D)) {
            for (ParmVarDecl *pvd : fd->parameters()) {
                addChildToJson(resultJson, getNodeJson(pvd, depth + 1));
            }
        }
        if (init) {
            addChildToJson(resultJson, getNodeJson(init, depth + 1));
        }
        // Add body/sub-decls
        if (D->hasBody()) { // body will reach decls of the context at some point
            addChildToJson(resultJson, getNodeJson(D->getBody(), depth + 1));
        }
            // if no body just print the decls (e.g. namespace)
        else if (auto *DC = dyn_cast<DeclContext>(D)) {
            for (Decl *decl : DC->decls()) {
                addChildToJson(resultJson, getNodeJson(decl, depth + 1));
            }
        }
        // Add end-of node if necessary:
        return NodeJsonResultPairType(resultJson, getEndOfNodeJson(resultJson));
    }

    NodeJsonResultPairType getNodeJson(Stmt *S, int depth = 0) {
        if (!S || isOutOfProject(S)) {
            return NodeJsonResultPairType(nullptr, nullptr);
        }

        QualType type;
        std::string coarseKind = "Stmt";
        if (Expr *E = dyn_cast<Expr>(S)) {
            type = E->getType();
            coarseKind = "Expr";

            // Skip implicit stuff: BEHAVIOUR WILL CHANGE IN Clang 8 OR 9!
            Expr *prevE = nullptr;
            do {
                prevE = E;
                // Parenthesis, noop casts, and SubstNonTypeTemplateParmExpr:
                E = E->IgnoreParenNoopCasts(Context);
                // ExprWithCleanups, MaterializeTemporaryExpr, CXXBindTemporaryExpr, and
                // ImplicitCastExpr (not in a loop!)
                E = E->IgnoreImplicit();
            } while (E != prevE);
            S = E;
        }

        json resultJson = {
                {"kind",     getKindJson(std::move(coarseKind), S->getStmtClassName(), S->getStmtClass())},
                {"id",       getIdentifierJson(S)},
                {"type",     getTypeJson(type)},
                {"extent",   getSourceRangeJson(getSourceRange(S))},
                {"children", json::array()},
        };
        auto operJson = getOperatorKindJson(S);
        if (operJson != nullptr) {
            resultJson["operator"] = operJson;
        }
        auto valJson = getNodeValueJson(S);
        if (valJson != nullptr) {
            resultJson["value"] = valJson;
        }

        if (auto *dre = dyn_cast<DeclRefExpr>(S)) {
            if (NamedDecl *nd = dre->getFoundDecl()) { // Looks through using decls, etc
                resultJson["ref_id"] = getIdentifierJson(nd);
            }
        } else if (auto *me = dyn_cast<MemberExpr>(S)) { // Member access a->b, a.b (a is part of sub-ast. Add b)
            if (ValueDecl *vd = me->getMemberDecl()) { // Not sure if this can be null (for overloaded methods?)
                resultJson["ref_id"] = getIdentifierJson(vd);
            }
        } else if (auto *ce = dyn_cast<CallExpr>(S)) { // Member access a->b, a.b (a is part of sub-ast. Add b)
            if (FunctionDecl *fd = ce->getDirectCallee()) { // null e.g. in case of call of function pointer
                resultJson["ref_id"] = getIdentifierJson(fd);
            }
        }

        // Children:
        if (auto *DS = dyn_cast<DeclStmt>(S)) {
            for (auto *decl : DS->decls()) {
                addChildToJson(resultJson, getNodeJson(decl, depth + 1));
            }
        } else {
            for (auto *child : S->children()) {
                addChildToJson(resultJson, getNodeJson(child, depth + 1));
            }
        }
        // Add end-of node if necessary:
        return NodeJsonResultPairType(resultJson, getEndOfNodeJson(resultJson));
    }

    json getEndOfNodeJson(json &resultJson) {
        if (!addEndNodes) {
            return nullptr;
        }
        if (resultJson["children"].empty()) {
            // Do not End for single node constructs
            return nullptr;
        }
        json endOfKindJson = resultJson["kind"];

        return {
                {"kind",     getKindJson(std::string("EndOf") + std::string(endOfKindJson[0]),
                                         endOfKindJson[1],
                                         endOfKindJson[2])},
                {"id",       resultJson["id"]},
                {"children", json::array()},
        };
    }

    // Necessary when addEndOfNodes=true because there are two root nodes (actual TU node and end-of TU node)
    // associated with the translation unit then
    json getAstRootJson(NodeJsonResultPairType &&translationUnitNodes) {
        json astRootJson = {
                {"kind",     getKindJson(std::string("AstRoot"), "", -1)},
                {"children", translationUnitNodes},
        };
        if (addStaticKindInfo) {
            // Kind number intervals:
            astRootJson["kind_number_intervals"] = {
                    {"Decl", {Decl::Kind::firstDecl,              Decl::Kind::lastDecl}},
                    {"Stmt", {Stmt::StmtClass::firstStmtConstant, Stmt::StmtClass::lastStmtConstant}},
                    {"Type", {0,                                  Type::TypeClass::TypeLast}},
                    {"Expr", {Stmt::StmtClass::firstExprConstant, Stmt::StmtClass::lastExprConstant}},
            };

            // Compute kind info json for each possible kind
            std::vector<std::pair<const char *, const char *(*)(unsigned)>> kindsComputeInfo
                    = {{"Decl", &getDeclFineKindName},
                       {"Stmt", &getStmtFineKindName}, // includes Expr
                       {"Type", &getTypeFineKindName}};
            for (auto kindComputeInfo : kindsComputeInfo) {
                const char *&coarseKindName = kindComputeInfo.first;
                auto &getFineKindNameFromFineKindNumberFunc = kindComputeInfo.second;
                json kindJsons = json::array();
                for (unsigned fineKindNumber = astRootJson["kind_number_intervals"][coarseKindName][0];
                    // "<=" because intervals are inclusive:
                     fineKindNumber <= astRootJson["kind_number_intervals"][coarseKindName][1];
                     fineKindNumber++
                        ) {
                    auto coarseKindNameForKindJson = coarseKindName;
                    // Expr special case:
                    if (std::string(coarseKindName) == "Stmt"
                        && fineKindNumber >= astRootJson["kind_number_intervals"]["Expr"][0]
                        && fineKindNumber <= astRootJson["kind_number_intervals"]["Expr"][1]
                            ) {
                        coarseKindNameForKindJson = "Expr";
                    }
                    kindJsons.push_back(getKindJson(coarseKindNameForKindJson,
                                                    getFineKindNameFromFineKindNumberFunc(fineKindNumber),
                                                    fineKindNumber));
                }
                astRootJson["kinds"][coarseKindName] = kindJsons;
            }
        }
        return astRootJson;
    }

    static const char *getDeclFineKindName(unsigned declFineKindNumber) {
        switch (declFineKindNumber) {
            default:
                llvm_unreachable("Declaration not in DeclNodes.inc!");
#define DECL(DERIVED, BASE) case Decl::DERIVED: return #DERIVED;
#define ABSTRACT_DECL(DECL)

#include "clang/AST/DeclNodes.inc"
        }
    }

    static const char *getStmtFineKindName(unsigned stmtFineKindNumber) {
        switch (stmtFineKindNumber) {
            default:
                llvm_unreachable("Stmt not in StmtNodes.inc!");
#define ABSTRACT_STMT(STMT)
#define STMT(CLASS, PARENT) case Stmt::CLASS##Class: return #CLASS;
//   StmtClassInfo[(unsigned)Stmt::CLASS##Class].Name = #CLASS;    \
//   StmtClassInfo[(unsigned)Stmt::CLASS##Class].Size = sizeof(CLASS);
#include "clang/AST/StmtNodes.inc"
        }
    }

    static const char *getTypeFineKindName(unsigned typeFineKindNumber) {
        switch (typeFineKindNumber) {
            default:
                llvm_unreachable("Type not in TypeNodes.inc!");
#define ABSTRACT_TYPE(Derived, Base)
#define TYPE(Derived, Base) case Type::Derived: return #Derived;

#include "clang/AST/TypeNodes.def"
        }
    }


    // Copied from Clang 7 and modified to also print function contexts
    void printQualifiedNameWithFunctionsAndTemplates(const NamedDecl *namedDecl, raw_ostream &OS,
                                                     const PrintingPolicy &P) const {
        const DeclContext *Ctx = namedDecl->getDeclContext();

        // For ObjC methods, look through categories and use the interface as context.
        if (auto *MD = dyn_cast<ObjCMethodDecl>(namedDecl))
            if (auto *ID = MD->getClassInterface())
                Ctx = ID;

//Simon: Commented out:
//        if (Ctx->isFunctionOrMethod()) {
//            printName(OS);
//            return;
//        }

        using ContextsTy = SmallVector<const DeclContext *, 8>;
        ContextsTy Contexts;

        //Simon: If the decl-to-print is a TagDecl, then handle the decl
        // itself like its contexts. Thus e.g. template arguments of the decl
        // are printed.
        bool declIsHandled = false;
        if (isa<TagDecl>(namedDecl)) {
            Contexts.push_back(cast<TagDecl>(namedDecl));
            declIsHandled = true;
        }

        // Collect named contexts.
        while (Ctx) {
            if (isa<NamedDecl>(Ctx))
                Contexts.push_back(Ctx);
            Ctx = Ctx->getParent();
        }

        for (const DeclContext *DC : llvm::reverse(Contexts)) {
            if (const auto *Spec = dyn_cast<ClassTemplateSpecializationDecl>(DC)) {
                OS << Spec->getName();
                const TemplateArgumentList &TemplateArgs = Spec->getTemplateArgs();
                printTemplateArgumentList(OS, TemplateArgs.asArray(), P);
            } else if (const auto *ND = dyn_cast<NamespaceDecl>(DC)) {
                if (P.SuppressUnwrittenScope &&
                    (ND->isAnonymousNamespace() || ND->isInline()))
                    continue;
                if (ND->isAnonymousNamespace()) {
                    OS << (P.MSVCFormatting ? "`anonymous namespace\'"
                                            : "(anonymous namespace)");
                } else
                    OS << *ND;
            } else if (const auto *RD = dyn_cast<RecordDecl>(DC)) {
                if (!RD->getIdentifier())
                    OS << "(anonymous " << RD->getKindName() << ')';
                else
                    OS << *RD;
            } else if (const auto *FD = dyn_cast<FunctionDecl>(DC)) {
                const FunctionProtoType *FT = nullptr;
                if (FD->hasWrittenPrototype())
                    FT = dyn_cast<FunctionProtoType>(FD->getType()->castAs<FunctionType>());

                OS << *FD << '(';
                if (FT) {
                    unsigned NumParams = FD->getNumParams();
                    for (unsigned i = 0; i < NumParams; ++i) {
                        if (i)
                            OS << ", ";
                        OS << FD->getParamDecl(i)->getType().stream(P);
                    }

                    if (FT->isVariadic()) {
                        if (NumParams > 0)
                            OS << ", ";
                        OS << "...";
                    }
                }
                OS << ')';
            } else if (const auto *ED = dyn_cast<EnumDecl>(DC)) {
                // C++ [dcl.enum]p10: Each enum-name and each unscoped
                // enumerator is declared in the scope that immediately contains
                // the enum-specifier. Each scoped enumerator is declared in the
                // scope of the enumeration.
                // For the case of unscoped enumerator, do not include in the qualified
                // name any information about its enum enclosing scope, as its visibility
                // is global.
                if (ED->isScoped())
                    OS << *ED;
                else
                    continue;
            } else {
                OS << *cast<NamedDecl>(DC);
            }
            OS << "::";
        }

        // Simon: Added if for declIsHandled
        if (!declIsHandled) {
            if (namedDecl->getDeclName() || isa<DecompositionDecl>(namedDecl))
                OS << *namedDecl;
            else
                OS << "(anonymous)";
        }
    }
};

int main(int argc, const char **argv) {
    llvm::cl::OptionCategory ToolCategory("Ast2Json");
    CommonOptionsParser OptParser(argc, argv, ToolCategory);

    // Read arguments
    // TODO: Do the whole option parsing properly!
    std::vector<std::string> argumentSourcePaths = OptParser.getSourcePathList();
    std::string outPath;
    std::string projectPath;
    bool addEndNodes = false;
    bool addStaticKindInfo = false;
    bool verbose = false;

    // OptParser.getSourcePathList() is abused here because the arguments are not source paths anymore
    auto sourcePathCollectionJsons = std::vector<json>();
    if (argumentSourcePaths.size() == 6 && stringEndsWith(argumentSourcePaths[0], ".jsonl")) {
        outPath = argumentSourcePaths[1];
        projectPath = argumentSourcePaths[2];
        addEndNodes = isTrue(argumentSourcePaths[3]);
        addStaticKindInfo = isTrue(argumentSourcePaths[4]);
        verbose = isTrue(argumentSourcePaths[5]);

        // Read actual paths from file specified by first argument:
        std::ifstream fileWithPaths(argumentSourcePaths[0]);
        std::stringstream buffer;
        buffer << fileWithPaths.rdbuf();
        auto sourcePathCollectionJsonStrings = split(buffer, '\n');
        for (const auto &sourcePathCollectionJsonString : sourcePathCollectionJsonStrings) {
            sourcePathCollectionJsons.push_back(json::parse(sourcePathCollectionJsonString));
        }
    } else {
        // Add all path arguments as single source path collection:
        sourcePathCollectionJsons.emplace_back(argumentSourcePaths);
    }

    // Determine where the final json string should be output to:
    std::ostream *output;
    if (outPath.length() > 0) {
        output = new std::ofstream(outPath);
    } else {
        output = &std::cout;
    }

    for (json &sourcePathCollectionJson : sourcePathCollectionJsons) {
        // Convert json string array to std::vector<string>:
        auto sourcePaths = std::vector<std::string>();
        for (std::string sourcePath : sourcePathCollectionJson) {
            sourcePaths.emplace_back(sourcePath);
        }

        // Create tool for parsing of current source path collection:
        ClangTool tool(OptParser.getCompilations(), sourcePaths);
        // Parse and create an ast for each source file:
        std::vector<std::unique_ptr<ASTUnit>> asts;
        tool.buildASTs(asts);

        json outputJsonList = json::array();
        for (auto &ast : asts) {
            ASTUnit &astUnit = *ast;
            if (!astUnit.getDiagnostics().hasUnrecoverableErrorOccurred()) {
                ASTContext &astContext = astUnit.getASTContext();
                Ast2Json ast2json(astContext, projectPath, addEndNodes, addStaticKindInfo);
                if (verbose) {
                    std::cout << "Resulting AST for " << sourcePathCollectionJson << ":" << std::endl;
                    astContext.getTranslationUnitDecl()->dump(llvm::outs());
                }
                // Extract desired information from the ast, convert it to json, and store it in outputJsonList
                outputJsonList.push_back(ast2json.produceJson());

            } else {
                // Add null to keep line number matchings between input file path list and output json list
                outputJsonList.push_back(nullptr);
            }
        }
        *output << outputJsonList << std::endl;
    }

    if (outPath.length() > 0) {
        dynamic_cast<std::ofstream *>(output)->close();
        delete output;
    }
}
