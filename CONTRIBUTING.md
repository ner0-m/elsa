# Contributing to elsa
This page summarizes advice how to contribute to elsa.

## General Workflow

#### a) Internal Developers
- Make a local clone of the elsa repository
- Checkout a new branch and work on your new feature or bugfix
- Push your branch and create a merge request targeting master

#### b) External Contributors
- Fork the elsa git repository
- Make a local clone your own fork
- Checkout a new branch and work on your new feature or bugfix
- Push your branch and send us a merge request targeting master

A branch should be short-lived and specific for a feature or bugfix. Commits should be squashed
into relevant groups before merging into master. Use
```
git rebase -i ${commit_to_rebase_on}
```
to clean up the commit history.

If during the development of a feature, `master` received updates, a `git rebase` should be preferred
over a `git merge` except when the changes are specifically needed in the feature branch:
```
git checkout feature-branch-name
git rebase master
```

Commit messages should follow some guidelines (derived from [this
blogpost](https://chris.beams.io/posts/git-commit/)):

- Separate subject from body with a blank line
- Limit the subject line to 50 characters
- Use the imperative mood in the subject line
- Wrap the body at 72 characters
- Use the body to explain what and why vs. how

## Dependency Management

As of December 2020, we switched to [CPM](https://github.com/TheLartians/CPM.cmake) for our dependency management.
Hopefully, you don't really have to know and it doesn't get the way. See the [root CMakeLists.txt file](./CMakeLists.txt),
as an example for our usage.

It is recommended to set `CPM_SOURCE_CACHE` (see [here](https://github.com/TheLartians/CPM.cmake#cpm_source_cache) for
more info). It's an environment variable, that will save all dependencies outside of the build directory and -
for all projects using CPM - only one version of the dependency. This way no re-downloading is necessary. 
Set it in your e.g. `.bashrc`.
 
## Testing
You can run the elsa unit tests by running `ctest` in the build folder. To specify which tests run,
filter with `ctest -R regular_expression`.

We use a testing style similar described in [Behaviour-Driven
Development](https://github.com/onqtam/doctest/blob/master/doc/markdown/testcases.md#bdd-style-test-cases) (BDD). Please
follow this style when adding new tests. However, isntead of using `SCENARIO` use `TEST_CASE` with the name of the
class under test at the beginning of the test name. Also be sure to add the tests to the test suite associated
to the module of the test.
 
We're currently relying on [doctest](https://github.com/onqtam/doctest/) as our testing framework, when 
using assertion macros, please try to use the 
[binary and unary asserts](https://github.com/onqtam/doctest/blob/master/doc/markdown/assertions.md#binary-and-unary-asserts)
as much as possible. 
 
## Benchmarking
 
You can use the catch testing framework to do [benchmarking
](https://github.com/catchorg/Catch2/blob/master/docs/benchmarks.md). If so, add your benchmarking
case following this template
```cmake
if(ELSA_BENCHMARKS)
    ELSA_TEST(BenchmarkName)
endif()
```
which ensures that the test case is only registered and build if the cmake option was
enabled.

## Style Guide
We use the tool `clang-format` to autoformat our code with the [given style
file](.clang-format). Please make sure that your code is formatted accordingly, otherwise the CI
will fail in the first stage. You can either execute
```
find elsa/ -name '*.h' -or -name '*.hpp' -or -name '*.cpp' | xargs clang-format-10 -i -style=file $1
find examples/ -name '*.h' -or -name '*.hpp' -or -name '*.cpp' | xargs clang-format-10 -i -style=file $1
find benchmark/ -name '*.h' -or -name '*.hpp' -or -name '*.cpp' | xargs clang-format-10 -i -style=file $1
```
in the root folder, setup a git hook or integrate `clang-format` into your IDE. Note that we
currently use version 10.0.0, different versions might produce errors.

## Linting
We use `clang-tidy` with the enabled checks specified in [the configuration file](.clang-tidy). Note
that currently all `readability-*` checks have to pass, otherwise the CI will fail. We encourage
developers to check their code with `clang-tidy` and remove all warnings if applicable.
 
#### CMake 

We use [cmakelang](https://cmake-format.readthedocs.io/en/latest/index.html) to enforce
certain style guide and reduce the changes of error in our CMake code, please check the guide to install it.

Currently, only the `cmake-lint` is used, but sooner rather than later, we'll also start
using `cmake-format` of the same package.

Please check the link above on how to install the package.

## Code Coverage
We use `lcov` with `gcov` for test coverage information. If you want to run this locally you have to
do a debug build with coverage option enabled
```
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DELSA_COVERAGE=ON ../
make all -j4
make test_coverage
```
and then the results should be available at `build/test_coverage/index.html`. You can compare your
local results to [the latest master coverage results](https://ciip.in.tum.de/elsacoverage/).

## pre-commit

There is also a basic `.pre-commit-config.yaml` file to install pre-commit hooks using 
[pre-commit](https://pre-commit.com/). You are highly encouraged to install the pre-commits
with `pre-commit install` such that they are run before each commit.

None of the commit hooks will change anything in your commit, they mearly check and error if
something is wrong.
