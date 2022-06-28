# Contributing to elsa

We are happy to accept contributions. This should help to get you started.

First, feel free to contact the core developers with whatever inquiry or question you have.
This is easiest if you join our [Matrix chat room](https://matrix.to/#/#elsa:in.tum.de) and start
chatting!

elsa offers many different areas to work in. Check the
[milestones](https://gitlab.lrz.de/IP/elsa/-/milestones) and/or the [issue
tracker](https://gitlab.lrz.de/IP/elsa/-/issues) for topics that might interest you. If no issue
exits, feel free to open one and get in touch with the core developers. To avoid frustration and
work done for nothing, we encourage you to get in touch with us early. We don't bite
:stuck_out_tongue_winking_eye:

If you are a student of the Technical University of Munich and want your work to be part of you
curriculum, also get in touch with us.

## Kind of Contributions

### Bug reports

Report bugs to our [issue tracker](https://gitlab.lrz.de/IP/elsa/-/issues). In the description
include information about the elsa version, your system, a (if possible minimal) step by step guide
or example code to reproduce the error and what the expected behaviour should be.

### Fix issues

If you find and issue on the [issue tracker](https://gitlab.lrz.de/IP/elsa/-/issues), you are
interested in, feel free to work on it. Leave a message in the respective issue, to see if anyone is
already working on it.

### New features or refactors

There are many different areas elsa still needs improvement and new features.
As mentioned above, check the [milestones](https://gitlab.lrz.de/IP/elsa/-/milestones) and/or the
[issue tracker](https://gitlab.lrz.de/IP/elsa/-/issues). If you want to work on something
interesting, which is not on either of the lists, open an issue explain what you want to do
and why it would be useful.

If you find some places, which don't follow certain best practices for the given area, feel free to
get in touch and work on it. We are always trying to keep up with best practices.

### Improve documentation and examples

We are always happy to hear back if our documentation is unclear and can be improved. If you want
and can improve it yourself we are happy to accept contributions!

### Improve workflow (i.e. CI, infrastructure)

There is quite some work involved in keeping things like the CI pipeline and enforce certain best
practices, if possible, automatically. If you have ideas for improving in this area and are
interested in digging into DevOp work, feel free to get in touch.

## General Workflow

In the following a very short description is given about the workflow used. Most of the part will be
kept high-level and point to other resources. Certain details, about git commands and such are
rather specific and depend highly on your personal workflow. However, they will get you started and
can be used if in doubt.

The repository follows a similar workflow to
[OneFlow](https://www.endoflineblog.com/oneflow-a-git-branching-model-and-workflow). If you are
unfamiliar with it, you can check it out. But simplified, there exists one main branch (`master`),
from which all work is started. For your work you create a new branch and push is upstream, we'll
review it, and once it's done it will be rebased onto the `master` branch. However there is a
difference for internal developers and externals one, summarized in the following section:

#### a) Internal Developers
- Make a local clone of the elsa repository
- Checkout a new branch and work on your new feature or bugfix
- Push your branch and create a merge request targeting master

#### b) External Contributors
- Fork the elsa git repository
- Make a local clone your own fork
- Checkout a new branch and work on your new feature or bugfix
- Push your branch and send us a merge request targeting master

#### Some common guidelines

If you've not worked with such a system so far, don't worry. Here are some tips which make live
easier.

1. A branch should be short-lived and specific for a feature or bugfix. If you think the single
   feature is to large, then split it into multiple partial features. It's absolutely acceptable to
   have a partial feature (e.g. which is not used anywhere yet) merged into the main branch. The
   next merge request can work on that.
2. If during the development of a feature, `master` received updates, a `git rebase` should be
   preferred over a `git merge` except when the changes are specifically needed in the feature
   branch:
   ```
   git checkout feature-branch-name
   git rebase master
   ```
3. Squash related commits together and commit often (see [this blog
   post](https://sethrobertson.github.io/Git Best Practices/) for more on this). Commits are
   basically your signpost. They show you the way back, if you need it. Once you're done, squash
   them together. Use
   ```
   git rebase -i ${commit_to_rebase_on}
   ```
   to clean up the commit history. This is easiest if you didn't push so far. But **if** you are the
   only person working on the branch, rewriting history isn't a problem. But please be sure you are
   the only one, because else, it can be troublesome.
4. Commit messages should follow [Conventional Commit](https://www.conventionalcommits.org/en/v1.0.0/).

some guidelines (derived from [this
   blogpost](https://chris.beams.io/posts/git-commit/)):
   - Separate subject from body with a blank line
   - Limit the subject line to 50 characters
   - Use the imperative mood in the subject line
   - Wrap the body at 72 characters
   - Use the body to explain what and why vs. how

#### Commit messages

Without wanting to sound pedantic, but a good commit history is important! It's not uncommon to
review code commit by commit. If the commits are not to large, self-contained, in a logical order
and have good commit messages, reviewing is a hell lot easier. Hence, we'd advice everyone to pay
attention to it. But also don't worry to much, if you feel unsure what to do, again get in touch!

As mentioned above, we follow the Conventional Commit style. I.e. the commit message header is
structured as `<type>(<scope>): <short summary>`. The type and summary are mandatory, the scope is
optional. Following the [Angular commit format](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit),
the type must be one of the following: `build | ci | docs | feat | fix | perf | refactor | test`.
For short descriptions of the different types see the above link. The summary should be in present
tense, not capitalied and without period at the end. The scope should be an issue number if present.

Some further guidelines for good commit message can be taken from [this
blogpost](https://chris.beams.io/posts/git-commit/):
   - Separate subject from body with a blank line
   - Limit the subject line to 50 characters
   - Use the imperative mood in the subject line
   - Wrap the body at 72 characters
   - Use the body to explain what and why vs. how

Further [this blogpost](https://gist.github.com/brianclements/841ea7bffdb01346392c) is a recommended
read, if you feel unsure about git in general.

## Merge request checklist

Before pushing and finishing your merge request consider ask yourself the following questions:
- Are there tests for my new/changed code?
- Do all tests from the complete testsuite still pass?
- If there are new features, or behaviour, is the documentation updated accordingly?
- Is the code properly linted?
- Are commits grouped logically and squshed if appropriate?
- Are the commit messages clear and emphesize what and why?

If any of the question is answered with no, clean that up, or clearly state why not. This helps the
person(s) reviewing a lot and will result in faster merges.

## Development details

This section should help you understand certain aspects of the workflow a little better, and might
help improve your workflow.

### Dependency Management

As of December 2020, we switched to [CPM](https://github.com/TheLartians/CPM.cmake) for our dependency management.
Hopefully, you don't really have to know and it doesn't get the way. See the [root CMakeLists.txt file](./CMakeLists.txt),
as an example for our usage.

It is recommended to set `CPM_SOURCE_CACHE` (see [here](https://github.com/TheLartians/CPM.cmake#cpm_source_cache) for
more info). It's an environment variable, that will save all dependencies outside of the build directory and -
for all projects using CPM - only one version of the dependency. This way no re-downloading is necessary.
Set it in your e.g. `.bashrc`.

### Testing
You can run the elsa unit tests by running `ctest` in the build folder. To specify which tests run,
filter with `ctest -R regular_expression`.

We use a testing style similar described in [Behaviour-Driven
Development](https://github.com/onqtam/doctest/blob/master/doc/markdown/testcases.md#bdd-style-test-cases) (BDD). Please
follow this style when adding new tests. However, instead of using `SCENARIO` use `TEST_CASE` with the name of the
class under test at the beginning of the test name. Also be sure to add the tests to the test suite associated
to the module of the test.

We're currently relying on [doctest](https://github.com/onqtam/doctest/) as our testing framework, when
using assertion macros, please try to use the
[binary and unary asserts](https://github.com/onqtam/doctest/blob/master/doc/markdown/assertions.md#binary-and-unary-asserts)
as much as possible.


### Style Guide
We use the tool `clang-format` to autoformat our code with the [given style
file](.clang-format). Please make sure that your code is formatted accordingly, otherwise the CI
will fail in the first stage. You can either execute
```
find elsa/ -name '*.h' -or -name '*.hpp' -or -name '*.cpp' | xargs clang-format -i -style=file $1
find examples/ -name '*.h' -or -name '*.hpp' -or -name '*.cpp' | xargs clang-format -i -style=file $1
find -name '*.h' -or -name '*.hpp' -or -name '*.cpp' | xargs clang-format -i -style=file $1
```
in the root folder, setup a git hook or integrate `clang-format` into your IDE. Note that we
currently use version 14.0.0, different versions might produce errors.

### Linting
We use `clang-tidy` with the enabled checks specified in [the configuration file](.clang-tidy). Note
that currently all `readability-*` checks have to pass, otherwise the CI will fail. We encourage
developers to check their code with `clang-tidy` and remove all warnings if applicable.

#### CMake

We use [cmakelang](https://cmake-format.readthedocs.io/en/latest/index.html) to enforce
certain style guide and reduce the changes of error in our CMake code, please check the guide to install it.

Currently, only the `cmake-lint` is used, but sooner rather than later, we'll also start
using `cmake-format` of the same package.

Please check the link above on how to install the package.

### Code Coverage
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

### pre-commit

There is also a basic `.pre-commit-config.yaml` file to install pre-commit hooks using
[pre-commit](https://pre-commit.com/). You are highly encouraged to install the pre-commits
with `pre-commit install -t pre-commit -t commit-msg` such that they are run before each commit.

None of the commit hooks will change anything in your commit, they mearly check and error if
something is wrong.

### Building the Documentation
The [elsa documentation](https://ciip.in.tum.de/elsadocs/) is automatically built and deployed through the CI for each commit to master.
To build it locally the following packages are required: `sphinx doxygen` which should be available in
most major linux distributions or via pip. Additionally, the following sphinx extensions need to be installed via pip:
`sphinx-rtd-theme sphinxcontrib-katex m2r2 breathe`.
Then simply build the documentation using ninja
```
mkdir -p build
cd build
cmake .. -GNinja
ninja docs
```
The docs should then be available at `build/docs/sphinx`.
