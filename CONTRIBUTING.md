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

## Testing
You can run the elsa unit tests by running `ctest` in the build folder. To specify which tests run,
filter with `ctest -R regular_expression`.

We use a testing style described as [Behaviour-Driven
Development](https://github.com/catchorg/Catch2/blob/master/docs/tutorial.md#bdd-style) (BDD). Please
follow this style when adding new tests.

## Style Guide
We use the tool `clang-format` to autoformat our code with the [given style
file](.clang-format). Please make sure that your code is formatted accordingly, otherwise the CI
will fail in the first stage. You can either execute
```
find elsa/ -name '*.h' -or -name '*.hpp' -or -name '*.cpp' | xargs clang-format-8 -i -style=file $1
```
in the root folder, setup a git hook or integrate `clang-format` into your IDE. Note that we
currently use version 8.0.0, different versions might produce errors.

## Linting
We use `clang-tidy` with the enabled checks specified in [the configuration file](.clang-tidy). Note
that currently all `readability-*` checks have to pass, otherwise the CI will fail. We encourage
developers to check their code with `clang-tidy` and remove all warnings if applicable.

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
local results to [the latest master coverage results](https://ip.campar.in.tum.de/elsacoverage/).
