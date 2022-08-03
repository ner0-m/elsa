# Credits to Matt Godbold for the idea of this!

THIS_DIR = $(shell pwd)

PHONY: help
help: # with thanks to Ben Rady
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

CMAKE ?= $(shell which cmake || echo .cmake-not-found)
FZF ?= $(shell which fzf)

AG ?= $(shell which ag)
ENTR ?= $(shell which entr)
CT ?= $(shell which ct)

# Choose clang over gcc if present
ifeq ($(origin CC),default)
  CC = $(shell (type clang > /dev/null 2>&1 && echo "clang") || \
	               (type gcc > /dev/null 2>&1 && echo "gcc") || \
		       echo .c-compiler-not-found)
endif
ifeq ($(origin CXX),default)
  CXX = $(shell (type clang++ > /dev/null 2>&1 && echo "clang++") || \
	               (type g++ > /dev/null 2>&1 && echo "g++") || \
		       echo .cxx-compiler-not-found)
endif

BUILD_TYPE?=RelWithDebInfo
BUILD_OPTIONS=

# Choose build directory depending on compiler and build type
ifeq ($(CXX),clang++)
  BUILD_ROOT:=$(CURDIR)/build/$(BUILD_TYPE)/clang
endif

ifeq ($(CXX),g++)
  BUILD_ROOT:=$(CURDIR)/build/$(BUILD_TYPE)/gcc
endif

# Use ninja if present
ifeq ($(shell which ninja),)
  CMAKE_GENERATOR_FLAGS?=
else
  CMAKE_GENERATOR_FLAGS?=-GNinja

  ifeq ($(CXX),clang++)
    CXXFLAGS += -fcolor-diagnostics
  endif

  ifeq ($(CXX),g++)
    CXXFLAGS += -fdiagnostics-color=always
  endif
endif


USE_CUDA?=y
ifeq ($(USE_CUDA),y)
# Check for NVCC
  ifneq ($(shell which nvcc),)
    CUDA_OPTIONS?=-DELSA_BUILD_CUDA_PROJECTORS=ON
  else
    CUDA_OPTIONS?=
  endif
else
  BUILD_OPTIONS+=-DELSA_BUILD_CUDA_PROJECTORS=OFF
endif

USE_DNNL?=y
ifeq ($(USE_DNNL),y)
  BUILD_OPTIONS+=-DELSA_BUILD_ML_DNNL=ON
else
  BUILD_OPTIONS+=-DELSA_BUILD_ML_DNNL=OFF
endif

USE_CUDNN?=n
ifeq ($(USE_CUDNN),y)
  BUILD_OPTIONS+=-DELSA_BUILD_ML_CUDNN=ON
else
  BUILD_OPTIONS+=-DELSA_BUILD_ML_CUDNN=OFF
endif

GENERATE_PYBINDS?=n
ifeq ($(GENERATE_PYBINDS),y)
  BUILD_OPTIONS+=-DELSA_BUILD_PYTHON_BINDINGS=ON
else
  BUILD_OPTIONS+=-DELSA_BUILD_PYTHON_BINDINGS=OFF
endif

# all targets defined by cmake, but each on of the line starts with xxx, so that we can split it later as newline gets lost
CMAKE_TARGETS = $(shell cmake --build $(BUILD_ROOT) --target help | sed s/$$/xxx/)

# Turn all arguments after build, into targets to build (default) all
ifeq (build,$(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "run"
  BUILD_TARGETS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval $(BUILD_TARGETS):;@:)
endif

ifeq ($(BUILD_TARGETS),)
  BUILD_TARGETS=all
endif

selected_test ?=

ifeq (test,$(firstword $(MAKECMDGOALS)))
  selected_test = y
else ifeq (watch, $(firstword $(MAKECMDGOALS)))
  selected_test = y
endif

# Turn all arguments after test, into targets to build (default) all
ifeq ($(selected_test),y)
  # use the rest as arguments for "run"
  SELECTED_TEST_TARGET := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval $(SELECTED_TEST_TARGET):;@:)

  RUN_TEST_TARGET ?=

  # replace the xxx, remove empty lines, only select targets starting with test_, remove targets with '/' then cut everything away after colon, see if grep can find the target and remove whitespaces with xargs
  FILTERED_TEST_TARGETS = $(shell echo -n $(CMAKE_TARGETS)  | sed 's/xxx/\n/g' | grep "\S" | grep test\_ | grep -v '/' | cut -d: -f1 | xargs)

  ifeq ($(FZF),)
    # If no argument is passed, we can't handle it without fzf
    ifeq ($(SELECTED_TEST_TARGET),)
      $(error Please specify a test target (or install fzf to interactivly select one))
    endif

    # Check if grep can find it
    RUN_TEST_TARGET = $(shell echo -n $(FILTERED_TEST_TARGETS) | tr ' ' '\n' | grep -i $(SELECTED_TEST_TARGET))

    $(info $(SELECTED_TEST_TARGET))
    ifeq ($(RUN_TEST_TARGET),)
      $(error Can not find unique a unique test target with argument $(SELECTED_TEST_TARGET))
    endif
  else
    # use fzf to (if necessary interactivly) find a target
    ifneq ($(SELECTED_TEST_TARGET),)
      RUN_TEST_TARGET := $(shell echo -n $(FILTERED_TEST_TARGETS) | tr ' ' '\n' | $(FZF) -q $(SELECTED_TEST_TARGET) -1 -0)
    else
      RUN_TEST_TARGET := $(shell echo -n $(FILTERED_TEST_TARGETS) | tr ' ' '\n' | $(FZF) -1 -0)
    endif
  endif

  TEST_EXECUTABLE = $(shell find $(BUILD_ROOT)/bin/tests -iname $(RUN_TEST_TARGET))
endif

.%-not-found:
	@echo "-----------------------"
	@echo "elsa needs $(@:.%-not-found=%) to build. Please install it "
	@echo "-----------------------"
	@exit 1

.PHONY: configure
configure: $(BUILD_ROOT)/CMakeCache.txt ## Configure elsa

.PHONY: build tests test watch

build: $(BUILD_ROOT)/CMakeCache.txt ## Build provided targets (default: all)
	$(CMAKE) --build $(BUILD_ROOT) --target $(BUILD_TARGETS)
	$(shell notify-send "elsa "Building target \"$(BUILD_TARGETS)\" has finished")

tests: build ## build and run all tests
	$(CMAKE) --build $(BUILD_ROOT) -- tests

test: $(BUILD_ROOT)/CMakeCache.txt ## build and run individual test (e.g. make test mytest)
	CXXFLAGS=$(CXXFLAGS) $(CMAKE) --build $(BUILD_ROOT) --target $(RUN_TEST_TARGET)
	$(shell find $(BUILD_ROOT)/bin/tests -iname $(RUN_TEST_TARGET)) && notify-send "elsa" "Test $(RUN_TEST_TARGET) passed all tests" || notify-send --urgency=critical "elsa" "Test $(RUN_TEST_TARGET) failed"

watch: $(BUILD_ROOT)/CMakeCache.txt ## build and run individual test continuously on code changes (e.g. make watch mytest)
ifneq ($(CT),)
	$(AG) -l "cpp|h|hpp|cu|cuh" | $(ENTR) -s "make CC=$(CC) CXX=$(CXX) GENERATE_PYBINDS=$(GENERATE_PYBINDS) USE_DNNL=$(USE_DNNL) USE_CUDNN=$(USE_CUDNN) USE_CUDA=$(USE_CUDA) BUILD_TYPE=$(BUILD_TYPE) test $(RUN_TEST_TARGET) | $(CT) && notify-send \"elsa\" \"finished building\" || notify-send \"elsa\" \"build failed\""
else
	$(AG) -l "cpp|h|hpp|cu|cuh" | $(ENTR) -s "make CC=$(CC) CXX=$(CXX) GENERATE_PYBINDS=$(GENERATE_PYBINDS) USE_DNNL=$(USE_DNNL) USE_CUDNN=$(USE_CUDNN) USE_CUDA=$(USE_CUDA) BUILD_TYPE=$(BUILD_TYPE) test $(RUN_TEST_TARGET) && notify-send \"elsa\" \"finished building\" || notify-send \"elsa\" \"build failed\""
endif


ifneq ($(FZF),)
select-targets: configure ## select one of the available targets to build
	$(eval MYTEST = $(shell echo -n $(CMAKE_TARGETS) ' ' | sed 's/xxx/\n/g' | grep -v '/' | $(FZF)  | cut -d: -f1 | xargs))
	CXXFLAGS=$(CXXFLAGS) $(CMAKE) --build $(BUILD_ROOT) --target $(MYTEST)

select-test: configure ## select one of the available tests to build and run
	$(eval MYTEST = $(shell echo -n $(CMAKE_TARGETS) ' ' | sed 's/xxx/\n/g' | grep test\_ | grep -v '/'  | $(FZF) | cut -d: -f1 | xargs))
	CXXFLAGS=$(CXXFLAGS) $(CMAKE) --build $(BUILD_ROOT) --target $(MYTEST)
	$(shell find $(BUILD_ROOT)/bin/tests -iname $(MYTEST)) && notify-send "elsa" "Running $(MYTEST) passed" || notify-send --urgency=critical "elsa" "Running $(MYTEST) failed"
endif

$(BUILD_ROOT)/CMakeCache.txt:
	@mkdir -p $(BUILD_ROOT)
	CC=$(CC) CXX=$(CXX) $(CMAKE) -S . -B $(BUILD_ROOT) $(CMAKE_GENERATOR_FLAGS) \
	    -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CUDA_OPTIONS) $(BUILD_OPTIONS)

.PHONY: clean
clean:  ## Remove build artifacts
	$(CMAKE) --build $(BUILD_ROOT) --target clean

.PHONY: distclean
distclean:  ## Clean up everything
	rm -rf build/ $(shell find . -name '*.egg-info') _skbuild
