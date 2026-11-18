# Cross-platform Makefile for compiling the DTFE code on Mac and Linux systems
#
# SUPPORTED ARCHITECTURES:
# ========================
# - x86_64 (Intel/AMD 64-bit) on both macOS and Linux
# - ARM64 (aarch64/Apple Silicon) on both macOS and Linux
# The Makefile automatically detects the architecture and sets appropriate paths.
#
# USAGE:
# ======
# This Makefile automatically detects your operating system and sets appropriate
# defaults for library paths and compiler settings.
#
# macOS (Homebrew):
#   Default paths assume Homebrew installation. Just run 'make DTFE'
#
# Linux:
#   Uses standard system paths (/usr). Install development packages:
#   sudo apt-get install libgsl-dev libboost-dev libboost-thread-dev libboost-filesystem-dev \
#       libboost-program-options-dev libcgal-dev libmpfr-dev libhdf5-dev libgmp-dev libtbb-dev libfftw3-dev
#   or equivalent for your distribution, then run 'make DTFE'
#
# CUSTOMIZATION:
# ==============
# Override library paths by setting environment variables or make variables:
#   make DTFE GSL_PATH_OVERRIDE=/custom/path/to/gsl
#   make DTFE BOOST_PATH_OVERRIDE=/custom/path/to/boost
#   make DTFE CC_OVERRIDE=g++-11
#
# Or set them as environment variables:
#   export GSL_PATH_OVERRIDE=/custom/path/to/gsl
#   make DTFE
#
# Build in debug mode with sanitizers:
#   make DTFE BUILD_MODE=debug
#   make library BUILD_MODE=debug
#
# Add or override compiler flags:
#   make DTFE EXTRA_FLAGS="-Wno-unused -march=native"
#
# Note: Library headers are included with -isystem to suppress warnings from
# external code you cannot modify. Your own code warnings will still be shown.
#
# TARGETS:
# ========
# make DTFE                      - Build standard DTFE executable (Eulerian triangulation)
# make PS-DTFE                   - Build PS-DTFE executable (phase-space, Lagrangian triangulation)
# make DTFE BUILD_MODE=debug     - Build with debug symbols, no optimization, and sanitizers
# make library                   - Build the shared library (libDTFE.so/.dylib)
# make library BUILD_MODE=debug  - Build library in debug mode
# make clean                     - Clean object files and executables
# make check                     - Quick test of the built ./DTFE and ./PS-DTFE (three suites; no simulation data)
#

# Detect operating system
UNAME_S := $(shell uname -s)

# Set platform-specific variables
ifeq ($(UNAME_S),Darwin)  # macOS
    PLATFORM := macos
    SHARED_EXT := .dylib
    EXE_EXT :=
    OBJ_EXT := .o
else ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
    SHARED_EXT := .so
    EXE_EXT :=
    OBJ_EXT := .o
else
    $(error Unsupported operating system: $(UNAME_S). Only macOS and Linux are supported.)
endif

# Platform-specific library paths and compiler settings
ifeq ($(PLATFORM),macos)
    # macOS with Homebrew - detect architecture for correct paths
    ARCH := $(shell uname -m)
    ifeq ($(ARCH),arm64)
        # Apple Silicon (ARM64) - Homebrew installs to /opt/homebrew
        BREW_PREFIX = /opt/homebrew
    else
        # Intel (x86_64) - Homebrew installs to /usr/local
        BREW_PREFIX = /usr/local
    endif

    GSL_PATH   = $(BREW_PREFIX)/opt/gsl
    BOOST_PATH = $(BREW_PREFIX)/opt/boost
    CGAL_PATH  = $(BREW_PREFIX)/opt/cgal
    MPFR_PATH  = $(BREW_PREFIX)/opt/mpfr
    HDF5_PATH  = $(BREW_PREFIX)/opt/hdf5
    GMP_PATH   = $(BREW_PREFIX)/opt/gmp
    FFTW_PATH  = $(BREW_PREFIX)/opt/fftw

    # Try different compiler locations
    CC := $(shell which $(BREW_PREFIX)/opt/llvm/bin/clang++ 2>/dev/null || which clang++ 2>/dev/null || which g++ 2>/dev/null || echo "clang++")
else ifeq ($(PLATFORM),linux)
    # Linux - try to auto-detect common package manager installations
    GSL_PATH   = $(shell pkg-config --variable=prefix gsl 2>/dev/null || echo "/usr")
    BOOST_PATH = /usr
    CGAL_PATH  = /usr
    MPFR_PATH  = /usr
    HDF5_PATH  = /usr
    GMP_PATH = /usr
    FFTW_PATH = /usr
    # Debian/Ubuntu put the serial HDF5 headers and libraries in a sub-directory that is NOT
    # on the default search path (libhdf5-dev installs to /usr/include/hdf5/serial), so a
    # plain -I/usr/include build dies on '#include <H5Cpp.h>'. Pick the sub-directory up
    # automatically when it exists (harmless elsewhere).
    HDF5_EXTRA_INC := $(firstword $(wildcard /usr/include/hdf5/serial))
    HDF5_EXTRA_LIB := $(firstword $(wildcard /usr/lib/*/hdf5/serial))
    CC := $(shell which g++ 2>/dev/null || which clang++ 2>/dev/null || echo "g++")
endif

# Common utilities
MKDIR_P = mkdir -p
RM_RF = rm -rf

# Allow user override of library paths
GSL_PATH   := $(or $(GSL_PATH_OVERRIDE),$(GSL_PATH))
BOOST_PATH := $(or $(BOOST_PATH_OVERRIDE),$(BOOST_PATH))
CGAL_PATH  := $(or $(CGAL_PATH_OVERRIDE),$(CGAL_PATH))
MPFR_PATH  := $(or $(MPFR_PATH_OVERRIDE),$(MPFR_PATH))
HDF5_PATH  := $(or $(HDF5_PATH_OVERRIDE),$(HDF5_PATH))
GMP_PATH   := $(or $(GMP_PATH_OVERRIDE),$(GMP_PATH))
FFTW_PATH  := $(or $(FFTW_PATH_OVERRIDE),$(FFTW_PATH))
CC         := $(or $(CC_OVERRIDE),$(CC))


# paths to where to put the object files and the executables files. If you build the DTFE library than you also need to specify the directory where to put the library and the directory where to copy the header files needed by the library (choose an empty directory for the header files).
# o / o_ps hold the single-precision pair's objects, o_d / o_ps_d the DOUBLE=1 pair's (OBJ_SUFFIX
# is set in the precision block below; make expands it when the directory names are used)
OBJ_DIR = ./o$(OBJ_SUFFIX)
OBJ_DIR_PS = ./o_ps$(OBJ_SUFFIX)
BIN_DIR = ./
LIB_DIR = ./
INC_DIR = ./DTFE_include

############################# Choose the compiler directives ##################################

############################# Overall options ##################################
# Common options shared by both standard DTFE and PS-DTFE builds
OPTIONS_COMMON =
#------------------------ set the number of spatial dimensions (2 or 3 dimensions)
# Override per build with 'make DTFE DIM=2'. The dimension is part of the object-directory
# stamp below, so switching it wipes the objects exactly like a backend switch does -- mixing
# 2D and 3D objects would otherwise link into a binary that disagrees with itself about NO_DIM.
DIM ?= 3
ifeq ($(filter $(DIM),2 3),)
$(error DIM must be 2 or 3 (got '$(DIM)'))
endif
OPTIONS_COMMON += -DNO_DIM=$(DIM)
#------------------------ floating-point precision: 'make PS-DTFE DOUBLE=1' builds every position,
# density, field and accumulator in double (Real = double; outputs become float64), end to end
# including the HDF5 reads (TNG stores Coordinates in float64, which the default float build
# rounds on read). Costs ~2x the memory. The double pair is a SEPARATE set of binaries,
# DTFE-double / PS-DTFE-double, built from its own object directories o_d / o_ps_d, so both
# precisions coexist (the run scripts pick a pair with DTFE_PRECISION=double, the GUI with its
# precision setting). METAL=1 is allowed: the GPU deposit kernels themselves are single
# precision, so a double build hands the GPU float copies of its tetrahedra and adds the float
# sums into its double grids -- everything else (reading, triangulation, densities, point
# evaluation, the CPU deposits) stays double.
DOUBLE ?= 0
ifeq ($(DOUBLE),1)
OPTIONS_COMMON += -DDOUBLE
PRECISION = double
PREC_BUILD_ARG = DOUBLE=1
BIN_SUFFIX = -double
OBJ_SUFFIX = _d
else ifeq ($(DOUBLE),0)
PRECISION = float
PREC_BUILD_ARG =
BIN_SUFFIX =
OBJ_SUFFIX =
else
$(error DOUBLE must be 0 or 1 (got '$(DOUBLE)'))
endif
# DIM=2 builds are a set of their own as well: DTFE-2d / PS-DTFE-2d (-2d-double with DOUBLE=1), objects in
# o_2d / o_ps_2d (_2d_d), so the 2D and 3D binaries coexist instead of a 2D build replacing ./DTFE.
ifeq ($(DIM),2)
BIN_SUFFIX := -2d$(BIN_SUFFIX)
OBJ_SUFFIX := _2d$(OBJ_SUFFIX)
DIM_BUILD_ARG = DIM=2
else
DIM_BUILD_ARG =
endif

############################# Quantities to be computed ##################################
#------------------------ set which quantities can be computed (can save memory by leaving some out)
# Comment this line if you don't need to compute velocity and velocity related components
OPTIONS_COMMON += -DVELOCITY
# Comment this line if you don't need to interpolate additional fields stored in the scalar variable
OPTIONS_COMMON += -DSCALAR
# number of components of the scalar variable
OPTIONS_COMMON += -DNO_SCALARS=1

############################# Input and output operations default settings ##################################
#------------------------ set which are the default input and output functions for doing data io
# default function to read the input data (101-binary gadget snapshot in one file or several, 105-HDF5 gadget file, 111-text file, ... see documentation for more options). The input file type can be set during runtime using the option '--input-format'. This makefile option only sets a default input file in the case none is given via the program options.
OPTIONS_COMMON += -DINPUT_FILE_DEFAULT=105
# default value for the units of the input data (value=what is 1 Mpc in the units of the data - in this example the data is in kpc). You can change this also during runtime using the program option '--mpc-unit'.
OPTIONS_COMMON += -DMPC_UNIT=1000
# default function to write the output data (101-binary file, 111-text file, ... see documentation for more options). The output file type can be set during runtime using the option '--output-format'. This makefile option only sets a default output file in the case none is given via the program options.
OPTIONS_COMMON += -DOUTPUT_FILE_DEFAULT=101
#101 for binary file, 100 my density file

############################# additional compiler options ##################################
# enable this option if to use OpenMP (share the workload between CPU cores sharing the same RAM)
OPTIONS_COMMON += -DOPEN_MP
# enable to check if the padding gives a complete Delaunay Tesselation of the region of interest
OPTIONS_COMMON += -DTEST_PADDING
# keeps the '--redshiftSpace' option registered; the shift itself is NOT implemented in this version (the option
# refuses the run with a clear error rather than silently skipping the shift)
OPTIONS_COMMON += -DREDSHIFT_SPACE

# Standard DTFE build: no PHASE_SPACE flag (triangulates in Eulerian space, standard density)
OPTIONS = $(OPTIONS_COMMON)

# PS-DTFE build: includes PHASE_SPACE flag (triangulates in Lagrangian space, multi-stream regions)
OPTIONS_PS = $(OPTIONS_COMMON) -DPHASE_SPACE

# ---- GPU backend (optional): METAL=1 on macOS/Apple Silicon, the only one. It defines DTFE_GPU / PS_GPU (the
# guards used by the callers) plus GPU_BACKEND_NAME, links the Metal host object and frameworks, and the binary is
# run with '--gpu' (both binaries). Without it the build is CPU-only and the flag falls
# back to the CPU with a warning. Metal embeds + runtime-compiles the kernels in metal/*.metal. The CUDA/HIP port
# and its CPU emulation (CUDA=1, HIP=1, CUDAEMU=1) are kept on the branch archive/cuda-hip-2026-11-08, not here:
# an old command line naming them stops at once rather than building a CPU-only binary under that name.
ifneq ($(filter 1,$(CUDA) $(HIP) $(CUDAEMU)),)
$(error CUDA=1, HIP=1 and CUDAEMU=1 are gone: this tree is Metal-only (METAL=1); the CUDA/HIP port is on the branch archive/cuda-hip-2026-11-08)
endif
GPU_MODE = off
ifeq ($(METAL),1)
GPU_MODE = metal
endif

DTFE_GPU_OBJS =
DTFE_GPU_L_OBJS =
PS_GPU_OBJS =
GPU_LIBS =
GPU_BUILD_ARG =

# The GPU kernels deposit tetrahedra into 3D cells: a DIM=2 build is CPU-only whatever backend is
# asked for (its --gpu then falls back with the usual warning).
ifeq ($(DIM),2)
ifneq ($(GPU_MODE),off)
$(info >> DIM=2: the GPU kernels are 3D only; this 2D build is CPU-only)
GPU_MODE = off
endif
endif
ifeq ($(GPU_MODE),metal)
OPTIONS    += -DDTFE_GPU -DGPU_BACKEND_NAME=\"Metal\"
OPTIONS_PS += -DPS_GPU -DGPU_BACKEND_NAME=\"Metal\"
DTFE_GPU_OBJS = $(OBJ_DIR)/dtfe_metal_host$(OBJ_EXT)
DTFE_GPU_L_OBJS = $(OBJ_DIR)/dtfe_metal_host_l$(OBJ_EXT)
PS_GPU_OBJS = $(OBJ_DIR_PS)/ps_metal_host$(OBJ_EXT)
GPU_LIBS = -framework Metal -framework Foundation
GPU_BUILD_ARG = METAL=1
endif

# A build must never mix GPU-mode and plain objects (an incremental 'make PS-DTFE' after a
# METAL=1 build would silently produce a binary whose components disagree about PS_GPU).
# The mode is stamped in the object dir; when it changes, all its objects are wiped first. '.build_mode' records the make argument ("METAL=1",
# empty for CPU-only) so tests can rebuild in the same mode: make PS-DTFE $$(cat o_ps/.build_mode)
.PHONY: ps_gpu_mode_check
ps_gpu_mode_check:
	@$(MKDIR_P) $(OBJ_DIR_PS)
	@if [ ! -f $(OBJ_DIR_PS)/.gpu_mode_$(GPU_MODE) ] || [ ! -f $(OBJ_DIR_PS)/.dim_$(DIM) ] || [ ! -f $(OBJ_DIR_PS)/.prec_$(PRECISION) ] || [ ! -f $(OBJ_DIR_PS)/.tbb_$(TBB_PS) ] || [ ! -f $(OBJ_DIR_PS)/.deflate_$(HAVE_LIBDEFLATE) ] || [ ! -f $(OBJ_DIR_PS)/$(FLAGS_STAMP) ] || { [ -n "$(FPC_STAMP)" ] && [ ! -f $(OBJ_DIR_PS)/$(FPC_STAMP) ]; }; then \
		echo ">> PS build mode is now '$(GPU_MODE)', $(DIM)D, $(PRECISION), $(BUILD_MODE) (flags $(FLAGS_SUM)), TBB=$(TBB_PS), libdeflate=$(HAVE_LIBDEFLATE)$(if $(FPC_STAMP),$(comma) no FMA contraction); wiping $(OBJ_DIR_PS) to avoid mixed objects"; \
		rm -f $(OBJ_DIR_PS)/*$(OBJ_EXT) $(OBJ_DIR_PS)/*.d $(OBJ_DIR_PS)/ps_deposit_msl.h $(OBJ_DIR_PS)/.gpu_mode_* $(OBJ_DIR_PS)/.dim_* $(OBJ_DIR_PS)/.prec_* $(OBJ_DIR_PS)/.tbb_* $(OBJ_DIR_PS)/.deflate_* $(OBJ_DIR_PS)/.fpc_* $(OBJ_DIR_PS)/.flags_* $(OBJ_DIR_PS)/.metal_mode_* $(OBJ_DIR_PS)/.build_mode; \
		touch $(OBJ_DIR_PS)/.gpu_mode_$(GPU_MODE) $(OBJ_DIR_PS)/.dim_$(DIM) $(OBJ_DIR_PS)/.prec_$(PRECISION) $(OBJ_DIR_PS)/.tbb_$(TBB_PS) $(OBJ_DIR_PS)/.deflate_$(HAVE_LIBDEFLATE) $(OBJ_DIR_PS)/$(FLAGS_STAMP) $(if $(FPC_STAMP),$(OBJ_DIR_PS)/$(FPC_STAMP)); \
		printf '%s' "$(strip $(GPU_BUILD_ARG) $(PREC_BUILD_ARG) $(TBB_BUILD_ARG) $(DIM_BUILD_ARG))" > $(OBJ_DIR_PS)/.build_mode; \
	fi

# Same guard for the standard build: the mode is stamped in o/ (which also holds the *_l library
# objects); when it changes, all standard objects are wiped first.
.PHONY: dtfe_gpu_mode_check
dtfe_gpu_mode_check:
	@$(MKDIR_P) $(OBJ_DIR)
	@if [ ! -f $(OBJ_DIR)/.gpu_mode_$(GPU_MODE) ] || [ ! -f $(OBJ_DIR)/.dim_$(DIM) ] || [ ! -f $(OBJ_DIR)/.prec_$(PRECISION) ] || [ ! -f $(OBJ_DIR)/.tbb_$(TBB_STD) ] || [ ! -f $(OBJ_DIR)/.deflate_$(HAVE_LIBDEFLATE) ] || [ ! -f $(OBJ_DIR)/$(FLAGS_STAMP) ] || { [ -n "$(FPC_STAMP)" ] && [ ! -f $(OBJ_DIR)/$(FPC_STAMP) ]; }; then \
		echo ">> DTFE build mode is now '$(GPU_MODE)', $(DIM)D, $(PRECISION), $(BUILD_MODE) (flags $(FLAGS_SUM)), libdeflate=$(HAVE_LIBDEFLATE)$(if $(FPC_STAMP),$(comma) no FMA contraction); wiping $(OBJ_DIR) to avoid mixed objects"; \
		rm -f $(OBJ_DIR)/*$(OBJ_EXT) $(OBJ_DIR)/*.d $(OBJ_DIR)/dtfe_deposit_msl.h $(OBJ_DIR)/.gpu_mode_* $(OBJ_DIR)/.dim_* $(OBJ_DIR)/.prec_* $(OBJ_DIR)/.tbb_* $(OBJ_DIR)/.deflate_* $(OBJ_DIR)/.fpc_* $(OBJ_DIR)/.flags_* $(OBJ_DIR)/.metal_mode_* $(OBJ_DIR)/.build_mode; \
		touch $(OBJ_DIR)/.gpu_mode_$(GPU_MODE) $(OBJ_DIR)/.dim_$(DIM) $(OBJ_DIR)/.prec_$(PRECISION) $(OBJ_DIR)/.tbb_$(TBB_STD) $(OBJ_DIR)/.deflate_$(HAVE_LIBDEFLATE) $(OBJ_DIR)/$(FLAGS_STAMP) $(if $(FPC_STAMP),$(OBJ_DIR)/$(FPC_STAMP)); \
		printf '%s' "$(strip $(GPU_BUILD_ARG) $(PREC_BUILD_ARG) $(DIM_BUILD_ARG))" > $(OBJ_DIR)/.build_mode; \
	fi

#------------------------ options usefull when using DTFE as a library
# uncomment the line to get access to a function that returns the Delaunay triangulation of the point set
OPTIONS_COMMON += -DTRIANGULATION


OPTIONS_COMMON += -DBOOST_TIMER_ENABLE_DEPRECATED
OPTIONS_COMMON += -DBOOST_ALLOW_DEPRECATED_HEADERS




###############  DO NOT MODIFY BELOW THIS LINE  ###########################
# do not modify below this line
SRC = ./src
# third-party headers come in by -isystem: their own warnings (Boost's under recent LLVM) stay out of
# the build log, while every warning in this code is still reported. sysinc gives a folder its -isystem,
# except the compiler's own system folders (a Linux package's /usr/include): an explicit -isystem moves
# them ahead of libstdc++'s headers, whose '#include_next <math.h>' then fails under GCC, and they are
# system folders (no warnings) already
sysinc = $(if $(filter /usr/include /usr/local/include,$(abspath $(strip $(1)))),,-isystem $(strip $(1)))
INCLUDES =
LIBRARIES =

# Library path setup
ifneq ($(strip $(GSL_PATH)),)
    INCLUDES += $(call sysinc,$(strip $(GSL_PATH))/include)
    LIBRARIES += -L$(strip $(GSL_PATH))/lib
endif
ifneq ($(strip $(BOOST_PATH)),)
    INCLUDES += $(call sysinc,$(strip $(BOOST_PATH))/include)
    LIBRARIES += -L$(strip $(BOOST_PATH))/lib
endif
ifneq ($(strip $(CGAL_PATH)),)
    INCLUDES += $(call sysinc,$(strip $(CGAL_PATH))/include)
    LIBRARIES += -L$(strip $(CGAL_PATH))/lib
endif
ifneq ($(strip $(GMP_PATH)),)
    INCLUDES += $(call sysinc,$(strip $(GMP_PATH))/include)
    LIBRARIES += -L$(strip $(GMP_PATH))/lib
endif
ifneq ($(strip $(MPFR_PATH)),)
    INCLUDES += $(call sysinc,$(strip $(MPFR_PATH))/include)
    LIBRARIES += -L$(strip $(MPFR_PATH))/lib
endif
ifneq ($(strip $(HDF5_PATH)),)
    INCLUDES += $(call sysinc,$(strip $(HDF5_PATH))/include)
    LIBRARIES += -L$(strip $(HDF5_PATH))/lib   # -lhdf5/-lhdf5_cpp added once via HDF5_LIBS (avoid duplicate-library linker warning)
    ifneq ($(strip $(HDF5_EXTRA_INC)),)
        INCLUDES += $(call sysinc,$(HDF5_EXTRA_INC))
    endif
    ifneq ($(strip $(HDF5_EXTRA_LIB)),)
        LIBRARIES += -L$(strip $(HDF5_EXTRA_LIB))
    endif
    OPTIONS += -DHDF5
    OPTIONS_PS += -DHDF5
endif
ifneq ($(strip $(FFTW_PATH)),)
    INCLUDES += $(call sysinc,$(strip $(FFTW_PATH))/include)
    LIBRARIES += -L$(strip $(FFTW_PATH))/lib
endif

# Compiler flags (same for both platforms)
# Build mode can be set with: make DTFE BUILD_MODE=debug
BUILD_MODE ?= release

ifeq ($(BUILD_MODE),debug)
    # Debug build: no optimization, with debug symbols and sanitizers
    BASE_CFLAGS = -O0 -g3 -DDEBUG $(OPTIONS)
    BASE_CFLAGS_PS = -O0 -g3 -DDEBUG $(OPTIONS_PS)
    # Add sanitizers for debug builds (catch memory errors, undefined behavior, etc.)
    SANITIZER_FLAGS = -fsanitize=address -fsanitize=undefined -fsanitize=leak
    DEBUG_FLAGS = $(SANITIZER_FLAGS) -fno-omit-frame-pointer
else
    # Release build: full optimization
    BASE_CFLAGS = -O3 -DNDEBUG $(OPTIONS)
    BASE_CFLAGS_PS = -O3 -DNDEBUG $(OPTIONS_PS)
    DEBUG_FLAGS =
endif

# macOS SDK pinning: after an OS / Command-Line-Tools update the Homebrew clang
# can derive a non-existent SDK (e.g. .../MacOSX26.sdk) and then fail on the
# C/C++ standard headers (mbstate_t, wcschr, FP_NAN, ...). Detect an SDK that
# actually exists and pass it explicitly as -isysroot (a command-line -isysroot
# overrides clang's bad guess). Empty on Linux. Override with MACOS_SDK_OVERRIDE.
MACOS_ISYSROOT =
ifeq ($(PLATFORM),macos)
    MACOS_SDK := $(or $(MACOS_SDK_OVERRIDE),$(shell s=$$(xcrun --show-sdk-path 2>/dev/null); \
        if [ -z "$$s" ] || [ ! -d "$$s" ]; then \
            s=$$(ls -d /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX*.sdk /Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk 2>/dev/null | sort -V | tail -1); \
        fi; \
        [ -n "$$s" ] && [ -d "$$s" ] && printf '%s' "$$s"))
    ifneq ($(strip $(MACOS_SDK)),)
        MACOS_ISYSROOT := -isysroot $(MACOS_SDK)
    endif
endif

# Minimal compiler flags (from original Makefile)
# Additional warnings and quality flags can be enabled in the EXTRA_FLAGS section above
# -Wno-deprecated-declarations: silences Boost.MultiArray's internal use of the
# deprecated boost::array::assign() (third-party headers, not our code).
# -MMD -MP: every compile also writes a .d makefile next to its object with the REAL include
# graph (-MP adds phony targets so a deleted header cannot strand a stale .d). The .d files are
# pulled in by the -include lines further down; they supplement the hand-written
# prerequisite lists (which are correct but incomplete), so header edits rebuild every affected
# object without a clean.
COMPILE_FLAGS = $(BASE_CFLAGS) -std=c++17 -Wno-psabi -Wno-cpp -Wno-deprecated-declarations -frounding-math -MMD -MP $(DEBUG_FLAGS) $(EXTRA_FLAGS) $(MACOS_ISYSROOT)
COMPILE_FLAGS_PS = $(BASE_CFLAGS_PS) -std=c++17 -Wno-psabi -Wno-cpp -Wno-deprecated-declarations -frounding-math -MMD -MP $(DEBUG_FLAGS) $(EXTRA_FLAGS) $(MACOS_ISYSROOT)
# NOTE: no -lboost_system: Boost.System is header-only since Boost 1.69,
# and recent Homebrew/Boost do not ship libboost_system ("library not found").
BASE_LIBS = -lboost_thread -lboost_filesystem -lboost_program_options -lgsl -lgslcblas -lm -lgmp -lmpfr -lfftw3f -lfftw3
HDF5_LIBS = -lhdf5 -lhdf5_cpp
# zlib: the tessellation cache deflates its body (CGAL_triangulation/tessellation_cache.h). A system
# library on both platforms, and HDF5 already depends on it.
BASE_LIBS += -lz

# libdeflate (optional): the cache's chunked gzip members are compressed and decompressed with it when
# it is installed (macOS: brew install libdeflate; Debian/Ubuntu: libdeflate-dev) -- 2.4x faster
# compression than zlib, decompression 1.13x faster than macOS's tuned zlib (more against stock zlib).
# The files stay plain gzip, so any build reads any cache. LIBDEFLATE=auto (default) uses it when found,
# LIBDEFLATE=0 forces zlib, LIBDEFLATE=1 requires it; LIBDEFLATE_PATH_OVERRIDE=/path points elsewhere.
LIBDEFLATE ?= auto
ifeq ($(PLATFORM),macos)
    LIBDEFLATE_PATH := $(or $(LIBDEFLATE_PATH_OVERRIDE),$(BREW_PREFIX)/opt/libdeflate)
else
    LIBDEFLATE_PATH := $(or $(LIBDEFLATE_PATH_OVERRIDE),/usr)
endif
HAVE_LIBDEFLATE := 0
ifeq ($(LIBDEFLATE),1)
    HAVE_LIBDEFLATE := 1
endif
ifeq ($(LIBDEFLATE),auto)
    ifneq ($(wildcard $(LIBDEFLATE_PATH)/include/libdeflate.h),)
        HAVE_LIBDEFLATE := 1
    endif
endif
ifeq ($(HAVE_LIBDEFLATE),1)
    OPTIONS_COMMON += -DDTFE_HAVE_LIBDEFLATE
    ifneq ($(LIBDEFLATE_PATH),/usr)
        INCLUDES  += $(call sysinc,$(LIBDEFLATE_PATH)/include)
        BASE_LIBS += -L$(LIBDEFLATE_PATH)/lib
    endif
    BASE_LIBS += -ldeflate
endif

# TBB-parallel Delaunay insertion (CGAL's Parallel_tag data structure + lock grid,
# CGAL_triangulation/triangulation.cpp). At run time the parallel insert builds tessellations of at least
# 1e6 points where more than one core is free for them (the result is the same
# Delaunay tessellation), or always with --parallel-triangulation or
# DTFE_TRIANGULATION_THREADS. 'auto': compiled in when the tbb package
# is installed (macOS: `brew install tbb`; Debian/Ubuntu: libtbb-dev); TBB=0 forces it off, TBB=1 requires it.
# The DT type changes with it, so o_ps/ is stamped and wiped on a toggle like the GPU mode. Override the TBB
# location with TBB_PATH_OVERRIDE=/path.
TBB ?= auto
ifeq ($(PLATFORM),macos)
    TBB_PATH := $(or $(TBB_PATH_OVERRIDE),$(BREW_PREFIX)/opt/tbb)
else
    TBB_PATH := $(or $(TBB_PATH_OVERRIDE),/usr)
endif
TBB_PS := 0
ifeq ($(TBB),1)
    TBB_PS := 1
endif
ifeq ($(TBB),auto)
    ifneq ($(wildcard $(TBB_PATH)/include/tbb/task_arena.h $(TBB_PATH)/include/oneapi/tbb/task_arena.h),)
        TBB_PS := 1
    endif
endif
ifeq ($(TBB_PS),1)
    OPTIONS_PS   += -DPARALLEL_TRIANGULATION -DCGAL_LINKED_WITH_TBB
    INCLUDES     += $(call sysinc,$(TBB_PATH)/include)
    PS_TBB_LIBS  := -L$(TBB_PATH)/lib -ltbb -ltbbmalloc
    TBB_BUILD_ARG := $(if $(filter 1,$(TBB)),TBB=1,)
else
    PS_TBB_LIBS  :=
    TBB_BUILD_ARG := $(if $(filter 0,$(TBB)),TBB=0,)
endif
# TBB_STD: the same data structure in the STANDARD binaries (DTFE, DTFE-double, DTFE-2d and the library); follows
# TBB; TBB_STD=0 keeps them on the sequential insert with the Fast_location hierarchy. The DT type is part of the
# library's DTFE() signature, so a program linked against libDTFE must be compiled with the same defines:
# demo/examples/Makefile repeats this block (keep the two the same). o/ is stamped and wiped on a toggle.
TBB_STD ?= $(TBB_PS)
ifeq ($(TBB_STD),1)
    OPTIONS      += -DPARALLEL_TRIANGULATION -DCGAL_LINKED_WITH_TBB
    INCLUDES     += $(call sysinc,$(TBB_PATH)/include)
    STD_TBB_LIBS := -L$(TBB_PATH)/lib -ltbb -ltbbmalloc
else
    STD_TBB_LIBS :=
endif

# Platform-specific OpenMP settings only
ifeq ($(PLATFORM),macos)
    COMPILE_FLAGS += -fopenmp=libomp
    COMPILE_FLAGS_PS += -fopenmp=libomp
else ifeq ($(PLATFORM),linux)
    COMPILE_FLAGS += -fopenmp
    COMPILE_FLAGS_PS += -fopenmp
endif

# Native-CPU tuning: let the compiler vectorise/schedule the interpolation hot loop for
# this exact chip. Pure tuning -- no -ffast-math, and with GCC no fused multiply-adds (below),
# so FP results stay bit-identical to a generic build (safe for comparing runs). Override with
# ARCH_FLAGS=..., disable with ARCH_FLAGS=
ifeq ($(PLATFORM),macos)
    ifeq ($(ARCH),arm64)
        ARCH_FLAGS ?= -mcpu=native
    else
        ARCH_FLAGS ?= -march=native
    endif
else
    ARCH_FLAGS ?= -march=native
endif
COMPILE_FLAGS    += $(ARCH_FLAGS)
COMPILE_FLAGS_PS += $(ARCH_FLAGS)

# GCC fuses a*b + c into one rounding (FMA) ACROSS statements by default (-ffp-contract=fast)
# wherever the target has FMA -- every arm64 machine, and x86 under -march=native -- so two code
# paths written to do the same arithmetic round differently, and a value's last bit depends on
# what the optimizer fused. With -ffp-contract=off a GCC build rounds each product where the
# source writes it, so its outputs do not depend on what the optimizer fused.
# Clang (the macOS build) only fuses within one expression, and is left as it is.
CC_IS_GCC := $(shell $(CC) --version 2>/dev/null | grep -qi 'free software foundation' && echo 1)
ifeq ($(CC_IS_GCC),1)
    COMPILE_FLAGS    += -ffp-contract=off
    COMPILE_FLAGS_PS += -ffp-contract=off
endif
# stamped in the object dirs (the mode checks above): objects compiled without the flag are
# wiped once instead of being linked with the new ones. Clang builds carry no stamp (no flag added).
FPC_STAMP := $(if $(filter 1,$(CC_IS_GCC)),.fpc_off,)
comma := ,
# BUILD_MODE, ARCH_FLAGS and EXTRA_FLAGS change every object as well: stamped in the object dirs (the mode
# and a checksum of the three), so a switch wipes the objects like a GPU-mode switch (otherwise
# 'make DTFE BUILD_MODE=debug' after a release build would find every object up to date and link the -O3
# objects with the sanitizers).
FLAGS_SUM := $(shell printf '%s' '$(subst ',,$(BUILD_MODE)|$(ARCH_FLAGS)|$(EXTRA_FLAGS))' | cksum | cut -d' ' -f1)
FLAGS_STAMP := .flags_$(BUILD_MODE)_$(FLAGS_SUM)

DTFE_INC = $(INCLUDES)

# Linking
ifeq ($(findstring -DHDF5,$(OPTIONS)),-DHDF5)
    DTFE_LIB = $(LIBRARIES) $(BASE_LIBS) $(HDF5_LIBS)   # OpenMP runtime auto-linked by -fopenmp; no -lomp / -lgomp (a duplicate-library warning)
else
    DTFE_LIB = $(LIBRARIES) $(BASE_LIBS)   # OpenMP runtime auto-linked by -fopenmp; no -lomp / -lgomp (a duplicate-library warning)
endif


# Build provenance (src/build_info.cc): recompiled by the programs' link recipe (program_rule), i.e. on every link, so
# the binary always reports the revision and time it was linked at ('--version', and the 'Build:'
# line every run prints into its log). Outside a git checkout the revision reads 'unknown'.
BUILD_REV  := $(shell git describe --always --dirty --abbrev=10 2>/dev/null || echo unknown)
BUILD_TIME := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
BUILD_INFO_FLAGS = -DDTFE_BUILD_REV='"$(BUILD_REV)"' -DDTFE_BUILD_TIME='"$(BUILD_TIME)"'

# (every header these include goes with them: message.h reads env_knobs.h, user_options.h run_mode.h and fixed_units.h)
HEADERS_1 = DTFE.h define.h user_options.h particle_data.h quantities.h Pvector.h math_functions.h  message.h box.h miscellaneous.h interpolations.h env_knobs.h run_mode.h fixed_units.h
HEADERS_2 = $(addprefix CGAL_triangulation/, CGAL_include_2D.h CGAL_include_3D.h vertexData.h particle_data_traits.h)

# Make's built-in rules are off: every object and binary here has a rule of its own, and make must never fall back on
# one of its own. After a build-mode switch it once ran its built-in link rule ('%: %.o') on a few objects while it
# re-read the .d files below ('clang++ o/r3d.o -o o/r3d', failing, the error ignored as for any -include'd file).
# That was never reproduced since; without the rule it cannot happen.
.SUFFIXES:
%: %.o

# compiler-generated header dependencies (see -MMD -MP at COMPILE_FLAGS); wildcard expands to
# nothing on a fresh checkout / after clean, so missing .d files never break the build
-include $(wildcard $(OBJ_DIR)/*.d)
-include $(wildcard $(OBJ_DIR_PS)/*.d)

# Declare phony targets
.PHONY: DTFE PS-DTFE library DTFE-build PS-DTFE-build library-build clean test-platform copy_headers set_directories set_directories_ps set_directories_2


############################# Objects ##################################
# One object per NAME.deps below, compiled the same way by the three builds: the standard programs in $(OBJ_DIR), the
# PS programs in $(OBJ_DIR_PS) (COMPILE_FLAGS_PS: -DPHASE_SPACE) and the library, whose position-independent *_l
# objects sit next to the standard ones. NAME.deps lists an object's source first, then the files it is rebuilt for
# (under $(SRC)): correct but incomplete lists, which the compiler's .d files (-MMD -MP at COMPILE_FLAGS) complete.
IO_SOURCES = $(addprefix io/, input_output.h gadget_reader_header.cc gadget_reader_binary.cc gadget_reader_HDF5.cc text_io.cc binary_io.cc)
TRIANG_HEADERS = $(addprefix CGAL_triangulation/, triangulation_common.h triangulation_miscellaneous.h field_computation.h sample_eval.h padding_test.h my_function.h CGAL_include_2D.h CGAL_include_3D.h vertexData.h particle_data_traits.h) define.h particle_data.h user_options.h box.h quantities.h Pvector.h message.h math_functions.h miscellaneous.h ps_point_eval.h thread_count.h
# The parts of the three split units, relative to $(SRC): each root (DTFE.cpp, auto_tune.cc,
# CGAL_triangulation/ps_interpolation.cc) includes its folder's parts in order and is still ONE object; a part is never
# compiled alone, so no object list may come from a glob over the sources. The parts follow their root's source in
# its NAME.deps, so an edit of a part rebuilds the root's objects in every build; they are set here, above the rules
# below, because $(eval) expands NAME.deps where it runs.
DTFE_PARTS      = $(patsubst $(SRC)/%,%,$(wildcard $(SRC)/dtfe/*.cc))
AUTO_TUNE_PARTS = $(patsubst $(SRC)/%,%,$(wildcard $(SRC)/auto_tune/*.cc))
PS_INTERP_PARTS = $(patsubst $(SRC)/%,%,$(wildcard $(SRC)/CGAL_triangulation/ps_interpolation/*.cc))
main.deps                     = main.cpp DTFE.h message.h user_options.h io/io.h interlacing.h
DTFE.deps                     = DTFE.cpp $(DTFE_PARTS) define.h particle_data.h user_options.h box.h quantities.h subpartition.h interpolations.h kdtree/kdtree2.hpp Pvector.h message.h miscellaneous.h auto_tune.h ps_point_eval.h web_fields.h
triangulation.deps            = CGAL_triangulation/triangulation.cpp $(TRIANG_HEADERS)
input_output.deps             = input_output.cc $(IO_SOURCES) user_options.h define.h quantities.h message.h particle_data.h box.h ps_point_eval.h
user_options.deps             = user_options.cc user_options.h define.h
quantities.deps               = quantities.cc quantities.h define.h
NGP_interpolation.deps        = NGP_interpolation.cc interpolations.h define.h
CIC_interpolation.deps        = CIC_interpolation.cc interpolations.h define.h grid_deposit_engine.h
TSC_interpolation.deps        = TSC_interpolation.cc interpolations.h define.h grid_deposit_engine.h
PCS_interpolation.deps        = PCS_interpolation.cc interpolations.h define.h grid_deposit_engine.h
SPH_interpolation.deps        = SPH_interpolation.cc interpolations.h define.h
interlacing.deps              = interlacing.cc interlacing.h define.h
random.deps                   = random.cc define.h user_options.h
# the derived velocity fields and the V-web / T-web
web_fields.deps               = web_fields.cc web_fields.h define.h user_options.h quantities.h message.h fftw_plans64.h
# the auto-tuner's model (auto_tune.h is its interface: an edit of the model rebuilds this object only)
auto_tune.deps                = auto_tune.cc $(AUTO_TUNE_PARTS) auto_tune.h user_options.h define.h
# --scratch-dir out-of-core backing replaces the global operator new/delete, so it goes into the programs only; the
# library links scratch_alloc_stub (which refuses) instead
scratch_alloc.deps            = scratch_alloc.cc scratch_alloc.h
scratch_alloc_stub.deps       = scratch_alloc_stub.cc scratch_alloc.h
unaveraged_interpolation.deps = CGAL_triangulation/unaveraged_interpolation.cc $(TRIANG_HEADERS)
averaged_interpolation_1.deps = CGAL_triangulation/averaged_interpolation_1.cc CGAL_triangulation/gpu_host.h $(TRIANG_HEADERS)
averaged_interpolation_2.deps = CGAL_triangulation/averaged_interpolation_2.cc $(TRIANG_HEADERS)
ps_interpolation.deps         = CGAL_triangulation/ps_interpolation.cc $(PS_INTERP_PARTS) CGAL_triangulation/gpu_host.h $(TRIANG_HEADERS)
ps_point_eval.deps            = CGAL_triangulation/ps_point_eval.cc CGAL_triangulation/ps_point_eval_detail.h $(TRIANG_HEADERS)
ps_serve.deps                 = CGAL_triangulation/ps_serve.cc CGAL_triangulation/ps_point_eval_detail.h $(TRIANG_HEADERS)
ps_mean_streams.deps          = CGAL_triangulation/ps_mean_streams.cc $(TRIANG_HEADERS)

# The objects in link order (the order fixes the binaries' layout). build_info is compiled by the programs' link
# recipe itself; kdtree2 and r3d have rules of their own (below), and the library links $(OBJ_DIR)'s r3d.
CC_STEMS     = user_options quantities NGP_interpolation CIC_interpolation TSC_interpolation PCS_interpolation SPH_interpolation interlacing random web_fields
TRIANG_STEMS = unaveraged_interpolation averaged_interpolation_1 averaged_interpolation_2 ps_interpolation ps_point_eval ps_serve ps_mean_streams
PROGRAM_HEAD = DTFE triangulation main kdtree2 r3d
PROGRAM_TAIL = $(CC_STEMS) scratch_alloc auto_tune input_output $(TRIANG_STEMS)
LIBRARY_OBJS = $(call objs,$(OBJ_DIR),DTFE_l triangulation_l kdtree2_l r3d scratch_alloc_stub_l $(addsuffix _l,$(CC_STEMS) auto_tune input_output $(TRIANG_STEMS)))
# $(call objs,DIR,NAMES): DIR/NAME$(OBJ_EXT) for each of NAMES
objs = $(patsubst %,$(1)/%$(OBJ_EXT),$(2))
# -fPIC for the library's objects (the *_l ones), nothing for the programs'
PIC = $(if $(filter %_l$(OBJ_EXT),$@),-fPIC)

# $(call cxx_rule,OBJECT,NAME,FLAGS): OBJECT$(OBJ_EXT) from NAME's source, with the compile flags in the variable FLAGS
define cxx_rule
$(1)$(OBJ_EXT): $(addprefix $(SRC)/,$($(2).deps)) Makefile
	$$(CC) $$($(3)) $$(PIC) $$(DTFE_INC) -o $$@ -c $(SRC)/$(firstword $($(2).deps))
endef
$(foreach n,DTFE triangulation main $(PROGRAM_TAIL),$(eval $(call cxx_rule,$(OBJ_DIR)/$(n),$(n),COMPILE_FLAGS)))
$(foreach n,DTFE triangulation main $(PROGRAM_TAIL),$(eval $(call cxx_rule,$(OBJ_DIR_PS)/$(n),$(n),COMPILE_FLAGS_PS)))
$(foreach n,DTFE triangulation scratch_alloc_stub $(CC_STEMS) auto_tune input_output $(TRIANG_STEMS),$(eval $(call cxx_rule,$(OBJ_DIR)/$(n)_l,$(n),COMPILE_FLAGS)))

# kdtree2 keeps flags of its own (-ffast-math, none of the project's defines); one object per build
$(OBJ_DIR)/kdtree2$(OBJ_EXT) $(OBJ_DIR_PS)/kdtree2$(OBJ_EXT) $(OBJ_DIR)/kdtree2_l$(OBJ_EXT): $(SRC)/kdtree/kdtree2.hpp $(SRC)/kdtree/kdtree2.cpp Makefile
	$(CC) -O3 -ffast-math -fomit-frame-pointer $(PIC) -Wno-deprecated-declarations -MMD -MP $(MACOS_ISYSROOT) $(DTFE_INC) -o $@ -c $(SRC)/kdtree/kdtree2.cpp

# vendored r3d (third_party/r3d, plain C99; see its README for license/citation): exact
# tetrahedron-cell intersection moments for --exact (both binaries' default '_a' average). One
# object per dir; -fPIC so the o/ object also links into libDTFE (one shared object,
# no _l twin). -x c: $(CC) is a C++ driver.
R3D_SRC = third_party/r3d/r3d.c
R3D_FLAGS = -x c -std=c99 -O3 -fPIC -fomit-frame-pointer -MMD -MP $(MACOS_ISYSROOT)

$(OBJ_DIR)/r3d$(OBJ_EXT) $(OBJ_DIR_PS)/r3d$(OBJ_EXT): $(R3D_SRC) third_party/r3d/r3d.h third_party/r3d/r3d-config.h Makefile
	$(CC) $(R3D_FLAGS) -o $@ -c $(R3D_SRC)

# The Metal GPU deposit hosts (METAL=1 only) embed their kernel source through a generated header, so the binary
# needs no runtime file lookup; Metal compiles it once (ps_metal_host.cc pipes()). $(call embed_kernel,DIR,NAME,
# ARRAY,PIECES) writes DIR/NAME_msl.h: the PIECES, concatenated in this order, as the string ARRAY. The pieces: the
# exact clip helpers and the run-wide fixed-point words (metal/fixed_words.metal.inc), the host's own parameter header
# (plain C++ that is valid MSL: ps_kernel_params.h / dtfe_deposit_params.h), for the PS kernels their shared guard,
# 64-bit add and items' map reader (metal/ps_common.metal.inc), then the kernels.
define embed_kernel
$(1)/$(2)_msl.h: $(4) Makefile
	@$$(MKDIR_P) $(1)
	{ printf 'static const char $(3)[] = R"MSL(\n'; cat $(4); printf '\n)MSL";\n'; } > $$@.tmp
	mv $$@.tmp $$@
endef
$(eval $(call embed_kernel,$(OBJ_DIR_PS),ps_deposit,PS_DEPOSIT_MSL,metal/exact_clip.metal.inc metal/fixed_words.metal.inc $(SRC)/CGAL_triangulation/ps_kernel_params.h metal/ps_common.metal.inc metal/ps_deposit.metal))
$(eval $(call embed_kernel,$(OBJ_DIR),dtfe_deposit,DTFE_DEPOSIT_MSL,metal/exact_clip.metal.inc metal/fixed_words.metal.inc $(SRC)/CGAL_triangulation/dtfe_deposit_params.h metal/dtfe_deposit.metal))

$(OBJ_DIR_PS)/ps_metal_host$(OBJ_EXT): $(SRC)/CGAL_triangulation/ps_metal_host.cc $(SRC)/CGAL_triangulation/gpu_host.h $(SRC)/CGAL_triangulation/gpu_metal_retry.h $(SRC)/CGAL_triangulation/ps_deposit_params.h $(SRC)/CGAL_triangulation/ps_kernel_params.h $(SRC)/fixed_units.h $(OBJ_DIR_PS)/ps_deposit_msl.h Makefile
	$(CC) $(COMPILE_FLAGS_PS) -isystem third_party/metal-cpp -I $(OBJ_DIR_PS) -o $@ -c $(SRC)/CGAL_triangulation/ps_metal_host.cc

# the standard host: one object for the programs, one (*_l) for the library
$(OBJ_DIR)/dtfe_metal_host$(OBJ_EXT) $(OBJ_DIR)/dtfe_metal_host_l$(OBJ_EXT): $(SRC)/CGAL_triangulation/dtfe_metal_host.cc $(SRC)/CGAL_triangulation/gpu_host.h $(SRC)/CGAL_triangulation/gpu_metal_retry.h $(SRC)/CGAL_triangulation/dtfe_gpu_fixed.h $(SRC)/CGAL_triangulation/ps_deposit_params.h $(SRC)/CGAL_triangulation/dtfe_deposit_params.h $(OBJ_DIR)/dtfe_deposit_msl.h Makefile
	$(CC) $(COMPILE_FLAGS) $(PIC) -isystem third_party/metal-cpp -I $(OBJ_DIR) -o $@ -c $(SRC)/CGAL_triangulation/dtfe_metal_host.cc

############################# The programs and the library ##################################
# DTFE triangulates in Eulerian space (standard DTFE); PS-DTFE (-DPHASE_SPACE) in Lagrangian space (multi-stream
# regions); DOUBLE=1 and DIM=2 name their pairs by BIN_SUFFIX.
# The mode check must FINISH before any object/header rule starts: under -j its wipe runs
# concurrently with sibling prerequisites and can delete files mid-recipe (observed: the
# generated *_msl.h lost its opening raw-string line). The recursive $(MAKE) starts only
# after the check's recipe completes, so the real build can never race the wipe.
DTFE: dtfe_gpu_mode_check
	@$(MAKE) DTFE-build

PS-DTFE: ps_gpu_mode_check
	@$(MAKE) PS-DTFE-build

# $(call program_rule,NAME,DIR,FLAGS,MKDIR,GPU-OBJECTS,TBB-LIBRARIES): NAME-build compiles build_info and links
# $(BIN_DIR)/NAME$(BIN_SUFFIX) from DIR's objects
define program_rule
$(1)-build: $(4) $(call objs,$(2),$(PROGRAM_HEAD) $(PROGRAM_TAIL)) $(5) Makefile
	$$(CC) $$($(3)) $$(BUILD_INFO_FLAGS) -o $(2)/build_info$(OBJ_EXT) -c $(SRC)/build_info.cc
	$$(CC) $$($(3)) $(call objs,$(2),$(PROGRAM_HEAD) build_info $(PROGRAM_TAIL)) $(5) $$(DTFE_LIB) $$(GPU_LIBS) $(6) -o $(BIN_DIR)/$(1)$$(BIN_SUFFIX)$(EXE_EXT)
endef
$(eval $(call program_rule,DTFE,$(OBJ_DIR),COMPILE_FLAGS,set_directories,$(DTFE_GPU_OBJS),$(STD_TBB_LIBS)))
$(eval $(call program_rule,PS-DTFE,$(OBJ_DIR_PS),COMPILE_FLAGS_PS,set_directories_ps,$(PS_GPU_OBJS),$(PS_TBB_LIBS)))

# the shared library libDTFE: the same wipe-vs-build serialization (it shares $(OBJ_DIR))
library: dtfe_gpu_mode_check
	@$(MAKE) library-build

library-build: set_directories set_directories_2 copy_headers $(LIBRARY_OBJS) $(DTFE_GPU_L_OBJS) Makefile
	$(CC) $(COMPILE_FLAGS) -shared $(LIBRARY_OBJS) $(DTFE_GPU_L_OBJS) $(DTFE_LIB) $(GPU_LIBS) $(STD_TBB_LIBS) -o $(LIB_DIR)/libDTFE$(SHARED_EXT)


clean:
	$(RM_RF) $(BIN_DIR)/DTFE$(EXE_EXT) $(BIN_DIR)/PS-DTFE$(EXE_EXT) $(BIN_DIR)/DTFE-double$(EXE_EXT) $(BIN_DIR)/PS-DTFE-double$(EXE_EXT) $(BIN_DIR)/DTFE-2d*$(EXE_EXT) $(BIN_DIR)/PS-DTFE-2d*$(EXE_EXT) ./o_2d*/*$(OBJ_EXT) ./o_ps_2d*/*$(OBJ_EXT) ./o_2d*/*.d ./o_ps_2d*/*.d ./o/*$(OBJ_EXT) ./o_ps/*$(OBJ_EXT) ./o/*.d ./o_ps/*.d ./o_d/*$(OBJ_EXT) ./o_ps_d/*$(OBJ_EXT) ./o_d/*.d ./o_ps_d/*.d $(LIB_DIR)/*DTFE$(SHARED_EXT)

# ---- make check: a quick test of the built 3D pair, ./DTFE and ./PS-DTFE, with no simulation data. It builds nothing
# (a plain 'make DTFE' could change the build mode and wipe a METAL=1 build). tests/run_tests.py: standard DTFE on
# small inputs it writes, the last test against the stored reference in tests/reference;
# tests/dtfe_mc_average_check.sh: standard DTFE on the bundled demo/z0_64.gadget (the '_a' average keeps the mass,
# whatever the split); tests/ps_smoke_test.sh: PS-DTFE end to end on a generated Zel'dovich pancake (finite, mass kept,
# multi-stream). The runs use the CPU, but for the demo suite's '--gpu' part on a Metal build. Needs the python with
# numpy and h5py that scripts/h5py_python.sh finds. The full suites: tests/ci_suite.sh.
.PHONY: check
check:
	@for b in DTFE PS-DTFE; do [ -x ./$$b ] || { echo ">> make check: ./$$b is not built (make DTFE PS-DTFE)"; exit 1; }; done
	@. scripts/h5py_python.sh && env -u DTFE_TEST_REFERENCE_DIR -u DTFE_TEST_PRECISION "$$(h5py_python)" tests/run_tests.py
	@env -u DTFE_TEST_PRECISION tests/dtfe_mc_average_check.sh --no-build
	@env -u DTFE_TEST_PRECISION tests/ps_smoke_test.sh --no-build
	@echo ">> make check PASSED: tests/run_tests.py, tests/dtfe_mc_average_check.sh, tests/ps_smoke_test.sh"

# ---- deps-check: verify the required headers/libraries exist BEFORE a long compile dies
# mid-build with a cryptic missing-header error, and print the per-platform install command
# for anything missing. Each spec is header:name:brew-package:apt-package; the header is
# searched in every -I directory this Makefile will pass to the compiler.
DEPS_SPECS = \
	gsl/gsl_math.h:GSL:gsl:libgsl-dev \
	boost/program_options.hpp:Boost:boost:libboost-all-dev \
	CGAL/version.h:CGAL:cgal:libcgal-dev \
	gmp.h:GMP:gmp:libgmp-dev \
	mpfr.h:MPFR:mpfr:libmpfr-dev \
	H5Cpp.h:HDF5-C++:hdf5:libhdf5-dev \
	fftw3.h:FFTW3:fftw:libfftw3-dev

.PHONY: deps-check
deps-check:
	@echo ">> checking build dependencies (compiler: $(CC))"
	@command -v $(firstword $(CC)) >/dev/null 2>&1 \
		|| { echo "  MISSING  C++ compiler '$(CC)' (macOS: xcode-select --install && brew install llvm libomp; Linux: install g++)"; exit 1; }
	@missing_brew=""; missing_apt=""; fail=0; \
	incdirs="$(subst -I ,,$(INCLUDES)) /usr/include /usr/local/include /usr/include/*-linux-gnu"; \
	for spec in $(DEPS_SPECS); do \
		hdr=$$(echo "$$spec" | cut -d':' -f1); \
		name=$$(echo "$$spec" | cut -d':' -f2); \
		brewpkg=$$(echo "$$spec" | cut -d':' -f3); \
		aptpkg=$$(echo "$$spec" | cut -d':' -f4); \
		found=""; \
		for inc in $$incdirs; do \
			if [ -e "$$inc/$$hdr" ]; then found="$$inc/$$hdr"; break; fi; \
		done; \
		if [ -n "$$found" ]; then \
			printf '   OK       %-12s %s\n' "$$name" "$$found"; \
		else \
			printf '   MISSING  %-12s (header %s not found on the include path)\n' "$$name" "$$hdr"; \
			fail=1; missing_brew="$$missing_brew $$brewpkg"; missing_apt="$$missing_apt $$aptpkg"; \
		fi; \
	done; \
	if [ -e "$(LIBDEFLATE_PATH)/include/libdeflate.h" ]; then \
		printf '   OK       %-12s %s (optional: the tessellation cache codec)\n' "libdeflate" "$(LIBDEFLATE_PATH)/include/libdeflate.h"; \
	else \
		printf '   optional %-12s not found: the tessellation cache uses zlib (brew install libdeflate / apt-get install libdeflate-dev)\n' "libdeflate"; \
	fi; \
	if [ "$$fail" -eq 1 ]; then \
		echo ""; \
		echo ">> missing dependencies -- install them with:"; \
		echo "   macOS (Homebrew):  brew install$$missing_brew"; \
		echo "   Ubuntu/Debian:     sudo apt-get install$$missing_apt"; \
		echo "   Fedora/RHEL:       see the README 'Prerequisites' section"; \
		exit 1; \
	fi; \
	echo ">> all build dependencies found"

# Platform detection test (useful for debugging)
test-platform:
	@echo "Detected platform: $(PLATFORM)"
	@echo "Operating system: $(UNAME_S)"
	@echo "Architecture: $(shell uname -m)"
ifeq ($(PLATFORM),macos)
	@echo "Homebrew prefix: $(BREW_PREFIX)"
endif
	@echo "Build mode: $(BUILD_MODE)"
	@echo "Compiler: $(CC)"
	@echo "Executable extension: '$(EXE_EXT)'"
	@echo "Shared library extension: '$(SHARED_EXT)'"
	@echo "Object file extension: '$(OBJ_EXT)'"
	@echo "GSL path: $(GSL_PATH)"
	@echo "Boost path: $(BOOST_PATH)"
	@echo "CGAL path: $(CGAL_PATH)"
	@echo "Compile flags: $(COMPILE_FLAGS)"
	@echo "Include flags: $(INCLUDES)"
	@echo "Libraries: $(DTFE_LIB)"

# the folders are made here too: under 'make -j' this rule can run before set_directories_2, and on a
# fresh checkout (CI) './DTFE_include' does not exist yet -- 'make library' would fail with 'cp: target ... No such
# file'
copy_headers:
	@$(MKDIR_P) $(INC_DIR)/CGAL_triangulation
	cp $(addprefix $(SRC)/, $(HEADERS_1)) $(INC_DIR)
	cp $(addprefix $(SRC)/, $(HEADERS_2)) $(INC_DIR)/CGAL_triangulation

set_directories:
	@$(MKDIR_P) $(OBJ_DIR)
	@$(MKDIR_P) $(BIN_DIR)

set_directories_ps:
	@$(MKDIR_P) $(OBJ_DIR_PS)

set_directories_2:
	@$(MKDIR_P) $(LIB_DIR)
	@$(MKDIR_P) $(INC_DIR)
	@$(MKDIR_P) $(INC_DIR)/CGAL_triangulation
