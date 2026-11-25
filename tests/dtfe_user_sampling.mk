# Builds tests/dtfe_user_sampling.cpp against libDTFE with demo/examples/Makefile's compiler, flags and libraries.
#   make -C demo/examples -f Makefile -f ../../tests/dtfe_user_sampling.mk USER_SAMPLING_BIN=<path> <path>
# On macOS it pins the SDK as the root Makefile does. The rpath finds libDTFE.so on Linux; on macOS the library's
# install name is its build path relative to the repository root, where the suite runs the driver.
USER_SAMPLING_BIN ?= dtfe_user_sampling
USER_SAMPLING_SYSROOT =
ifeq ($(UNAME_S),Darwin)
    USER_SAMPLING_SDK := $(shell s=$$(xcrun --show-sdk-path 2>/dev/null); \
        if [ -z "$$s" ] || [ ! -d "$$s" ]; then \
            s=$$(ls -d /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX*.sdk /Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk 2>/dev/null | sort -V | tail -1); \
        fi; \
        [ -n "$$s" ] && [ -d "$$s" ] && printf '%s' "$$s")
    ifneq ($(strip $(USER_SAMPLING_SDK)),)
        USER_SAMPLING_SYSROOT := -isysroot $(USER_SAMPLING_SDK)
    endif
endif
$(USER_SAMPLING_BIN): $(ROOT)/tests/dtfe_user_sampling.cpp
	$(CC) $(COMPILE_FLAGS) $(USER_SAMPLING_SYSROOT) $(INCLUDES) $< $(LIBS) -Wl,-rpath,$(abspath $(ROOT)) -o $@
