# Portable zero-dependency Makefile for ASH-MD C++17 Engine
# - On Linux x86_64 (Instabox AMD EPYC): compiles with -mavx2 -mfma -pthread
# - On macOS arm64 (Apple M4): compiles with -O3 -pthread (ARM64 NEON native)

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

CXX ?= c++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -fPIC -pthread

ifeq ($(UNAME_S),Darwin)
    SHARED_EXT := dylib
    LDFLAGS := -dynamiclib -pthread
    ifeq ($(UNAME_M),arm64)
        CXXFLAGS += -mcpu=native
    endif
else
    SHARED_EXT := so
    LDFLAGS := -shared -pthread
    ifeq ($(UNAME_M),x86_64)
        CXXFLAGS += -mavx2 -mfma
    endif
endif

BUILD_DIR := build
TARGET_LIB := $(BUILD_DIR)/libashmd.$(SHARED_EXT)

HEADERS := $(wildcard include/ashmd/*.hpp)

all: $(TARGET_LIB)

$(TARGET_LIB): src/ashmd_engine.cpp $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ src/ashmd_engine.cpp

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
