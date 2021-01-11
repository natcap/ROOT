.PHONY: binaries deploy

PYTHON:=python
GSUTIL:=gsutil
# alternative zip would be zip -r.  7zip is what's available on appveyor.
ZIP:=7z a -r
ARCH:=$(shell $(PYTHON) -c "import sys; print('x86' if sys.maxsize <= 2**32 else 'x64')")
OS:=$(shell $(PYTHON) -c"import platform; print(platform.system())")
VERSION:=$(shell $(PYTHON) -c "import setuptools_scm; print(setuptools_scm.get_version(version_scheme='post-release',local_scheme='node-and-date'))")
BUILD_DIR:=build
DIST_DIR:=dist

# We're assuming that this will be run on github actions, where bash is always available.
ifeq ($(OS),Windows_NT)
	SHELL := /usr/bin/bash
else
	SHELL := /bin/bash
endif

BIN_DIR=$(DIST_DIR)/root-$(ARCH)-$(OS)
BIN_ZIP=$(DIST_DIR)/root-$(VERSION)-$(OS)-$(ARCH).zip

# very useful for debugging variables!
print-%:
	@echo "$* = $($*)"

$(DIST_DIR) $(BUILD_DIR):
	mkdir -p $@

$(BIN_DIR): dist
	rm -rf $(BIN_DIR) $(BUILD_DIR)
	mkdir -p $(BIN_DIR)
	$(PYTHON) -m PyInstaller --workpath $(BUILD_DIR)/pyi-build --clean --distpath $(BIN_DIR) --additional-hooks-dir=exe/hooks exe/root.spec
	$(BIN_DIR)/root/root --test-imports
	cp root_launcher.bat $(BIN_DIR)/root.bat

binaries: $(BIN_ZIP)
$(BIN_ZIP): $(BIN_DIR)
	cd $(DIST_DIR) && $(ZIP) $(notdir $(BIN_ZIP)) $(notdir $(BIN_DIR))
