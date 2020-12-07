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

BIN_DIR=$(DIST_DIR)/root-$(ARCH)-$(OS)
BIN_ZIP=$(DIST_DIR)/root-$(VERSION)-$(OS)-$(ARCH).zip

# very useful for debugging variables!
print-%:
	@echo "$* = $($*)"

$(DIST_DIR) $(BUILD_DIR):
	powershell.exe mkdir -Force -Path $@

$(BIN_DIR): dist
	-powershell.exe mkdir -Force -Path $(BIN_DIR)
	$(PYTHON) -m PyInstaller --workpath $(BUILD_DIR)/pyi-build --clean --distpath $(BIN_DIR) --additional-hooks-dir=exe/hooks exe/root.spec
	$(BIN_DIR)/root/root.exe --test-imports
	cp root_launcher.bat $(BIN_DIR)/root.bat

binaries: $(BIN_ZIP)
$(BIN_ZIP): $(BIN_DIR)
	cd $(DIST_DIR) && $(ZIP) $(notdir $(BIN_ZIP)) $(notdir $(BIN_DIR))

FORKNAME := $(filter-out ssh: http: https:, $(subst /, ,$(shell hg config paths.default)))
FORKUSER := $(word 2, $(subst /, ,$(FORKNAME)))
ifeq ($(FORKUSER),natcap)
	BUCKET := gs://releases.naturalcapitalproject.org
	DIST_URL_BASE := $(BUCKET)/root/$(VERSION)
else
	BUCKET := gs://natcap-dev-build-artifacts
	DIST_URL_BASE := $(BUCKET)/root/$(FORKUSER)/$(VERSION)
endif
DOWNLOAD_DIR_URL := $(subst gs://,https://storage.googleapis.com/,$(DIST_URL_BASE))

deploy: $(BIN_ZIP)
	$(GSUTIL) cp $(BIN_ZIP) $(DIST_URL_BASE)/$(notdir $(BIN_ZIP))
	@echo "Binaries have been copied to:"
	@echo "  * $(DOWNLOAD_DIR_URL)/$(subst $(DIST_DIR)/,,$(BIN_ZIP))"
