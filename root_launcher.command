#!/bin/bash

# the QT_MAC_WANTS_LAYER definition is supposed to have been set by the
# runtime hook, but doesn't seem to be working.  Setting it here allows the
# binary to run on OSX Big Sur.
QT_MAC_WANTS_LAYER=1 `dirname $0`/root/root "$@"
