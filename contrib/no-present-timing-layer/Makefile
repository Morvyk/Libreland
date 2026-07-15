# Build + install the no-present-timing Vulkan implicit layer.
#
#   make            build the .so
#   make install    install .so + manifest (honours DESTDIR / PREFIX)
#   make uninstall  remove them
#
# The manifest's library_path is an absolute /usr/lib path, so PREFIX must
# stay /usr for an implicit layer the loader can resolve.

PREFIX    ?= /usr
LIBDIR    ?= $(PREFIX)/lib
LAYERDIR  ?= $(PREFIX)/share/vulkan/implicit_layer.d

SO   := libVkLayer_libreland_no_present_timing.so
JSON := VkLayer_libreland_no_present_timing.json

CC      ?= cc
CFLAGS  ?= -O2
override CFLAGS += -std=c11 -fPIC -Wall -Wextra
LDFLAGS ?=
override LDFLAGS += -shared

$(SO): no_present_timing.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

.PHONY: install uninstall clean
install: $(SO)
	install -Dm755 $(SO)   $(DESTDIR)$(LIBDIR)/$(SO)
	install -Dm644 $(JSON) $(DESTDIR)$(LAYERDIR)/$(JSON)

uninstall:
	rm -f $(DESTDIR)$(LIBDIR)/$(SO) $(DESTDIR)$(LAYERDIR)/$(JSON)

clean:
	rm -f $(SO)
