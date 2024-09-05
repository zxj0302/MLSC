ESCAPE_HOME := ../

TARGETS := count_three count_four count_five count_closures count_orbit_three count_orbit_four count_orbit_five

OBJECTS := $(TARGETS:%=%.o)

include ../common.mk

INCLUDES += -I..
LDFLAGS += -L..

count_% : count_%.o ../libescape.a
	$(COMPILE_AND_LINK)
