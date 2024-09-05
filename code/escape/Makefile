ESCAPE_HOME := .

OBJECTS := Graph.o GraphIO.o TriangleProgram.o

TARGETS := libescape.a

all: tests exe

libescape.a : $(OBJECTS)
	ar cruv $@ $^

include common.mk

tests: libescape.a
	$(MAKE) -C tests

cleantests:
	$(MAKE) -C tests clean cleandep

exe: libescape.a
	$(MAKE) -C exe

cleanexe:
	$(MAKE) -C exe clean cleandep

.PHONY: tests exe



