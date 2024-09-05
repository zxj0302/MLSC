ESCAPE_HOME := ../

TARGETS := test_GraphIO test_DAG test_tri test_four test_five test_TriangleProgram test_fivecycle \
            test_almostclique test_FiveFromTriangles test_four_old test_trees test_three test_timefour test_oldfive test_2truss test_trussfour test_prunefour test_trussfive

OBJECTS := $(TARGETS:%=%.o)

include ../common.mk

INCLUDES += -I..
LDFLAGS += -L..

test_% : test_%.o ../libescape.a
	$(COMPILE_AND_LINK)
