# CMake generated Testfile for 
# Source directory: /workspace/code/motivo
# Build directory: /workspace/code/motivo/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(build-graph "/workspace/code/motivo/build/bin/motivo-graph" "--input" "/workspace/code/motivo/graphs/test-graph.txt" "--output" "test-graph")
set_tests_properties(build-graph PROPERTIES  WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;4;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-tests "/workspace/code/motivo/build/tests/motivo-tests" "-tce=*slow")
set_tests_properties(motivo-tests PROPERTIES  DEPENDS "build-graph" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;5;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-build-1 "/workspace/code/motivo/build/bin/motivo-build" "-g" "test-graph" "-s" "1" "-c" "5" "-o" "test" "--seed" "42")
set_tests_properties(motivo-build-1 PROPERTIES  DEPENDS "build-graph" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;9;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-merge-1 "/workspace/code/motivo/build/bin/motivo-merge" "-o" "test.1" "test.1.cnt")
set_tests_properties(motivo-merge-1 PROPERTIES  DEPENDS "motivo-build-1" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;12;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-build-2 "/workspace/code/motivo/build/bin/motivo-build" "-g" "test-graph" "-s" "2" "-i" "test" "-o" "test")
set_tests_properties(motivo-build-2 PROPERTIES  DEPENDS "motivo-merge-1" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;18;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-merge-2 "/workspace/code/motivo/build/bin/motivo-merge" "-o" "test.2" "test.2.cnt")
set_tests_properties(motivo-merge-2 PROPERTIES  DEPENDS "motivo-build-2" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;21;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-build-3 "/workspace/code/motivo/build/bin/motivo-build" "-g" "test-graph" "-s" "3" "-i" "test" "-o" "test")
set_tests_properties(motivo-build-3 PROPERTIES  DEPENDS "motivo-merge-2" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;18;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-merge-3 "/workspace/code/motivo/build/bin/motivo-merge" "-o" "test.3" "test.3.cnt")
set_tests_properties(motivo-merge-3 PROPERTIES  DEPENDS "motivo-build-3" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;21;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-build-4 "/workspace/code/motivo/build/bin/motivo-build" "-g" "test-graph" "-s" "4" "-i" "test" "-o" "test")
set_tests_properties(motivo-build-4 PROPERTIES  DEPENDS "motivo-merge-3" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;18;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-merge-4 "/workspace/code/motivo/build/bin/motivo-merge" "-o" "test.4" "test.4.cnt")
set_tests_properties(motivo-merge-4 PROPERTIES  DEPENDS "motivo-build-4" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;21;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-build-5 "/workspace/code/motivo/build/bin/motivo-build" "-g" "test-graph" "-s" "5" "-i" "test" "-o" "test")
set_tests_properties(motivo-build-5 PROPERTIES  DEPENDS "motivo-merge-4" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;18;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-merge-5 "/workspace/code/motivo/build/bin/motivo-merge" "-o" "test.5" "test.5.cnt")
set_tests_properties(motivo-merge-5 PROPERTIES  DEPENDS "motivo-build-5" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;21;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-build-mt-1 "/workspace/code/motivo/build/bin/motivo-build" "-g" "test-graph" "-s" "1" "-c" "5" "-o" "test-mt" "--seed" "42")
set_tests_properties(motivo-build-mt-1 PROPERTIES  DEPENDS "build-graph" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;25;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-merge-mt-1 "/workspace/code/motivo/build/bin/motivo-merge" "-o" "test-mt.1" "test.1.cnt")
set_tests_properties(motivo-merge-mt-1 PROPERTIES  DEPENDS "motivo-build-mt-1" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;28;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-build-mt-2 "/workspace/code/motivo/build/bin/motivo-build" "-g" "test-graph" "-s" "2" "-i" "test-mt" "-o" "test-mt" "--threads" "0")
set_tests_properties(motivo-build-mt-2 PROPERTIES  DEPENDS "motivo-merge-mt-1" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;34;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-merge-mt-2 "/workspace/code/motivo/build/bin/motivo-merge" "-o" "test-mt.2" "test-mt.2.cnt")
set_tests_properties(motivo-merge-mt-2 PROPERTIES  DEPENDS "motivo-build-mt-2" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;37;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-dtz-mt-matches-st-2 "/usr/bin/cmake" "-E" "compare_files" "test-mt.2.dtz" "test.2.dtz")
set_tests_properties(motivo-dtz-mt-matches-st-2 PROPERTIES  DEPENDS "motivo-merge-mt-2" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;40;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-rts-mt-matches-st-2 "/usr/bin/cmake" "-E" "compare_files" "test-mt.2.rts" "test.2.rts")
set_tests_properties(motivo-rts-mt-matches-st-2 PROPERTIES  DEPENDS "motivo-merge-mt-2" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;43;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-build-mt-3 "/workspace/code/motivo/build/bin/motivo-build" "-g" "test-graph" "-s" "3" "-i" "test-mt" "-o" "test-mt" "--threads" "0")
set_tests_properties(motivo-build-mt-3 PROPERTIES  DEPENDS "motivo-merge-mt-2" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;34;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-merge-mt-3 "/workspace/code/motivo/build/bin/motivo-merge" "-o" "test-mt.3" "test-mt.3.cnt")
set_tests_properties(motivo-merge-mt-3 PROPERTIES  DEPENDS "motivo-build-mt-3" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;37;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-dtz-mt-matches-st-3 "/usr/bin/cmake" "-E" "compare_files" "test-mt.3.dtz" "test.3.dtz")
set_tests_properties(motivo-dtz-mt-matches-st-3 PROPERTIES  DEPENDS "motivo-merge-mt-3" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;40;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-rts-mt-matches-st-3 "/usr/bin/cmake" "-E" "compare_files" "test-mt.3.rts" "test.3.rts")
set_tests_properties(motivo-rts-mt-matches-st-3 PROPERTIES  DEPENDS "motivo-merge-mt-3" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;43;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-build-mt-4 "/workspace/code/motivo/build/bin/motivo-build" "-g" "test-graph" "-s" "4" "-i" "test-mt" "-o" "test-mt" "--threads" "0")
set_tests_properties(motivo-build-mt-4 PROPERTIES  DEPENDS "motivo-merge-mt-3" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;34;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-merge-mt-4 "/workspace/code/motivo/build/bin/motivo-merge" "-o" "test-mt.4" "test-mt.4.cnt")
set_tests_properties(motivo-merge-mt-4 PROPERTIES  DEPENDS "motivo-build-mt-4" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;37;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-dtz-mt-matches-st-4 "/usr/bin/cmake" "-E" "compare_files" "test-mt.4.dtz" "test.4.dtz")
set_tests_properties(motivo-dtz-mt-matches-st-4 PROPERTIES  DEPENDS "motivo-merge-mt-4" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;40;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-rts-mt-matches-st-4 "/usr/bin/cmake" "-E" "compare_files" "test-mt.4.rts" "test.4.rts")
set_tests_properties(motivo-rts-mt-matches-st-4 PROPERTIES  DEPENDS "motivo-merge-mt-4" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;43;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-build-mt-5 "/workspace/code/motivo/build/bin/motivo-build" "-g" "test-graph" "-s" "5" "-i" "test-mt" "-o" "test-mt" "--threads" "0")
set_tests_properties(motivo-build-mt-5 PROPERTIES  DEPENDS "motivo-merge-mt-4" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;34;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-merge-mt-5 "/workspace/code/motivo/build/bin/motivo-merge" "-o" "test-mt.5" "test-mt.5.cnt")
set_tests_properties(motivo-merge-mt-5 PROPERTIES  DEPENDS "motivo-build-mt-5" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;37;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-dtz-mt-matches-st-5 "/usr/bin/cmake" "-E" "compare_files" "test-mt.5.dtz" "test.5.dtz")
set_tests_properties(motivo-dtz-mt-matches-st-5 PROPERTIES  DEPENDS "motivo-merge-mt-5" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;40;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
add_test(motivo-rts-mt-matches-st-5 "/usr/bin/cmake" "-E" "compare_files" "test-mt.5.rts" "test.5.rts")
set_tests_properties(motivo-rts-mt-matches-st-5 PROPERTIES  DEPENDS "motivo-merge-mt-5" WORKING_DIRECTORY "tests" _BACKTRACE_TRIPLES "/workspace/code/motivo/testing.cmake;43;add_test;/workspace/code/motivo/testing.cmake;0;;/workspace/code/motivo/CMakeLists.txt;196;include;/workspace/code/motivo/CMakeLists.txt;0;")
subdirs("src/common")
subdirs("src/builder")
subdirs("src/merger")
subdirs("src/sampler")
subdirs("src/tools")
subdirs("src/tests")