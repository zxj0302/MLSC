# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /workspace/code/motivo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workspace/code/motivo/build

# Include any dependencies generated for this target.
include src/merger/CMakeFiles/motivo-merge.dir/depend.make

# Include the progress variables for this target.
include src/merger/CMakeFiles/motivo-merge.dir/progress.make

# Include the compile flags for this target's objects.
include src/merger/CMakeFiles/motivo-merge.dir/flags.make

src/merger/CMakeFiles/motivo-merge.dir/__/common/platform/platform.cpp.o: src/merger/CMakeFiles/motivo-merge.dir/flags.make
src/merger/CMakeFiles/motivo-merge.dir/__/common/platform/platform.cpp.o: ../src/common/platform/platform.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/code/motivo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/merger/CMakeFiles/motivo-merge.dir/__/common/platform/platform.cpp.o"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/motivo-merge.dir/__/common/platform/platform.cpp.o -c /workspace/code/motivo/src/common/platform/platform.cpp

src/merger/CMakeFiles/motivo-merge.dir/__/common/platform/platform.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/motivo-merge.dir/__/common/platform/platform.cpp.i"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/code/motivo/src/common/platform/platform.cpp > CMakeFiles/motivo-merge.dir/__/common/platform/platform.cpp.i

src/merger/CMakeFiles/motivo-merge.dir/__/common/platform/platform.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/motivo-merge.dir/__/common/platform/platform.cpp.s"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/code/motivo/src/common/platform/platform.cpp -o CMakeFiles/motivo-merge.dir/__/common/platform/platform.cpp.s

src/merger/CMakeFiles/motivo-merge.dir/__/common/graph/UndirectedGraph.cpp.o: src/merger/CMakeFiles/motivo-merge.dir/flags.make
src/merger/CMakeFiles/motivo-merge.dir/__/common/graph/UndirectedGraph.cpp.o: ../src/common/graph/UndirectedGraph.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/code/motivo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/merger/CMakeFiles/motivo-merge.dir/__/common/graph/UndirectedGraph.cpp.o"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/motivo-merge.dir/__/common/graph/UndirectedGraph.cpp.o -c /workspace/code/motivo/src/common/graph/UndirectedGraph.cpp

src/merger/CMakeFiles/motivo-merge.dir/__/common/graph/UndirectedGraph.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/motivo-merge.dir/__/common/graph/UndirectedGraph.cpp.i"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/code/motivo/src/common/graph/UndirectedGraph.cpp > CMakeFiles/motivo-merge.dir/__/common/graph/UndirectedGraph.cpp.i

src/merger/CMakeFiles/motivo-merge.dir/__/common/graph/UndirectedGraph.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/motivo-merge.dir/__/common/graph/UndirectedGraph.cpp.s"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/code/motivo/src/common/graph/UndirectedGraph.cpp -o CMakeFiles/motivo-merge.dir/__/common/graph/UndirectedGraph.cpp.s

src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/Treelet.cpp.o: src/merger/CMakeFiles/motivo-merge.dir/flags.make
src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/Treelet.cpp.o: ../src/common/treelets/Treelet.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/code/motivo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/Treelet.cpp.o"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/motivo-merge.dir/__/common/treelets/Treelet.cpp.o -c /workspace/code/motivo/src/common/treelets/Treelet.cpp

src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/Treelet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/motivo-merge.dir/__/common/treelets/Treelet.cpp.i"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/code/motivo/src/common/treelets/Treelet.cpp > CMakeFiles/motivo-merge.dir/__/common/treelets/Treelet.cpp.i

src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/Treelet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/motivo-merge.dir/__/common/treelets/Treelet.cpp.s"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/code/motivo/src/common/treelets/Treelet.cpp -o CMakeFiles/motivo-merge.dir/__/common/treelets/Treelet.cpp.s

src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTableCollection.cpp.o: src/merger/CMakeFiles/motivo-merge.dir/flags.make
src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTableCollection.cpp.o: ../src/common/treelets/TreeletTableCollection.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/code/motivo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTableCollection.cpp.o"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTableCollection.cpp.o -c /workspace/code/motivo/src/common/treelets/TreeletTableCollection.cpp

src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTableCollection.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTableCollection.cpp.i"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/code/motivo/src/common/treelets/TreeletTableCollection.cpp > CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTableCollection.cpp.i

src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTableCollection.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTableCollection.cpp.s"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/code/motivo/src/common/treelets/TreeletTableCollection.cpp -o CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTableCollection.cpp.s

src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTable.cpp.o: src/merger/CMakeFiles/motivo-merge.dir/flags.make
src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTable.cpp.o: ../src/common/treelets/TreeletTable.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/code/motivo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTable.cpp.o"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTable.cpp.o -c /workspace/code/motivo/src/common/treelets/TreeletTable.cpp

src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTable.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTable.cpp.i"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/code/motivo/src/common/treelets/TreeletTable.cpp > CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTable.cpp.i

src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTable.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTable.cpp.s"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/code/motivo/src/common/treelets/TreeletTable.cpp -o CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTable.cpp.s

src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletStructureSelector.cpp.o: src/merger/CMakeFiles/motivo-merge.dir/flags.make
src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletStructureSelector.cpp.o: ../src/common/treelets/TreeletStructureSelector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/code/motivo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletStructureSelector.cpp.o"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletStructureSelector.cpp.o -c /workspace/code/motivo/src/common/treelets/TreeletStructureSelector.cpp

src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletStructureSelector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletStructureSelector.cpp.i"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/code/motivo/src/common/treelets/TreeletStructureSelector.cpp > CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletStructureSelector.cpp.i

src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletStructureSelector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletStructureSelector.cpp.s"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/code/motivo/src/common/treelets/TreeletStructureSelector.cpp -o CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletStructureSelector.cpp.s

src/merger/CMakeFiles/motivo-merge.dir/__/common/io/ConcurrentWriter.cpp.o: src/merger/CMakeFiles/motivo-merge.dir/flags.make
src/merger/CMakeFiles/motivo-merge.dir/__/common/io/ConcurrentWriter.cpp.o: ../src/common/io/ConcurrentWriter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/code/motivo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object src/merger/CMakeFiles/motivo-merge.dir/__/common/io/ConcurrentWriter.cpp.o"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/motivo-merge.dir/__/common/io/ConcurrentWriter.cpp.o -c /workspace/code/motivo/src/common/io/ConcurrentWriter.cpp

src/merger/CMakeFiles/motivo-merge.dir/__/common/io/ConcurrentWriter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/motivo-merge.dir/__/common/io/ConcurrentWriter.cpp.i"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/code/motivo/src/common/io/ConcurrentWriter.cpp > CMakeFiles/motivo-merge.dir/__/common/io/ConcurrentWriter.cpp.i

src/merger/CMakeFiles/motivo-merge.dir/__/common/io/ConcurrentWriter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/motivo-merge.dir/__/common/io/ConcurrentWriter.cpp.s"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/code/motivo/src/common/io/ConcurrentWriter.cpp -o CMakeFiles/motivo-merge.dir/__/common/io/ConcurrentWriter.cpp.s

src/merger/CMakeFiles/motivo-merge.dir/__/common/io/CompressedRecordFile.cpp.o: src/merger/CMakeFiles/motivo-merge.dir/flags.make
src/merger/CMakeFiles/motivo-merge.dir/__/common/io/CompressedRecordFile.cpp.o: ../src/common/io/CompressedRecordFile.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/code/motivo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object src/merger/CMakeFiles/motivo-merge.dir/__/common/io/CompressedRecordFile.cpp.o"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/motivo-merge.dir/__/common/io/CompressedRecordFile.cpp.o -c /workspace/code/motivo/src/common/io/CompressedRecordFile.cpp

src/merger/CMakeFiles/motivo-merge.dir/__/common/io/CompressedRecordFile.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/motivo-merge.dir/__/common/io/CompressedRecordFile.cpp.i"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/code/motivo/src/common/io/CompressedRecordFile.cpp > CMakeFiles/motivo-merge.dir/__/common/io/CompressedRecordFile.cpp.i

src/merger/CMakeFiles/motivo-merge.dir/__/common/io/CompressedRecordFile.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/motivo-merge.dir/__/common/io/CompressedRecordFile.cpp.s"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/code/motivo/src/common/io/CompressedRecordFile.cpp -o CMakeFiles/motivo-merge.dir/__/common/io/CompressedRecordFile.cpp.s

src/merger/CMakeFiles/motivo-merge.dir/__/common/io/RecordCompressor.cpp.o: src/merger/CMakeFiles/motivo-merge.dir/flags.make
src/merger/CMakeFiles/motivo-merge.dir/__/common/io/RecordCompressor.cpp.o: ../src/common/io/RecordCompressor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/code/motivo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object src/merger/CMakeFiles/motivo-merge.dir/__/common/io/RecordCompressor.cpp.o"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/motivo-merge.dir/__/common/io/RecordCompressor.cpp.o -c /workspace/code/motivo/src/common/io/RecordCompressor.cpp

src/merger/CMakeFiles/motivo-merge.dir/__/common/io/RecordCompressor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/motivo-merge.dir/__/common/io/RecordCompressor.cpp.i"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/code/motivo/src/common/io/RecordCompressor.cpp > CMakeFiles/motivo-merge.dir/__/common/io/RecordCompressor.cpp.i

src/merger/CMakeFiles/motivo-merge.dir/__/common/io/RecordCompressor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/motivo-merge.dir/__/common/io/RecordCompressor.cpp.s"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/code/motivo/src/common/io/RecordCompressor.cpp -o CMakeFiles/motivo-merge.dir/__/common/io/RecordCompressor.cpp.s

src/merger/CMakeFiles/motivo-merge.dir/__/common/io/PropertyStore.cpp.o: src/merger/CMakeFiles/motivo-merge.dir/flags.make
src/merger/CMakeFiles/motivo-merge.dir/__/common/io/PropertyStore.cpp.o: ../src/common/io/PropertyStore.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/code/motivo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object src/merger/CMakeFiles/motivo-merge.dir/__/common/io/PropertyStore.cpp.o"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/motivo-merge.dir/__/common/io/PropertyStore.cpp.o -c /workspace/code/motivo/src/common/io/PropertyStore.cpp

src/merger/CMakeFiles/motivo-merge.dir/__/common/io/PropertyStore.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/motivo-merge.dir/__/common/io/PropertyStore.cpp.i"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/code/motivo/src/common/io/PropertyStore.cpp > CMakeFiles/motivo-merge.dir/__/common/io/PropertyStore.cpp.i

src/merger/CMakeFiles/motivo-merge.dir/__/common/io/PropertyStore.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/motivo-merge.dir/__/common/io/PropertyStore.cpp.s"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/code/motivo/src/common/io/PropertyStore.cpp -o CMakeFiles/motivo-merge.dir/__/common/io/PropertyStore.cpp.s

src/merger/CMakeFiles/motivo-merge.dir/__/common/OptionsParser.cpp.o: src/merger/CMakeFiles/motivo-merge.dir/flags.make
src/merger/CMakeFiles/motivo-merge.dir/__/common/OptionsParser.cpp.o: ../src/common/OptionsParser.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/code/motivo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object src/merger/CMakeFiles/motivo-merge.dir/__/common/OptionsParser.cpp.o"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/motivo-merge.dir/__/common/OptionsParser.cpp.o -c /workspace/code/motivo/src/common/OptionsParser.cpp

src/merger/CMakeFiles/motivo-merge.dir/__/common/OptionsParser.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/motivo-merge.dir/__/common/OptionsParser.cpp.i"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/code/motivo/src/common/OptionsParser.cpp > CMakeFiles/motivo-merge.dir/__/common/OptionsParser.cpp.i

src/merger/CMakeFiles/motivo-merge.dir/__/common/OptionsParser.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/motivo-merge.dir/__/common/OptionsParser.cpp.s"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/code/motivo/src/common/OptionsParser.cpp -o CMakeFiles/motivo-merge.dir/__/common/OptionsParser.cpp.s

src/merger/CMakeFiles/motivo-merge.dir/__/common/util.cpp.o: src/merger/CMakeFiles/motivo-merge.dir/flags.make
src/merger/CMakeFiles/motivo-merge.dir/__/common/util.cpp.o: ../src/common/util.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/code/motivo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object src/merger/CMakeFiles/motivo-merge.dir/__/common/util.cpp.o"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/motivo-merge.dir/__/common/util.cpp.o -c /workspace/code/motivo/src/common/util.cpp

src/merger/CMakeFiles/motivo-merge.dir/__/common/util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/motivo-merge.dir/__/common/util.cpp.i"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/code/motivo/src/common/util.cpp > CMakeFiles/motivo-merge.dir/__/common/util.cpp.i

src/merger/CMakeFiles/motivo-merge.dir/__/common/util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/motivo-merge.dir/__/common/util.cpp.s"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/code/motivo/src/common/util.cpp -o CMakeFiles/motivo-merge.dir/__/common/util.cpp.s

src/merger/CMakeFiles/motivo-merge.dir/merger.cpp.o: src/merger/CMakeFiles/motivo-merge.dir/flags.make
src/merger/CMakeFiles/motivo-merge.dir/merger.cpp.o: ../src/merger/merger.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/code/motivo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object src/merger/CMakeFiles/motivo-merge.dir/merger.cpp.o"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/motivo-merge.dir/merger.cpp.o -c /workspace/code/motivo/src/merger/merger.cpp

src/merger/CMakeFiles/motivo-merge.dir/merger.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/motivo-merge.dir/merger.cpp.i"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/code/motivo/src/merger/merger.cpp > CMakeFiles/motivo-merge.dir/merger.cpp.i

src/merger/CMakeFiles/motivo-merge.dir/merger.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/motivo-merge.dir/merger.cpp.s"
	cd /workspace/code/motivo/build/src/merger && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/code/motivo/src/merger/merger.cpp -o CMakeFiles/motivo-merge.dir/merger.cpp.s

# Object files for target motivo-merge
motivo__merge_OBJECTS = \
"CMakeFiles/motivo-merge.dir/__/common/platform/platform.cpp.o" \
"CMakeFiles/motivo-merge.dir/__/common/graph/UndirectedGraph.cpp.o" \
"CMakeFiles/motivo-merge.dir/__/common/treelets/Treelet.cpp.o" \
"CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTableCollection.cpp.o" \
"CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTable.cpp.o" \
"CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletStructureSelector.cpp.o" \
"CMakeFiles/motivo-merge.dir/__/common/io/ConcurrentWriter.cpp.o" \
"CMakeFiles/motivo-merge.dir/__/common/io/CompressedRecordFile.cpp.o" \
"CMakeFiles/motivo-merge.dir/__/common/io/RecordCompressor.cpp.o" \
"CMakeFiles/motivo-merge.dir/__/common/io/PropertyStore.cpp.o" \
"CMakeFiles/motivo-merge.dir/__/common/OptionsParser.cpp.o" \
"CMakeFiles/motivo-merge.dir/__/common/util.cpp.o" \
"CMakeFiles/motivo-merge.dir/merger.cpp.o"

# External object files for target motivo-merge
motivo__merge_EXTERNAL_OBJECTS =

bin/motivo-merge: src/merger/CMakeFiles/motivo-merge.dir/__/common/platform/platform.cpp.o
bin/motivo-merge: src/merger/CMakeFiles/motivo-merge.dir/__/common/graph/UndirectedGraph.cpp.o
bin/motivo-merge: src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/Treelet.cpp.o
bin/motivo-merge: src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTableCollection.cpp.o
bin/motivo-merge: src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletTable.cpp.o
bin/motivo-merge: src/merger/CMakeFiles/motivo-merge.dir/__/common/treelets/TreeletStructureSelector.cpp.o
bin/motivo-merge: src/merger/CMakeFiles/motivo-merge.dir/__/common/io/ConcurrentWriter.cpp.o
bin/motivo-merge: src/merger/CMakeFiles/motivo-merge.dir/__/common/io/CompressedRecordFile.cpp.o
bin/motivo-merge: src/merger/CMakeFiles/motivo-merge.dir/__/common/io/RecordCompressor.cpp.o
bin/motivo-merge: src/merger/CMakeFiles/motivo-merge.dir/__/common/io/PropertyStore.cpp.o
bin/motivo-merge: src/merger/CMakeFiles/motivo-merge.dir/__/common/OptionsParser.cpp.o
bin/motivo-merge: src/merger/CMakeFiles/motivo-merge.dir/__/common/util.cpp.o
bin/motivo-merge: src/merger/CMakeFiles/motivo-merge.dir/merger.cpp.o
bin/motivo-merge: src/merger/CMakeFiles/motivo-merge.dir/build.make
bin/motivo-merge: /usr/lib/x86_64-linux-gnu/liblz4.so
bin/motivo-merge: src/merger/CMakeFiles/motivo-merge.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/code/motivo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Linking CXX executable ../../bin/motivo-merge"
	cd /workspace/code/motivo/build/src/merger && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/motivo-merge.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/merger/CMakeFiles/motivo-merge.dir/build: bin/motivo-merge

.PHONY : src/merger/CMakeFiles/motivo-merge.dir/build

src/merger/CMakeFiles/motivo-merge.dir/clean:
	cd /workspace/code/motivo/build/src/merger && $(CMAKE_COMMAND) -P CMakeFiles/motivo-merge.dir/cmake_clean.cmake
.PHONY : src/merger/CMakeFiles/motivo-merge.dir/clean

src/merger/CMakeFiles/motivo-merge.dir/depend:
	cd /workspace/code/motivo/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/code/motivo /workspace/code/motivo/src/merger /workspace/code/motivo/build /workspace/code/motivo/build/src/merger /workspace/code/motivo/build/src/merger/CMakeFiles/motivo-merge.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/merger/CMakeFiles/motivo-merge.dir/depend

