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
CMAKE_SOURCE_DIR = /workspace/LPP-code/src/gmatch/subcounting/SubgraphMatching

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workspace/LPP-code/src/gmatch/subcounting/SubgraphMatching/build

# Include any dependencies generated for this target.
include matching/CMakeFiles/EdgeListConverter.out.dir/depend.make

# Include the progress variables for this target.
include matching/CMakeFiles/EdgeListConverter.out.dir/progress.make

# Include the compile flags for this target's objects.
include matching/CMakeFiles/EdgeListConverter.out.dir/flags.make

matching/CMakeFiles/EdgeListConverter.out.dir/EdgeListToCSR.cpp.o: matching/CMakeFiles/EdgeListConverter.out.dir/flags.make
matching/CMakeFiles/EdgeListConverter.out.dir/EdgeListToCSR.cpp.o: ../matching/EdgeListToCSR.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/LPP-code/src/gmatch/subcounting/SubgraphMatching/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object matching/CMakeFiles/EdgeListConverter.out.dir/EdgeListToCSR.cpp.o"
	cd /workspace/LPP-code/src/gmatch/subcounting/SubgraphMatching/build/matching && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/EdgeListConverter.out.dir/EdgeListToCSR.cpp.o -c /workspace/LPP-code/src/gmatch/subcounting/SubgraphMatching/matching/EdgeListToCSR.cpp

matching/CMakeFiles/EdgeListConverter.out.dir/EdgeListToCSR.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/EdgeListConverter.out.dir/EdgeListToCSR.cpp.i"
	cd /workspace/LPP-code/src/gmatch/subcounting/SubgraphMatching/build/matching && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/LPP-code/src/gmatch/subcounting/SubgraphMatching/matching/EdgeListToCSR.cpp > CMakeFiles/EdgeListConverter.out.dir/EdgeListToCSR.cpp.i

matching/CMakeFiles/EdgeListConverter.out.dir/EdgeListToCSR.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/EdgeListConverter.out.dir/EdgeListToCSR.cpp.s"
	cd /workspace/LPP-code/src/gmatch/subcounting/SubgraphMatching/build/matching && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/LPP-code/src/gmatch/subcounting/SubgraphMatching/matching/EdgeListToCSR.cpp -o CMakeFiles/EdgeListConverter.out.dir/EdgeListToCSR.cpp.s

# Object files for target EdgeListConverter.out
EdgeListConverter_out_OBJECTS = \
"CMakeFiles/EdgeListConverter.out.dir/EdgeListToCSR.cpp.o"

# External object files for target EdgeListConverter.out
EdgeListConverter_out_EXTERNAL_OBJECTS =

matching/EdgeListConverter.out: matching/CMakeFiles/EdgeListConverter.out.dir/EdgeListToCSR.cpp.o
matching/EdgeListConverter.out: matching/CMakeFiles/EdgeListConverter.out.dir/build.make
matching/EdgeListConverter.out: matching/CMakeFiles/EdgeListConverter.out.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/LPP-code/src/gmatch/subcounting/SubgraphMatching/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable EdgeListConverter.out"
	cd /workspace/LPP-code/src/gmatch/subcounting/SubgraphMatching/build/matching && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/EdgeListConverter.out.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
matching/CMakeFiles/EdgeListConverter.out.dir/build: matching/EdgeListConverter.out

.PHONY : matching/CMakeFiles/EdgeListConverter.out.dir/build

matching/CMakeFiles/EdgeListConverter.out.dir/clean:
	cd /workspace/LPP-code/src/gmatch/subcounting/SubgraphMatching/build/matching && $(CMAKE_COMMAND) -P CMakeFiles/EdgeListConverter.out.dir/cmake_clean.cmake
.PHONY : matching/CMakeFiles/EdgeListConverter.out.dir/clean

matching/CMakeFiles/EdgeListConverter.out.dir/depend:
	cd /workspace/LPP-code/src/gmatch/subcounting/SubgraphMatching/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/LPP-code/src/gmatch/subcounting/SubgraphMatching /workspace/LPP-code/src/gmatch/subcounting/SubgraphMatching/matching /workspace/LPP-code/src/gmatch/subcounting/SubgraphMatching/build /workspace/LPP-code/src/gmatch/subcounting/SubgraphMatching/build/matching /workspace/LPP-code/src/gmatch/subcounting/SubgraphMatching/build/matching/CMakeFiles/EdgeListConverter.out.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : matching/CMakeFiles/EdgeListConverter.out.dir/depend

