# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/server10/.local/lib/python3.8/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/server10/.local/lib/python3.8/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/server10/Synthetic_3D_Object_Generation

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/server10/Synthetic_3D_Object_Generation/build

# Include any dependencies generated for this target.
include patchworkpp/CMakeFiles/patchworkpp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include patchworkpp/CMakeFiles/patchworkpp.dir/compiler_depend.make

# Include the progress variables for this target.
include patchworkpp/CMakeFiles/patchworkpp.dir/progress.make

# Include the compile flags for this target's objects.
include patchworkpp/CMakeFiles/patchworkpp.dir/flags.make

patchworkpp/CMakeFiles/patchworkpp.dir/src/patchworkpp.cpp.o: patchworkpp/CMakeFiles/patchworkpp.dir/flags.make
patchworkpp/CMakeFiles/patchworkpp.dir/src/patchworkpp.cpp.o: ../patchworkpp/src/patchworkpp.cpp
patchworkpp/CMakeFiles/patchworkpp.dir/src/patchworkpp.cpp.o: patchworkpp/CMakeFiles/patchworkpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/server10/Synthetic_3D_Object_Generation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object patchworkpp/CMakeFiles/patchworkpp.dir/src/patchworkpp.cpp.o"
	cd /home/server10/Synthetic_3D_Object_Generation/build/patchworkpp && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT patchworkpp/CMakeFiles/patchworkpp.dir/src/patchworkpp.cpp.o -MF CMakeFiles/patchworkpp.dir/src/patchworkpp.cpp.o.d -o CMakeFiles/patchworkpp.dir/src/patchworkpp.cpp.o -c /home/server10/Synthetic_3D_Object_Generation/patchworkpp/src/patchworkpp.cpp

patchworkpp/CMakeFiles/patchworkpp.dir/src/patchworkpp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/patchworkpp.dir/src/patchworkpp.cpp.i"
	cd /home/server10/Synthetic_3D_Object_Generation/build/patchworkpp && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/server10/Synthetic_3D_Object_Generation/patchworkpp/src/patchworkpp.cpp > CMakeFiles/patchworkpp.dir/src/patchworkpp.cpp.i

patchworkpp/CMakeFiles/patchworkpp.dir/src/patchworkpp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/patchworkpp.dir/src/patchworkpp.cpp.s"
	cd /home/server10/Synthetic_3D_Object_Generation/build/patchworkpp && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/server10/Synthetic_3D_Object_Generation/patchworkpp/src/patchworkpp.cpp -o CMakeFiles/patchworkpp.dir/src/patchworkpp.cpp.s

# Object files for target patchworkpp
patchworkpp_OBJECTS = \
"CMakeFiles/patchworkpp.dir/src/patchworkpp.cpp.o"

# External object files for target patchworkpp
patchworkpp_EXTERNAL_OBJECTS =

patchworkpp/libpatchworkpp.so: patchworkpp/CMakeFiles/patchworkpp.dir/src/patchworkpp.cpp.o
patchworkpp/libpatchworkpp.so: patchworkpp/CMakeFiles/patchworkpp.dir/build.make
patchworkpp/libpatchworkpp.so: patchworkpp/CMakeFiles/patchworkpp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/server10/Synthetic_3D_Object_Generation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libpatchworkpp.so"
	cd /home/server10/Synthetic_3D_Object_Generation/build/patchworkpp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/patchworkpp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
patchworkpp/CMakeFiles/patchworkpp.dir/build: patchworkpp/libpatchworkpp.so
.PHONY : patchworkpp/CMakeFiles/patchworkpp.dir/build

patchworkpp/CMakeFiles/patchworkpp.dir/clean:
	cd /home/server10/Synthetic_3D_Object_Generation/build/patchworkpp && $(CMAKE_COMMAND) -P CMakeFiles/patchworkpp.dir/cmake_clean.cmake
.PHONY : patchworkpp/CMakeFiles/patchworkpp.dir/clean

patchworkpp/CMakeFiles/patchworkpp.dir/depend:
	cd /home/server10/Synthetic_3D_Object_Generation/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/server10/Synthetic_3D_Object_Generation /home/server10/Synthetic_3D_Object_Generation/patchworkpp /home/server10/Synthetic_3D_Object_Generation/build /home/server10/Synthetic_3D_Object_Generation/build/patchworkpp /home/server10/Synthetic_3D_Object_Generation/build/patchworkpp/CMakeFiles/patchworkpp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : patchworkpp/CMakeFiles/patchworkpp.dir/depend

