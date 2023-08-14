# This file is part of OpenKalman, a header-only C++ library for
# Kalman filters and other recursive filters.
#
# Created by Christopher Lee Ogden <ogden@gatech.edu> in 2017.
# Any copyright as to this file is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

set(MyDoc_FLAGS "-Wdocumentation")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MyDoc_FLAGS}")


find_package(Doxygen)
if (DOXYGEN_FOUND)
    set(DOXYGEN_REPEAT_BRIEF NO)
    set(DOXYGEN_INLINE_INHERITED_MEMB YES)
    set(DOXYGEN_TAB_SIZE 2)
    set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
    #set(DOXYGEN_DISTRIBUTE_GROUP_DOC YES)
    set(DOXYGEN_GROUP_NESTED_COMPOUNDS YES)
    set(DOXYGEN_INLINE_GROUPED_CLASSES YES)
    set(DOXYGEN_NUM_PROC_THREADS 0)
    set(DOXYGEN_EXTRACT_LOCAL_CLASSES NO)
    set(DOXYGEN_HIDE_UNDOC_MEMBERS YES)
    set(DOXYGEN_HIDE_UNDOC_CLASSES YES)
    set(DOXYGEN_GENERATE_TODOLIST NO)
    set(DOXYGEN_GENERATE_TESTLIST NO)
    set(DOXYGEN_GENERATE_BUGLIST NO)
    set(DOXYGEN_GENERATE_DEPRECATEDLIST NO)
    set(DOXYGEN_LAYOUT_FILE doxygen_layout.xml)
    set(DOXYGEN_SOURCE_BROWSER YES)
    #set(DOXYGEN_CLANG_ASSISTED_PARSING YES)
    #set(DOXYGEN_CLANG_OPTIONS -std=c++20)
    #set(DOXYGEN_HTML_STYLESHEET XXX)
    set(DOXYGEN_HTML_TIMESTAMP YES)
    set(DOXYGEN_GENERATE_TREEVIEW YES)
    set(DOXYGEN_GENERATE_LATEX NO)
    set(DOXYGEN_PREDEFINED "__cpp_concepts=201907L __cpp_nontype_template_args=201911L __clang__")

    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen)
    set(DOXYGEN_INTERNAL_DOCS NO)

    doxygen_add_docs(generate_doxygen ../OpenKalman COMMENT "Generating API documentation")

    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/development)
    set(DOXYGEN_INTERNAL_DOCS YES)

    doxygen_add_docs(generate_doxygen_dev ../OpenKalman COMMENT "Generating API documentation for developers")

else()
    message("Doxygen must be installed to generate the Doxygen documentation.")

endif()
