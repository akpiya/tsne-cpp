include(AddCatch2)


add_executable(unit_tests test.cpp)
target_link_libraries(unit_tests PRIVATE algo Catch2::Catch2WithMain)
target_include_directories(unit_tests PUBLIC "${PROJECT_SOURCE_DIR}/src")


include(CTest)
include(Catch)
catch_discover_tests(unit_tests)