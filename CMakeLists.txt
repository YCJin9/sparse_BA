cmake_minimum_required( VERSION 3.10 )
project( sparse_BA )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# opencv
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# eigen
find_package( Eigen3 3.1.0 REQUIRED )
include_directories( ${Eigen3_INCLUDE_DIRS} )

# sophus
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
add_executable( GN GN.cpp)
target_link_libraries( GN ${OpenCV_LIBS}  )

add_executable( solvePnP solvePnP.cpp)
target_link_libraries( solvePnP ${OpenCV_LIBS} ${Sophus_LIBRARIES} )
