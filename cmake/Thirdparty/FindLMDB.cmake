
FIND_PATH(LMDB_INCLUDE_DIR NAMES lmdb.h PATHS "$ENV{LMDB_DIR}/include")
FIND_LIBRARY(LMDB_LIBRARIES NAMES lmdb PATHS "$ENV{LMDB_DIR}/include")

INCLUDE(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LMDB DEFAULT_MSG LMDB_INCLUDE_DIR LMDB_LIBRARIES)

IF(LMDB_FOUND)
    MESSAGE(STATUS "Found lmdb at ${LMDB_INCLUDE_DIR}")
    MARK_AS_ADVANCED(LMDB_INCLUDE_DIR LMDB_LIBRARIES)
    
ENDIF()