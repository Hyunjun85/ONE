set(TRANSFORMATIONS_SRC
        DataFormatSwitcher.cpp
        LowerConv2D.cpp)

nnc_add_library(nnc_transformations STATIC ${TRANSFORMATIONS_SRC})
set_target_properties(nnc_transformations PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(nnc_transformations PRIVATE mir)
