# Converts a SPIR-V binary file to a C header with a uint32_t array.
# INPUT  – path to .spv
# OUTPUT – path to .inc to write
# VAR    – array symbol name
file(READ "${INPUT}" BIN HEX)
string(REGEX MATCHALL ".." BYTES "${BIN}")

set(OUT "/* auto-generated – do not edit */\n#include <cstdint>\n")
set(OUT "${OUT}alignas(4) const uint32_t ${VAR}[] = {\n")

set(LINE "")
foreach(b ${BYTES})
    set(LINE "${LINE}0x${b},")
    string(LENGTH "${LINE}" L)
    if(L GREATER 70)
        set(OUT "${OUT}    ${LINE}\n")
        set(LINE "")
    endif()
endforeach()
if(NOT LINE STREQUAL "")
    set(OUT "${OUT}    ${LINE}\n")
endif()

set(OUT "${OUT}};\n")
file(WRITE "${OUTPUT}" "${OUT}")
