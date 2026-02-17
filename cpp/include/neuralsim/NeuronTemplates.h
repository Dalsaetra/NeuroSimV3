#pragma once

#include <array>
#include <string>
#include <unordered_map>

namespace neuralsim {

/// Izhekevich neuron templates:
/// Each array is { k, a, b, d, C, Vr, Vt, Vpeak, c }
static const std::unordered_map<std::string, std::array<double,9>> neuron_type_IZ = {
    { "nb1",   { 0.3,   0.17,  5.0, 100.0,  20.0, -66.0, -40.0, 50.0, -45.0 } },
    { "p23",   { 3.0,   0.01,  5.0, 400.0, 100.0, -60.0, -50.0, 40.0, -57.5 } },
    { "b",     { 1.0,   0.15,  8.0, 200.0,  20.0, -55.0, -40.0, 25.0, -55.0 } },
    { "nb",    { 1.0,   0.03,  8.0,  20.0, 100.0, -56.0, -42.0, 40.0, -50.0 } },
    { "ss4",   { 3.0,   0.01,  5.0, 400.0, 100.0, -60.0, -50.0, 40.0, -55.0 } },
    { "p4",    { 3.0,   0.01,  5.0, 400.0, 100.0, -60.0, -50.0, 50.0, -55.0 } },
    { "p5_p6", { 3.0,   0.01,  5.0, 400.0, 100.0, -60.0, -50.0, 40.0, -55.0 } },
    { "TC",    { 1.6,   0.10, 15.0,  10.0, 200.0, -60.0, -50.0, 40.0, -60.0 } },
    { "TI",    { 0.5,   0.05,  7.0,  50.0,  20.0, -60.0, -50.0, 20.0, -65.0 } },
    { "TRN",   { 0.25, 0.015, 10.0,  50.0,  40.0, -60.0, -50.0,  0.0, -55.0 } },
    // …and so on for all of the 9-element templates in neuron_templates.py :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
};

} // namespace neuralsim