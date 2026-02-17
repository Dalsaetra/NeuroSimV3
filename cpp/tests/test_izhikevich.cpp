// tests/test_izhikevich.cpp

#include <iostream>
#include "neuralsim/IzhikevichNeuron.h"

int main() {
    using ParamArray = neuralsim::NeuronState::ParamArray;

    // Create one neuron with example params (copied from neuron_templates “nb1” + extras)
    std::vector<ParamArray> params(1);
    params[0] = {
        0.3,   0.17,  5.0, 100.0, 20.0,   // k, a, b, d, C
       -66.0, -40.0, 50.0, -45.0,         // Vr, Vt, Vpeak, c
         2.5,    0.0, 1.05, 0.0           // delta_V, bias, threshold_mult, threshold_decay
    };

    // Instantiate with deterministic Euler stepping
    neuralsim::NeuronState neuron(params,
        neuralsim::NeuronState::StepperType::EulerDeterministic
    );

    // Apply a constant current, step once, and print state
    std::vector<double> I_ext(1, 10.0);
    double dt = 0.1;

    neuron.step(I_ext, dt);

    std::cout
        << "After one step:\n"
        << "  V = " << neuron.V()[0]
        << ", u = " << neuron.u()[0]
        << ", spike = " << neuron.spike()[0]
        << "\n";

    return 0;
}