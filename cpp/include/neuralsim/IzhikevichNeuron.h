#pragma once

#include <vector>
#include <array>
#include <cstddef>
#include <random>
#include <stdexcept>

namespace neuralsim {

/**
 * NeuronState – state-holder + integrator for a population of independent
 * Izhikevich neurons (one instance manages <em>all</em> neurons).
 *
 * Parameter order per-neuron follows the Python implementation:
 *   k, a, b, d, C, Vr, Vt, Vpeak, c, delta_V, bias, threshold_mult, threshold_decay
 * (13 values total).
 */
class NeuronState {
public:
    /** Which numerical scheme + stochasticity to use. */
    enum class StepperType {
        Euler,
        EulerDeterministic,
        Adapt,
        AdaptDeterministic
    };

    using ParamArray = std::array<double, 13>;

    /**
     * Construct from a per-neuron parameter table.
     *
     * @param params   vector of ParamArray – one entry per neuron
     * @param stepper  integration / spiking scheme to use (default = Adapt)
     * @param state0   optional pointer to another NeuronState to copy
     */
    explicit NeuronState(const std::vector<ParamArray>& params,
                         StepperType stepper = StepperType::Adapt,
                         const NeuronState* state0 = nullptr);

    /** Advance all neurons one time-step. @param I external / synaptic current. */
    void step(const std::vector<double>& I, double dt);

    // ---------- read-only accessors ----------
    const std::vector<double>& V()     const noexcept { return V_; }
    const std::vector<double>& u()     const noexcept { return u_; }
    const std::vector<bool>&  spike() const noexcept { return spike_; }
    const std::vector<double>& T()     const noexcept { return T_; }
    std::size_t size()            const noexcept { return n_; }

private:
    std::size_t n_ = 0;  ///< number of neurons

    // --- parameters ---------------------------------------------------------
    std::vector<double> k_, a_, b_, d_, C_, Vr_, Vt_, Vpeak_, c_;
    std::vector<double> delta_V_, bias_, threshold_mult_, threshold_decay_;

    // --- dynamic state ------------------------------------------------------
    std::vector<double> V_, u_, T_;
    std::vector<bool>   spike_;

    // --- random number generation ------------------------------------------
    std::mt19937                                  rng_{std::random_device{}()};
    std::uniform_real_distribution<double>       uni_{0.0, 1.0};

    // --- internal dispatch --------------------------------------------------
    using StepFn = void (NeuronState::*)(const std::vector<double>&, double);
    StepFn step_fn_ = nullptr;

    // --- stepping routines (defined in .cpp) --------------------------------
    void step_euler(const std::vector<double>& I, double dt);
    void step_euler_det(const std::vector<double>& I, double dt);
    void step_adapt(const std::vector<double>& I, double dt);
    void step_adapt_det(const std::vector<double>& I, double dt);
};

} // namespace neuralsim
