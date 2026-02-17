#include "neuralsim/IzhikevichNeuron.h"

#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace neuralsim {

NeuronState::NeuronState(const std::vector<ParamArray>& params,
                                StepperType stepper,
                                const NeuronState* state0)
    : n_(params.size()),
      k_(n_), a_(n_), b_(n_), d_(n_), C_(n_), Vr_(n_), Vt_(n_), Vpeak_(n_), c_(n_),
      delta_V_(n_), bias_(n_), threshold_mult_(n_), threshold_decay_(n_),
      V_(n_), u_(n_), T_(n_), spike_(n_, false) {

    // copy parameter table into individual vectors (SoA layout)
    for (std::size_t i = 0; i < n_; ++i) {
        const auto& p = params[i];
        k_[i]              = p[0];
        a_[i]              = p[1];
        b_[i]              = p[2];
        d_[i]              = p[3];
        C_[i]              = p[4];
        Vr_[i]             = p[5];
        Vt_[i]             = p[6];
        Vpeak_[i]          = p[7];
        c_[i]              = p[8];
        delta_V_[i]        = p[9];
        bias_[i]           = p[10];
        threshold_mult_[i] = p[11];
        threshold_decay_[i]= p[12];

        // default state
        V_[i] = Vr_[i];
        u_[i] = 0.0;
        T_[i] = 0.0;
    }

    // overwrite with provided state0 (if any)
    if (state0) {
        V_     = state0->V_;
        u_     = state0->u_;
        T_     = state0->T_;
        spike_ = state0->spike_;
    }

    // select integration routine
    switch (stepper) {
        case StepperType::Euler:              step_fn_ = &NeuronState::step_euler;            break;
        case StepperType::EulerDeterministic: step_fn_ = &NeuronState::step_euler_det;        break;
        case StepperType::Adapt:              step_fn_ = &NeuronState::step_adapt;            break;
        case StepperType::AdaptDeterministic: step_fn_ = &NeuronState::step_adapt_det;        break;
    }
}

void NeuronState::step(const std::vector<double>& I, double dt) {
    if (I.size() != n_) throw std::runtime_error("I size mismatch");
    (this->*step_fn_)(I, dt);
}

// ---------------------------------------------------------------------------
void NeuronState::step_euler(const std::vector<double>& I, double dt) {
    for (std::size_t i = 0; i < n_; ++i) {
        double dV = (k_[i] * (V_[i] - Vr_[i]) * (V_[i] - Vt_[i]) - u_[i] + I[i] + bias_[i]) / C_[i];
        double du = a_[i] * (b_[i] * (V_[i] - Vr_[i]) - u_[i]);

        V_[i] += dt * dV;
        u_[i] += dt * du;

        double spike_prob = dt * std::exp( (V_[i] - Vpeak_[i]) / delta_V_[i] );
        bool s = uni_(rng_) < spike_prob;

        if (s) {
            V_[i] = c_[i];
            u_[i] += d_[i];
        }
        spike_[i] = s;
    }
}

// ---------------------------------------------------------------------------
void NeuronState::step_euler_det(const std::vector<double>& I, double dt) {
    for (std::size_t i = 0; i < n_; ++i) {
        double dV = (k_[i] * (V_[i] - Vr_[i]) * (V_[i] - Vt_[i]) - u_[i] + I[i] + bias_[i]) / C_[i];
        double du = a_[i] * (b_[i] * (V_[i] - Vr_[i]) - u_[i]);

        V_[i] += dt * dV;
        u_[i] += dt * du;

        bool s = V_[i] >= Vpeak_[i];
        if (s) {
            V_[i] = c_[i];
            u_[i] += d_[i];
        }
        spike_[i] = s;
    }
}

// ---------------------------------------------------------------------------
void NeuronState::step_adapt(const std::vector<double>& I, double dt) {
    for (std::size_t i = 0; i < n_; ++i) {
        double dV = (k_[i] * (V_[i] - Vr_[i]) * (V_[i] - Vt_[i]) - u_[i] + I[i] + bias_[i]) / C_[i];
        double du = a_[i] * (b_[i] * (V_[i] - Vr_[i]) - u_[i]);

        V_[i] += dt * dV;
        u_[i] += dt * du;

        double eff_thresh = Vpeak_[i] + T_[i];
        T_[i] *= threshold_decay_[i];

        double spike_prob = dt * std::exp( (V_[i] - eff_thresh) / delta_V_[i] );
        bool s = uni_(rng_) < spike_prob;

        if (s) {
            T_[i] *= threshold_mult_[i];
            V_[i] = c_[i];
            u_[i] += d_[i];
        }
        spike_[i] = s;
    }
}

// ---------------------------------------------------------------------------
void NeuronState::step_adapt_det(const std::vector<double>& I, double dt) {
    for (std::size_t i = 0; i < n_; ++i) {
        double dV = (k_[i] * (V_[i] - Vr_[i]) * (V_[i] - Vt_[i]) - u_[i] + I[i] + bias_[i]) / C_[i];
        double du = a_[i] * (b_[i] * (V_[i] - Vr_[i]) - u_[i]);

        V_[i] += dt * dV;
        u_[i] += dt * du;

        double eff_thresh = Vpeak_[i] + T_[i];
        T_[i] *= threshold_decay_[i];

        bool s = V_[i] >= eff_thresh;
        if (s) {
            T_[i] = eff_thresh * threshold_mult_[i] - Vpeak_[i];
            V_[i] = c_[i];
            u_[i] += d_[i];
        }
        spike_[i] = s;
    }
}

} // namespace neuralsim