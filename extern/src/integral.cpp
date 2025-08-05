#include "integral.h"


dbl energy(dbl y, dbl mass=0) {
    return sqrt(y*y + mass*mass);
}


std::pair<int, int> binary_find(const std::vector<dbl> &grid, dbl x) {
    int head(0), tail(grid.size() - 1);
    int middle;
    // Returns to keep p within the grid bounds
    if (grid[tail] < x) {
        return std::make_pair(tail, -1);
    }
    if (grid[head] > x) {
        return std::make_pair(-1, head);
    }

    while (tail - head > 1) {
        middle = (tail + head) / 2;
        if (grid[middle] > x) {
            tail = middle;
        } else {
            head = middle;
        }
    }

    if (grid[tail] == x) {
        return std::make_pair(tail, tail);
    }

    if (grid[head] == x) {
        return std::make_pair(head, head);
    }
    // the index of the grid that has momentum point p
    return std::make_pair(head, tail);
}


dbl distribution_interpolation(const std::vector<dbl> &grid,
                               const std::vector<dbl> &distribution,
                               dbl p, dbl m=0., int eta=1, dbl T=1.,
                               bool in_equilibrium=false) {

    if (in_equilibrium) {
        return 1. / (
           exp(energy(p, m) / T)
           + eta
       );
    }

    int i_lo, i_hi;
    std::tie(i_lo, i_hi) = binary_find(grid, p);
    if(i_lo == -1) {
        throw std::runtime_error("Input momentum is too small for the given grid");
    }
    // p greater than the grid, interpolates to higher p
    if(i_hi == -1) {
        return distribution[grid.size()-1]
            * exp((energy(grid[grid.size() -1], m) - energy(p, m)) / T);
    }
    // if p is in grid then return from pre-calculated distribution function
    if(i_lo == i_hi) {
        return distribution[i_lo];
    }

    // Interpolating the distribution function if grid is not fine enough
    // using exponential interpolation
    dbl p_lo = grid[i_lo],
        p_hi = grid[i_hi];

    // === Exponential interpolation ===
    dbl E_p  = energy(p, m),
        E_lo = energy(p_lo, m),
        E_hi = energy(p_hi, m);

    /*
    \begin{equation}
        g = \frac{ (E_p - E_{low}) g_{high} + (E_{high} - E_p) g_{low} }\
        { (E_{high} - E_{low}) }
    \end{equation}
    */

    dbl g_hi = distribution[i_hi],
        g_lo = distribution[i_lo];

    g_hi = (1. / g_hi - eta);
    if (g_hi > 0) {
        g_hi = log(g_hi);
    } else {
        return 0.;
    }

    g_lo = (1. / g_lo - eta);
    if (g_lo > 0) {
        g_lo = log(g_lo);
    } else {
        return 0.;
    }

    dbl g = ((E_p - E_lo) * g_hi + (E_hi - E_p) * g_lo) / (E_hi - E_lo);

    g = 1. / (exp(g) + eta);
    if (isnan(g)) {
        return 0.;
    }
    return g;

}


dbl distribution_interpolation(const particle_t &specie, dbl p) {

//    std::cout << "+++++++++++++++++++++++++++" << std::endl;
//    py::print("+++++++++++++++++++++++++++");
//    for (auto grid : specie.grid.grid) if (std::isnan(grid)) py::print("GRID: ", grid);
//    for (auto dist : specie.grid.distribution) if (std::isnan(dist)) py::print("DIST: ", dist);
//    py::print("P: ", p);
//    py::print("M: ", specie.m);
//    py::print("ETA: ", specie.eta);
//    py::print("T: ", specie.T);
//    py::print("EQ: ", specie.in_equilibrium);
//    for (auto grid : specie.grid.grid) if (std::isnan(grid)) std::cout << grid << " ";
//    std::cout << "DIST" << std::endl;
//    for (auto dist : specie.grid.distribution) if (std::isnan(dist)) std::cout << dist << " ";
//    std::cout << "DBL" << std::endl;
//    std::cout << p << " |p| " << std::endl;
//    std::cout << specie.m << " |m| " << std::endl;
//    std::cout << specie.eta << " |eta| " << std::endl;
//    std::cout << specie.T << " |T| " << std::endl;
//    std::cout << specie.in_equilibrium << " |equi| " << std::endl;
    dbl interp = distribution_interpolation(
        specie.grid.grid, specie.grid.distribution,
        p,
        specie.m, specie.eta,
        specie.T, specie.in_equilibrium
    );
//    std::cout << "--> interp: " << interp << " |inter| " << std::endl;
//    py::print("Interp: ", interp);
//    if (std::isnan(interp)) interp = 0.1;
    return interp;
}


/* ## F(fα) functional */

/* ### Naive form
    \begin{align}
        \mathcal{F} &= (1 \pm f_1)(1 \pm f_2) f_3 f_4 - f_1 f_2 (1 \pm f_3) (1 \pm f_4)
        \\\\ &= \mathcal{F}_B + \mathcal{F}_A
    \end{align}
*/

dbl F_A(const std::vector<reaction_t> &reaction, const std::array<dbl, 4> &f, int skip_index=-1) {
    /*
    Forward reaction distribution functional term
    \begin{equation}
        \mathcal{F}_A = f_1 f_2 (1 \pm f_3) (1 \pm f_4)
    \end{equation}
    Change from: \mathcal{F}_A = - f_1 f_2 (1 \pm f_3) (1 \pm f_4)
    :param skip_index: Particle to skip in the expression
    */

    // Use +1 instead
    dbl temp(1.);

    for (int i = 0; i < 4; ++i) {
        if (i != skip_index) {
            if (f[i] < 0) {
                throw std::runtime_error("Negative value of distribution function");
            }
            if (reaction[i].side == -1) {
                temp *= f[i];
            } else {
                temp *= 1. - reaction[i].specie.eta * f[i];
            }
        }
    }

    return temp;
}

dbl F_B(const std::vector<reaction_t> &reaction, const std::array<dbl, 4> &f, int skip_index=-1) {
    /*
    Backward reaction distribution functional term
    \begin{equation}
        \mathcal{F}_B = f_3 f_4 (1 \pm f_1) (1 \pm f_2)
    \end{equation}
    :param skip_index: Particle to skip in the expression
    */

    dbl temp(1.);

    for (int i = 0; i < 4; ++i) {
        if (i != skip_index) {
//            std::cout << "i/f[i]: " << i << " / " << f[i] << std::endl;
            if (f[i] < 0) {
                std::cout << "throw: -> i/f[i]: " << i << " / " << f[i] << " ||| " << std::endl;
                throw std::runtime_error("Negative value of distribution function");
            }
            // -1 is LHS and +1 is RHS of a+b <-> c+d reaction
            if (reaction[i].side == 1) {
                temp *= f[i];
            } else {
                temp *= 1. - reaction[i].specie.eta * f[i];
            }
        }
    }

    return temp;
}

/*
### Linearized in f1 form
\begin{equation}
    \mathcal{F}(f) = f_3 f_4 (1 \pm f_1) (1 \pm f_2) - f_1 f_2 (1 \pm f_3) (1 \pm f_4)
\end{equation}
\begin{equation}
    \mathcal{F}(f) = f_1 (\mp f_3 f_4 (1 \pm f_2) - f_2 (1 \pm f_3) (1 \pm f_4)) \
    + f_3 f_4 (1 \pm f_2)
\end{equation}
\begin{equation}
    \mathcal{F}(f) = \mathcal{F}_B^{(1)} + f_1 (\mathcal{F}_A^{(1)} \pm_1 \mathcal{F}_B^{(1)})
\end{equation}
(i) in F(i) means that the distribution function fi was omitted in the\
corresponding expression. ±j represents the η value of the particle j.
*/
dbl F_f(const std::vector<reaction_t> &reaction, const std::array<dbl, 4> &f) {
    /* Variable part of the distribution functional */
    return F_A(reaction, f, 0) - reaction[0].specie.eta * F_B(reaction, f, 0);
}

dbl F_1(const std::vector<reaction_t> &reaction, const std::array<dbl, 4> &f) {
    /* Constant part of the distribution functional */
    return F_B(reaction, f, 0);
}

dbl F_decay(const std::vector<reaction_t> &reaction, const std::array<dbl, 4> &f) {
    /* Variable part of the distribution functional */
    return F_A(reaction, f, 0);
}

dbl F_creation(const std::vector<reaction_t> &reaction, const std::array<dbl, 4> &f) {
    /* Constant part of the distribution functional */
    return F_B(reaction, f, -1);
}

dbl F_f_vacuum_decay(const std::vector<reaction_t> &reaction, const std::array<dbl, 4> &f) {
    /* Variable part of the distribution functional */
    return -1;
}

dbl F_1_vacuum_decay(const std::vector<reaction_t> &reaction, const std::array<dbl, 4> &f) {
    /* Constant part of the distribution functional */
    return f[3];
}


dbl in_bounds(const std::array<dbl, 4> p, const std::array<dbl, 4> E, const std::array<dbl, 4> m) {
    /* D-functions involved in the interactions imply a cut-off region for the collision\
        integrand. In the general case of arbitrary particle masses, this is a set of \
        irrational inequalities that can hardly be solved (at least, Wolfram Mathematica does\
        not succeed in this). To avoid excessive computations, it is convenient to do an early\
        `return 0` when the particles kinematics lay out of the cut-off region */
    dbl q1, q2, q3, q4;
    q1 = p[0];
    q2 = p[1];
    q3 = p[2];
    q4 = p[3];

    if (q1 < q2) { std::swap(q1, q2); }
    if (q3 < q4) { std::swap(q3, q4); }

    return (E[3] >= m[3] && q1 <= q2 + q3 + q4 && q3 <= q1 + q2 + q4);
}


dbl integrand_full(
    dbl p0, dbl p1, dbl p2,
    const std::vector<reaction_t> &reaction, const std::vector<M_t> &Ms,
    int kind
) {
    /*
    Collision integral interior.
    */

    dbl integrand = 0.;

    std::array<dbl, 4> m;
    std::array<int, 4> sides;

    for (int i = 0; i < 4; ++i) {
        sides[i] = reaction[i].side;
        m[i] = reaction[i].specie.m;
    }

    std::array<dbl, 4> p, E;
    p[0] = p0;
    p[1] = p1;
    p[2] = p2;
    p[3] = 0.;
    E[3] = 0.;
    for (int j = 0; j < 3; ++j) {
        E[j] = energy(p[j], m[j]);
        E[3] += sides[j] * E[j];
    }

    E[3] *= -sides[3];

    if (E[3] < m[3]) { return integrand; }

    p[3] = sqrt(pow(E[3], 2) - pow(m[3], 2));

    if (!in_bounds(p, E, m)) { return integrand; }

    dbl temp = 1.;

    // Avoid rounding errors and division by zero
    for (int k = 1; k < 3; ++k) {
        if (m[k] != 0.) {
            temp *= p[k] / E[k];
        }
    }

    if (temp == 0.) { return integrand; }

    dbl ds = 0.;

    if (Ms[0].K != 0.) {
        ds += Ms[0].K;
    } else {
        if (p[0] != 0.) {
            for (const M_t &M : Ms) {
                ds += D(p, E, m, M.K1, M.K2, M.order, sides);
            }
            ds /= p[0] * E[0];
        } else {
            for (const M_t &M : Ms) {
                ds += Db(p, E, m, M.K1, M.K2, M.order, sides);
            }
        }
    }

    temp *= ds;

    if (temp == 0.) { return integrand; }

    std::array<dbl, 4> f;
    for (int k = 0; k < 4; ++k) {
        const particle_t &specie = reaction[k].specie;
        f[k] = distribution_interpolation(specie, p[k]);
    }

    auto integral_kind = CollisionIntegralKind(kind);

    switch (integral_kind) {
        case CollisionIntegralKind::F_creation:
            return temp * F_creation(reaction, f);
        case CollisionIntegralKind::F_decay:
            return - temp * F_decay(reaction, f);
        case CollisionIntegralKind::F_1_vacuum_decay:
            return temp * F_1_vacuum_decay(reaction, f);
        case CollisionIntegralKind::F_f_vacuum_decay:
            return temp * F_f_vacuum_decay(reaction, f);
        case CollisionIntegralKind::Full_vacuum_decay:
            return temp * (F_1_vacuum_decay(reaction, f) + f[0] * F_f_vacuum_decay(reaction, f));
        case CollisionIntegralKind::F_1:
            return temp * F_1(reaction, f);
        case CollisionIntegralKind::F_f:
            return temp * F_f(reaction, f);
        case CollisionIntegralKind::Full:
        default:
            return temp * (F_1(reaction, f) + f[0] * F_f(reaction, f));
    }
}


Kinematics get_reaction_type(const std::vector<reaction_t> &reaction) {
    int reaction_type = 0;
    for (const reaction_t &reactant : reaction) {
        reaction_type += reactant.side;
    }
    return static_cast<Kinematics>(reaction_type);
}


struct integration_params {
    dbl p0;
    dbl p1;
    dbl p2;
    const std::vector<reaction_t> *reaction;
    const std::vector<M_t> *Ms;
    dbl min_1;
    dbl max_1;
    dbl min_2;
    dbl max_2;
    dbl max_3;
    int kind;
    dbl releps;
    dbl abseps;
    size_t subdivisions;
    gsl_integration_workspace *w;
};


dbl integrand_1st_integration(
    dbl p2, void *p
) {
    struct integration_params &params = *(struct integration_params *) p;
    dbl p0 = params.p0;
    dbl p1 = params.p1;
    return integrand_full(p0, p1, p2, *params.reaction, *params.Ms, params.kind);
}


dbl p2_max_dec(const std::vector<reaction_t> &reaction, dbl p0, dbl p1) {
    return sqrt(
            pow(
                energy(p0, reaction[0].specie.m)
                - energy(p1, reaction[1].specie.m)
                - reaction[3].specie.m
            , 2)
            - pow(reaction[2].specie.m, 2)
        );
}


dbl p2_min_creation(const std::vector<reaction_t> &reaction, dbl p0, dbl p1) {
    dbl min = reaction[3].specie.m - energy(p0, reaction[0].specie.m) - energy(p1, reaction[1].specie.m);
    dbl min2 = pow(min, 2) - pow(reaction[2].specie.m, 2);
    if (min <= 0 || min2 <= 0) {
        return 0.;
    }
    return sqrt(min2);
}


dbl p2_max_creation(const std::vector<reaction_t> &reaction, dbl p0, dbl p1) {
    dbl m0 = reaction[0].specie.m;
    dbl m1 = reaction[1].specie.m;
    dbl m2 = reaction[2].specie.m;
    dbl m3 = reaction[3].specie.m;

    return (pow(m3, 2) - pow(m0, 2) - pow(m1, 2) - pow(m2, 2)) / (2 * (1e-3 + m0)) - p1;
}


dbl p2_min_scat(const std::vector<reaction_t> &reaction, dbl p0, dbl p1) {
    dbl m0 = reaction[0].specie.m;
    dbl m1 = reaction[1].specie.m;
    dbl m2 = reaction[2].specie.m;
    dbl m3 = reaction[3].specie.m;

    if (p0 != 0){
        return 0.;
    }

    if (m0 != 0) {
        if (m1 == 0 && m2 == 0 && m3 == 0) {
            return m0 / 2.;
        }

        if (m0 + m1 > m2 + m3) {
            dbl temp1 = m0 + sqrt(pow(m1,2) + pow(p1,2));
            dbl temp2 = pow(m3,2) + pow(p1,2);
            dbl temp3 = ((temp2 - pow(temp1,2) - pow(m2,2)) * p1
                        + sqrt(
                            pow(temp1,2) * (pow(temp1,4) + pow(temp2,2)
                            + (4 * pow(p1,2) - 2 * temp2 + pow(m2,2)) * pow(m2,2)
                            - 2 * pow(temp1,2) * (temp2 + pow(m2,2)))
                        )) / (2 * (pow(temp1,2) - pow(p1,2)));
            return std::max(temp3, 0.);
        }

        return 0.;
    }

    if (m1 != 0) {
        dbl temp = (pow(p1,2) + pow(m1,2))
                    * (
                    pow(m1,4) + pow(pow(m2,2) - pow(m3,2), 2)
                    - 2 * pow(m1,2) * (pow(m2,2) + pow(m3,2))
                    );

        if (temp < 0) {
            temp = 0;
        }

        dbl temp1 = (
                    -p1 * (pow(m1,2) + pow(m2,2) - pow(m3,2))
                    + sqrt(temp)
                    )
                    / (2 * pow(m1,2));

        dbl temp2 = -temp1;

        return std::max(temp1, temp2);
    }

    return 0.;
}

dbl p2_max_scat(const std::vector<reaction_t> &reaction, dbl p0, dbl p1) {
    dbl m0 = reaction[0].specie.m;
    dbl m1 = reaction[1].specie.m;
    dbl m2 = reaction[2].specie.m;
    dbl m3 = reaction[3].specie.m;

    if (p0 != 0) {
        return sqrt(
                pow(
                    energy(p0, reaction[0].specie.m)
                    + energy(p1, reaction[1].specie.m)
                    - reaction[3].specie.m
                , 2)
                - pow(reaction[2].specie.m, 2)
            );
    }

    if (m0 != 0) {
        if (m1 == 0 && m2 == 0 && m3 == 0) {
            return (m0 + 2 * p1) / 2.;
        }

        if (m0 + m1 > m2 + m3) {
            dbl temp1 = m0 + sqrt(pow(m1,2) + pow(p1,2));
            dbl temp2 = pow(m3,2) + pow(p1,2);
            dbl temp3 = ((-temp2 + pow(temp1,2) + pow(m2,2)) * p1
                        + sqrt(
                            pow(temp1,2) * (pow(temp1,4) + pow(temp2,2)
                            + (4 * pow(p1,2) - 2 * temp2 + pow(m2,2)) * pow(m2,2)
                            - 2 * pow(temp1,2) * (temp2 + pow(m2,2)))
                        )) / (2 * (pow(temp1,2) - pow(p1,2)));

            return std::max(temp3, 0.);
        }

        return sqrt(
                pow(
                    m0 - m3
                    + energy(p1, m1)
                , 2)
                - pow(m2, 2)
            );
    }

    if (m1 != 0) {
        dbl temp = (pow(p1,2) + pow(m1,2))
                    * (
                    pow(m1,4) + pow(pow(m2,2) - pow(m3,2), 2)
                    - 2 * pow(m1,2) * (pow(m2,2) + pow(m3,2))
                    );

        if (temp < 0) {
            temp = 0;
        }

        return (
                p1 * (pow(m1,2) + pow(m2,2) - pow(m3,2))
                + sqrt(temp)
                )
                / (2 * pow(m1,2));
    }

    return p1;
}


dbl integrand_2nd_integration(
    dbl p1, void *p
) {
    struct integration_params &old_params = *(struct integration_params *) p;
    dbl min_2 = old_params.min_2;
    dbl max_2 = old_params.max_2;
    dbl max_3 = old_params.max_3;
    dbl p0 = old_params.p0;
    auto reaction = *old_params.reaction;

    auto reaction_type = get_reaction_type(reaction);

    if (reaction_type == Kinematics::DECAY) {
        min_2 = 0.;
        max_2 = p2_max_dec(reaction, p0, p1);
    }

     if (reaction_type == Kinematics::SCATTERING) {
        min_2 = p2_min_scat(reaction, p0, p1);
        max_2 = p2_max_scat(reaction, p0, p1);
     }

    if (reaction_type == Kinematics::CREATION) {
        dbl max = energy(max_3, reaction[3].specie.m) - energy(p0, reaction[0].specie.m) - energy(p1, reaction[1].specie.m);
        dbl max2 = pow(max, 2) - pow(reaction[2].specie.m, 2);
        dbl min = reaction[3].specie.m - energy(p0, reaction[0].specie.m) - energy(p1, reaction[1].specie.m);
        dbl min2 = pow(min, 2) - pow(reaction[2].specie.m, 2);
        if (max <= 0 || max2 <= 0) {
            return 0.;
        }
        else {
            max_2 = sqrt(max2);
        }
        if (min <= 0 || min2 <= 0) {
            min_2 = 0.;
        }
        else {
            min_2 = sqrt(min2);
        }
    }

    gsl_function F;
    struct integration_params params = old_params;
        params.p1 = p1;
    params.min_2 = min_2;
    params.max_2 = max_2;
    F.params = &params;

    dbl result, error;
    size_t status;
    F.function = &integrand_1st_integration;
    gsl_set_error_handler_off();
    status = gsl_integration_qag(&F, min_2, max_2, params.abseps, params.releps, params.subdivisions, GSL_INTEG_GAUSS15, params.w, &result, &error);
    if (status) {
        printf("(p0=%e, p1=%e) 1st integration result: %e ± %e. %i intervals. %s\n", params.p0, p1, result, error, (int) params.w->size, gsl_strerror(status));
        throw std::runtime_error("Integrator failed to reach required accuracy");
    }

    return result;
}


std::vector<dbl> integration(
    std::vector<dbl> ps, dbl min_1, dbl max_1, dbl min_2, dbl max_2, dbl max_3,
    const std::vector<reaction_t> &reaction,
    const std::vector<M_t> &Ms,
    dbl stepsize, int kind=0
) {

    std::vector<dbl> integral(ps.size(), 0.);

    // Determine the integration bounds
    auto reaction_type = get_reaction_type(reaction);

    // Note firstprivate() clause: those variables will be copied for each thread
    #pragma omp parallel for default(none) shared(std::cout,ps, Ms, reaction, integral, stepsize, kind, reaction_type) firstprivate(min_1, max_1, min_2, max_2, max_3)
    for (size_t i = 0; i < ps.size(); ++i) {
        dbl p0 = ps[i];

        if (reaction_type == Kinematics::DECAY) {
            max_1 = sqrt(
                pow(energy(p0, reaction[0].specie.m) - reaction[2].specie.m - reaction[3].specie.m, 2)
                - pow(reaction[1].specie.m, 2)
            );
        }
        if (reaction_type == Kinematics::SCATTERING) {
            dbl min = reaction[2].specie.m + reaction[3].specie.m - energy(p0, reaction[0].specie.m);
            dbl min2 = pow(min, 2) - pow(reaction[1].specie.m, 2);
            if (min <= 0 || min2 <= 0) {
                min_1 = 0.;
            }
            else {
                min_1 = sqrt(min2);
            }
            if (max_1 > 3. * min_1) {
                max_1 = max_1;
            }
            else {
                max_1 = 3. * min_1;
            }
        }
        if (reaction_type == Kinematics::CREATION) {
            if (reaction[3].specie.m == 0.) { continue; }
            dbl max = energy(max_3, reaction[3].specie.m) - reaction[2].specie.m - energy(p0, reaction[0].specie.m);
            dbl max2 = pow(max, 2) - pow(reaction[1].specie.m, 2);
            if (max <= 0 || max2 <= 0) {
                continue;
            }
            else {
                max_1 = sqrt(max2);
            }
            min_1 = 0.;
        }

        dbl result(0.), error(0.);
        size_t status;
        gsl_function F;
        F.function = &integrand_2nd_integration;

        dbl releps = 1e-2; // was 1e-2
        dbl abseps = releps / stepsize;
        auto integral_kind = CollisionIntegralKind(kind);
        if (integral_kind != CollisionIntegralKind::F_f
            && integral_kind != CollisionIntegralKind::F_f_vacuum_decay)
        {
            dbl f = distribution_interpolation(reaction[0].specie, p0);
            if (reaction[0].specie.m == 0.) {abseps *= f; }
            // else {abseps *= 1e-15;}
            // abseps *= 1e-20;
        }

        size_t subdivisions = 1000; // original = 100000
        gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(subdivisions);
        gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(subdivisions);
        struct integration_params params = {
            p0, 0., 0.,
            &reaction, &Ms,
            min_1, max_1, min_2, max_2, max_3,
            kind, releps, abseps,
            subdivisions, w2
        };
        F.params = &params;

        gsl_set_error_handler_off();

        status = gsl_integration_qag(&F, min_1, max_1, abseps, releps, subdivisions, GSL_INTEG_GAUSS15, w1, &result, &error);
        if (status) {
            printf("Just checking");
            printf("2nd integration_1 result: %e ± %e. %i intervals. %s\n", result, error, (int) w1->size, gsl_strerror(status));
//            throw std::runtime_error("Integrator failed to reach required accuracy");
        }
        gsl_integration_workspace_free(w1);
        gsl_integration_workspace_free(w2);
        integral[i] += result;
    }

    return integral;
}


PYBIND11_MODULE(integral, m) {
    m.def("distribution_interpolation", [](
        const std::vector<dbl> &grid,
        const std::vector<dbl> &distribution,
        const py::array_t<double> &ps, dbl m=0., int eta=1, dbl T=1.,
        bool in_equilibrium=false) {
            auto v = [grid, distribution, m, eta, T, in_equilibrium](double p) {
                return distribution_interpolation(grid, distribution, p, m, eta, T, in_equilibrium);
            };
            return py::vectorize(v)(ps);
        },
        "Exponential interpolation of distribution function",
        "grid"_a, "distribution"_a,
        "p"_a, "m"_a=0, "eta"_a=1, "T"_a=1., "in_equilibrium"_a=false
    );
    m.def("binary_find", &binary_find,
          "grid"_a, "x"_a);

    m.def("D1", &D1);
    m.def("D2", &D2);
    m.def("D3", &D3);
    m.def("Db1", &Db1);
    m.def("Db2", &Db2);
    m.def("D", &D,
          "p"_a, "E"_a, "m"_a,
          "K1"_a, "K2"_a, "order"_a, "sides"_a);
    m.def("Db", &Db,
          "p"_a, "E"_a, "m"_a,
          "K1"_a, "K2"_a, "order"_a, "sides"_a);

    m.def("integration", &integration,
          "ps"_a, "min_1"_a, "max_1"_a, "min_2"_a, "max_2"_a, "max_3"_a,
          "reaction"_a, "Ms"_a, "stepsize"_a, "kind"_a);

    py::enum_<CollisionIntegralKind>(m, "CollisionIntegralKind")
        .value("Full", CollisionIntegralKind::Full)
        .value("F_1", CollisionIntegralKind::F_1)
        .value("F_f", CollisionIntegralKind::F_f)
        .value("Full_vacuum_decay", CollisionIntegralKind::Full_vacuum_decay)
        .value("F_creation", CollisionIntegralKind::F_creation)
        .value("F_decay", CollisionIntegralKind::F_decay)
        .value("F_1_vacuum_decay", CollisionIntegralKind::F_1_vacuum_decay)
        .value("F_f_vacuum_decay", CollisionIntegralKind::F_f_vacuum_decay)
        .enum_::export_values();

    py::class_<M_t>(m, "M_t")
        .def(py::init<std::array<int, 4>, dbl, dbl, dbl>(),
             "order"_a, "K1"_a=0., "K2"_a=0., "K"_a=0.);

    py::class_<grid_t>(m, "grid_t")
        .def(py::init<std::vector<dbl>, std::vector<dbl>>(),
             "grid"_a, "distribution"_a);

    py::class_<particle_t>(m, "particle_t")
        .def(py::init<int, dbl, grid_t, int, dbl>(),
             "eta"_a, "m"_a, "grid"_a, "in_equilibrium"_a, "T"_a);

    py::class_<reaction_t>(m, "reaction_t")
        .def(py::init<particle_t, int>(),
             "specie"_a, "side"_a);
}
