#include "integral.h"

int main() {

const std::vector<double> test_vec = {1.,2.,3.};
double x = 5.0;
std::pair<int, int> ret = binary_find(test_vec, x);

std::cout << "First element: " << ret.first << std::endl;
std::cout << "Second element: " << ret.second << std::endl;

const std::vector<reaction_t> R{};
const std::vector<M_t> M{};
std::array<double,4> test{};
double ret3 = F_1(R, test);
std::cout << "First element: 3" << ret3 << std::endl;

// std::vector<double> integ = integration({1.,2.}, 0., 1., 0., 1., 1., R, M, 0.9, 0);

double ret2 = distribution_interpolation(test_vec, test_vec, 10., 1., 1, 1., false);
std::cout << "First element: 1" << ret2 << std::endl;

}
// dbl p, dbl m=0., int eta=1, dbl T=1., bool in_equilibrium=false

//604 std::vector<dbl> integration(
//605     std::vector<dbl> ps, dbl min_1, dbl max_1, dbl min_2, dbl max_2, dbl max_3,
//606     const std::vector<reaction_t> &reaction,
//607     const std::vector<M_t> &Ms,
//608     dbl stepsize, int kind=0
