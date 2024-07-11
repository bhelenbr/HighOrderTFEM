/**
 * Analytical solutions to the heat equation for verification purposes
 */
#ifndef highOrderTFEM_analytical_hpp
#define highOrderTFEM_analytical_hpp

#include <Kokkos_Core.hpp>
#include <vector>
#include <math.h> // for pi

namespace TFEM
{
    namespace Analytical
    {

        /**
         * Represents a single term in a solution that is a linear combination of terms c_i F_(nx_i, ny_i) (...)
         */
        struct Term
        {
            double coef;
            int nx;
            int ny;
        };

        struct ZeroBoundaryTerm
        {
            double amplitude;
            double coef_t;
            double coef_x;
            double coef_y;
        };

        /**
         * Analytical solutions to a rectangular region where boundaries are held to 0
         */
        template <class TermView = Kokkos::View<ZeroBoundaryTerm *>>
        class ZeroBoundary
        {
        protected:
            TermView terms;
            double x_shift;
            double y_shift;

        public:
            /**
             * Parameters:
             *  - k: stiffness parameter, i.e d/dt U - k nabla^2 U = 0
             *  - x_start: lowest x position
             *  - x_width: x-dimension of the rectangular region
             *  - y_start: lowest y position
             *  - y_width: y-dimension of the rectangular region
             */
            ZeroBoundary(double k, double x_start, double x_width, double y_start, double y_width, std::vector<Term> solution_terms)
                : x_shift(x_start),
                  y_shift(y_start)
            {
                // Assemble terms on host. Shouldn't be too many so not bothering with parallelism
                terms = TermView("analytic solution terms", solution_terms.size());
                auto terms_mirror = Kokkos::create_mirror_view(terms);

                for (int i = 0; i < solution_terms.size(); i++)
                {
                    Term t = solution_terms[i];
                    terms_mirror(i) = {
                        t.coef,                             // amplitude
                        -k * (pow(t.nx, 2) + pow(t.ny, 2)), // t
                        t.nx * M_PI / x_width,              // x
                        t.ny * M_PI / y_width};             // y
                }

                // copy host to device
                Kokkos::deep_copy(terms, terms_mirror);
            }

            /**
             * Computes the value of the solution at the given point and time. Must be
             * called somewhere where the templated exection space is
             */
            KOKKOS_INLINE_FUNCTION double
            operator()(double x, double y, double t) const
            {
                double result = 0;
                for (int i = 0; i < terms.extent(0); i++)
                {
                    const ZeroBoundaryTerm &term = terms(i);
                    result += term.amplitude * (exp(t * term.coef_t)) * sin((x - x_shift) * term.coef_x) * sin((y - y_shift) * term.coef_y);
                }
                return result;
            }
        };
    } // namsepace Analytical
} // namespace TFEM
#endif