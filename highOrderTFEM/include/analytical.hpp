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
    /**
     * Namespace for closed-form solutions to test against and use for initial conditions.
     */
    namespace Analytical
    {

        /**
         * Represents a single term in a solution that is a linear combination of terms c_i F_(nx_i, ny_i) (...)
         *
         * coef - scalar magnitude of the term
         * nx - Solutions are typically indexed in two dimensions. This is the x index of the term
         * ny - y index of the term
         */
        struct Term
        {
            double coef;
            int nx;
            int ny;
        };

        /**
         * Term specific to a zero-boundary rectangle. Each ZeroBoundaryTerm represents:
         *      amplitude * exp(coef_t * t) * sin(coef_x * x) * sin(coef_y * y)
         */
        struct ZeroBoundaryTerm
        {
            double amplitude;
            double coef_t;
            double coef_x;
            double coef_y;
        };

        /**
         * Analytical solutions to a rectangular region where boundaries are held to 0, which
         * are in the form of a sum over terms that look like:
         *
         * a * exp(-k * t * (bx^2 + by^2)) * sin(bx * x) * sin(by * y)
         *
         * Templated on view type to allow for placement on device or host.
         */
        template <class TermView = Kokkos::View<ZeroBoundaryTerm *>>
        class ZeroBoundary
        {
        protected:
            TermView terms;
            double x_shift; // rectangle doesn't always start at 0, so shift the coordinates
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
                // Translate generic index-based terms to specific coefficients
                // There shouldn't be too many terms so not bothering with parallelism
                terms = TermView("analytic solution terms", solution_terms.size());
                auto terms_mirror = Kokkos::create_mirror_view(terms);

                for (int i = 0; i < solution_terms.size(); i++)
                {
                    Term t = solution_terms[i];
                    double lambda_x = t.nx * M_PI / x_width;
                    double lambda_y = t.ny * M_PI / y_width;
                    terms_mirror(i) = {
                        t.coef,                                     // amplitude
                        -k * (pow(lambda_x, 2) + pow(lambda_y, 2)), // t
                        lambda_x,                                   // x
                        lambda_y};                                  // y
                }

                // The terms were assembled on the host, while the destination buffer
                // might be on the device. Copy over.
                Kokkos::deep_copy(terms, terms_mirror);
            }

            /**
             * Computes the value of the solution at the given point and time.
             * 
             * Make sure the calling execution space has access to the memory 
             * space where the templated view resides.
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