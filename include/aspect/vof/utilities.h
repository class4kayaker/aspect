/*
 Copyright (C) 2016 by the authors of the ASPECT code.

 This file is part of ASPECT.

 ASPECT is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2, or (at your option)
 any later version.

 ASPECT is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with ASPECT; see the file doc/COPYING.  If not see
 <http://www.gnu.org/licenses/>.
 */

#ifndef __aspect__vof_utilities_h
#define __aspect__vof_utilities_h

#include <aspect/global.h>

namespace aspect
{
  namespace VolumeOfFluid
  {
    using namespace dealii;

    // Utilitiess

    /**
     * Function to calculate volume fraction contained by indicator function
     * H(d-normal*(x'-x_{cen}')) on the [0, 1]^dim unit cell where x_{cen} is
     * the unit cell center.
     *
     * Currently only works assuming constant Jacobian determinant.
     */
    template<int dim>
    double vof_from_d (const Tensor<1, dim, double> normal,
                       const double d);

    /**
     * Function to calculate required value of d to obtain given volume
     * fraction for indicator function H(d-normal*(x'-x_{cen}')) on the [0,
     * 1]^dim unit cell where x_{cen} is the unit cell center.
     *
     * Currently only works assuming constant Jacobian determinant.
     */
    template<int dim>
    double d_from_vof (const Tensor<1, dim, double> normal,
                       const double vol);

    /**
     * Function to calculate xFEM-like function values on the unit cell.
     * Currently exact for basises up to degree 1.
     */
    template<int dim>
    void xFEM_Heaviside(const int degree,
                        const Tensor<1, dim, double> normal,
                        const double d,
                        const std::vector<Point<dim>> &points,
                        std::vector<double> &values);
    /**
     * Function to calculate xFEM-like function values on the unit cell.
     * Currently exact for basises up to degree 1.
     */
    template<int dim>
    void xFEM_Heaviside_d_d(const int degree,
                            const Tensor<1, dim, double> normal,
                            const double d,
                            const std::vector<Point<dim>> &points,
                            std::vector<double> &values);


    /**
     * Function to do newton iteration calculation of correct d for a given
     * normal to get vol_frac from xFEM_Heaviside integrated against the given
     * weights.
     */
    template<int dim>
    double d_from_vof_newton(const int degree,
                             const Tensor<1, dim, double> normal,
                             const double vol_frac,
                             const double vol,
                             const double epsilon,
                             const std::vector<Point<dim>> &points,
                             const std::vector<double> &weights);

    /**
     * Function to calculate volume fraction contained by indicator function
     * H(d-normal*(x'-x_{cen}')) on the [0, 1]^dim unit cell where x_{cen} is
     * the unit cell center, using a polynomial mapping of degree up to "degree".
     */
    template<int dim>
    double vol_from_d(const int degree,
                      const Tensor<1, dim, double> normal,
                      const double d,
                      const std::vector<Point<dim>> &points,
                      const std::vector<double> &weights);

    /**
     * Function to calculate flux volume fraction based on a method of
     * characteristics approximation of the interface on the cell's face over
     * the timestep.
     */
    template<int dim>
    double calc_vof_flux_edge (const unsigned int dir,
                               const double timeGrad,
                               const Tensor<1, dim, double> cell_normal,
                               const double d_face);
  }
}

#endif
