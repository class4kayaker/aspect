/*
 Copyright (C) 2016 - 2018 by the authors of the ASPECT code.

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

#include <aspect/global.h>
#include <aspect/simulator.h>
#include <aspect/vof/handler.h>
#include <aspect/vof/utilities.h>

namespace aspect
{
  using namespace dealii;

  template <>
  void VoFHandler<2>::update_vof_normals (const VoFField<2> field,
                                          LinearAlgebra::BlockVector &solution)
  {
    const int dim = 2;
    const unsigned int max_degree = 1;

    LinearAlgebra::BlockVector initial_solution;

    sim.computing_timer.enter_section("   Reconstruct VoF interfaces");

    initial_solution.reinit(sim.system_rhs, false);

    // Boundary reference
    typename DoFHandler<dim>::active_cell_iterator endc =
      this->get_dof_handler().end ();

    // Interface Reconstruction vars
    const unsigned int n_local = 9;

    Vector<double> local_vofs (n_local);
    std::vector<Point<dim>> resc_cell_centers (n_local);
    std::vector<typename DoFHandler<dim>::active_cell_iterator> neighbor_cells(n_local);

    const unsigned int n_sums = 3;
    std::vector<double> strip_sums (dim * n_sums);

    const unsigned int n_normals = 6+1;
    std::vector<Tensor<1, dim, double>> normals (n_normals);
    std::vector<double> d_vals (n_normals);
    std::vector<double> errs (n_normals);

    // Variables to do volume calculations

    QGauss<dim> quadrature(max_degree);

    std::vector<double> xFEM_values(quadrature.size());

    const FiniteElement<dim> &system_fe = this->get_fe();

    FEValues<dim> fevalues(this->get_mapping(), system_fe, quadrature,
                           update_JxW_values);

    // Normal holding vars
    Point<dim> uReCen;
    Tensor<1, dim, double> normal;
    double d;

    for (unsigned int i=0; i<dim; ++i)
      uReCen[i] = 0.5;

    std::vector<types::global_dof_index> cell_dof_indicies (system_fe.dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indicies (system_fe.dofs_per_cell);

    const FEVariable<dim> &vof_var = field.volume_fraction;
    const unsigned int vof_c_index = vof_var.first_component_index;
    const unsigned int vof_ind
      = this->get_fe().component_to_system_index(vof_c_index, 0);
    const unsigned int vof_blockidx = vof_var.block_index;

    const FEVariable<dim> &vofN_var = field.reconstruction;
    const unsigned int vofN_c_index = vofN_var.first_component_index;
    const unsigned int vofN_blockidx = vofN_var.block_index;

    const FEVariable<dim> &vofLS_var = field.level_set;
    const unsigned int vofLS_c_index = vofLS_var.first_component_index;
    const unsigned int n_vofLS_dofs = vofLS_var.fe->dofs_per_cell;
    const unsigned int vofLS_blockidx = vofLS_var.block_index;

    //Iterate over cells
    for (auto cell : this->get_dof_handler().active_cell_iterators ())
      {
        if (!cell->is_locally_owned ())
          continue;

        double cell_vof;

        // Obtain data for this cell and neighbors
        cell->get_dof_indices (local_dof_indicies);
        cell_vof = solution(local_dof_indicies[vof_ind]);

        normal[0] = 0.0;
        normal[1] = 0.0;
        d = -1.0;

        if (cell_vof > 1.0 - volume_fraction_threshold)
          {
            d = 1.0;
            initial_solution(local_dof_indicies[vof_ind]) = 1.0;
          }
        else if (cell_vof < volume_fraction_threshold)
          {
            d = -1.0;
            initial_solution(local_dof_indicies[vof_ind]) = 0.0;
          }
        else
          {
            initial_solution(local_dof_indicies[vof_ind]) = cell_vof;

            //Identify best normal
            // Get references to neighboring cells
            for (unsigned int i = 0; i < 3; ++i)
              {
                typename DoFHandler<dim>::active_cell_iterator cen;
                if (i == 0 || i == 2)
                  {
                    const unsigned int neighbor_no = (i/2);
                    const typename DoFHandler<dim>::face_iterator face = cell->face (neighbor_no);
                    if (face->at_boundary() && !cell->has_periodic_neighbor(neighbor_no)||
                        face->has_children())
                      cen = endc;
                    else
                      {
                        const typename DoFHandler<dim>::cell_iterator neighbor =
                          cell->neighbor_or_periodic_neighbor(neighbor_no);
                        if (neighbor->level() == cell->level() &&
                            neighbor->active())
                          cen = neighbor;
                        else
                          cen = endc;
                      }
                  }
                else
                  cen = cell;
                for (unsigned int j = 0; j < 3; ++j)
                  {
                    typename DoFHandler<dim>::active_cell_iterator curr;
                    if (cen == endc)
                      {
                        curr = endc;
                      }
                    else
                      {
                        if (j == 0 || j == 2)
                          {
                            const unsigned int neighbor_no = 2+(j/2);
                            const typename DoFHandler<dim>::face_iterator face = cen->face (neighbor_no);
                            if (face->at_boundary() && !cen->has_periodic_neighbor(neighbor_no)||
                                face->has_children())
                              curr = endc;
                            else
                              {
                                const typename DoFHandler<dim>::cell_iterator neighbor =
                                  cen->neighbor_or_periodic_neighbor(neighbor_no);
                                if (neighbor->level() == cell->level() &&
                                    neighbor->active())
                                  curr = neighbor;
                                else
                                  curr = endc;
                              }
                          }
                        else
                          curr = cen;
                      }
                    if (curr != endc)
                      {
                        curr->get_dof_indices (cell_dof_indicies);
                        resc_cell_centers[3 * j + i] = Point<dim> (-1.0 + i,
                                                                   -1.0 + j);
                      }
                    else
                      {
                        cell->get_dof_indices (cell_dof_indicies);
                        resc_cell_centers[3 * j + i] = Point<dim> (0.0,
                                                                   0.0);
                      }
                    local_vofs (3 * j + i) = solution (cell_dof_indicies[vof_ind]);
                    neighbor_cells[3 * j + i] = curr;
                  }
              }

            // Gather cell strip sums
            for (unsigned int i = 0; i < dim * n_sums; ++i)
              strip_sums[i] = 0.0;

            for (unsigned int i = 0; i < 3; ++i)
              {
                for (unsigned int j = 0; j < 3; ++j)
                  {
                    strip_sums[3 * 0 + i] += local_vofs (3 * j + i);
                    strip_sums[3 * 1 + j] += local_vofs (3 * j + i);
                  }
              }

            // std::cout << "  Strip sums: " << std::endl;
            // for (unsigned int i = 0; i < dim * n_sums; ++i)
            //   std::cout << "    " << strip_sums[i] << std::endl;

            // Calculate normal vectors for the 6 candidates from the efficient
            // least squares approach
            for (unsigned int di = 0; di < dim; ++di)
              {
                unsigned int di2 = (di + 1) % dim;
                for (unsigned int i = 0; i < 3; ++i)
                  {
                    normals[3 * di + i][di] = 0.0;
                    normals[3 * di + i][di2] = 0.0;
                    if (i % 2 == 0)
                      {
                        //use low sum
                        normals[3 * di + i][di] += strip_sums[3 * di + 0];
                        normals[3 * di + i][di2] += 1.0;
                      }
                    else
                      {
                        //use high sum
                        normals[3 * di + i][di] += strip_sums[3 * di + 1];
                        normals[3 * di + i][di2] += 0.0;
                      }
                    if (i == 0)
                      {
                        //use low sum
                        normals[3 * di + i][di] -= strip_sums[3 * di + 1];
                        normals[3 * di + i][di2] += 0.0;
                      }
                    else
                      {
                        //use high sum
                        normals[3 * di + i][di] -= strip_sums[3 * di + 2];
                        normals[3 * di + i][di2] += 1.0;
                      }
                    if (strip_sums[3 * di2 + 2] > strip_sums[3 * di2 + 0])
                      normals[3 * di + i][di2] *= -1.0;
                  }
              }

            // Add time extrapolated local normal as candidate
            // this is not expected to be the best candidate in general, but
            // should result in exact reconstruction for linear interface
            // translations
            // Inclusion of this candidate will not reduce accuracy due to it
            // only being selected if it produces a better interface
            // approximation than the ELS candidates. Note that this will
            // render linear translation problems less dependent on the
            // interface reconstruction, so other tests will also be necessary.
            for (unsigned int i=0; i<dim; ++i)
              normals[6][i] = solution(local_dof_indicies[system_fe
                                                          .component_to_system_index(vofN_c_index+i, 0)]);

            // If candidate normal too small, remove from consideration
            if (normals[6]*normals[6]< volume_fraction_threshold)
              {
                normals[6][0] = 0;
                normals[6][1] = 0;
              }

            unsigned int mn_ind = 0;
            {
              fevalues.reinit(cell);
              const std::vector<double> weights = fevalues.get_JxW_values();

              double cell_vol = 0.0;
              for (unsigned int j=0; j<weights.size(); ++j)
                {
                  cell_vol+=weights[j];
                }
              for (unsigned int nind = 0; nind < n_normals; ++nind)
                {
                  errs[nind] = 0.0;
                  const double normal_norm = normals[nind].norm_square();

                  if (normal_norm > volume_fraction_threshold) // If candidate normal too small set error to maximum
                    {
                      d_vals[nind] = VolumeOfFluid::compute_interface_location_newton<dim> (max_degree, normals[nind], cell_vof, cell_vol,
                                                                                            vof_reconstruct_epsilon,
                                                                                            quadrature.get_points(), weights);
                    }
                  else
                    {
                      errs[nind] = 9.0;
                    }
                }
            }

            for (unsigned int i = 0; i < n_local; ++i)
              {
                if (neighbor_cells[i] == endc)
                  {
                    continue;
                  }

                fevalues.reinit(neighbor_cells[i]);

                const std::vector<double> weights = fevalues.get_JxW_values();

                double cell_vol = 0.0;
                for (unsigned int j=0; j<weights.size(); ++j)
                  {
                    cell_vol+=weights[j];
                  }

                for (unsigned int nind = 0; nind < n_normals; ++nind)
                  {
                    const double normal_norm = normals[nind]*normals[nind];

                    if (normal_norm > volume_fraction_threshold) // If candidate normal too small skip as set to max already
                      {
                        double dot = 0.0;
                        for (unsigned int di = 0; di < dim; ++di)
                          dot += normals[nind][di] * resc_cell_centers[i][di];
                        const double n_vof = VolumeOfFluid::compute_fluid_volume_xFEM<dim> (max_degree, normals[nind], d_vals[nind]-dot,
                                                                                            quadrature.get_points(), weights)/cell_vol;
                        const double cell_err = local_vofs (i) - n_vof;
                        errs[nind] += cell_err * cell_err;
                      }
                  }
              }

            for (unsigned int nind = 0; nind < n_normals; ++nind)
              {
                if (errs[mn_ind] >= errs[nind])
                  mn_ind = nind;
                // std::cout << "\t" << normals[nind] << " d ";
                // std::cout << d_vals[nind] << " e ";
                // std::cout << errs[nind] << " " << mn_ind << std::endl;
              }

            normal = normals[mn_ind];
            d = d_vals[mn_ind];
          }

        // double n2 = sqrt(normal*normal);
        // if (n2 > volume_fraction_threshold)
        //   {
        //     normal = (normal / n2);
        //     d = d / n2;
        //   }
        // else
        //   {
        //     normal[0] = 0.0;
        //     normal[1] = 0.0;
        //   }

        for (unsigned int i=0; i<dim; ++i)
          initial_solution (local_dof_indicies[system_fe
                                               .component_to_system_index(vofN_c_index+i, 0)]) = normal[i];

        initial_solution (local_dof_indicies[system_fe
                                             .component_to_system_index(vofN_c_index+dim, 0)]) = d;

        for (unsigned int i=0; i<n_vofLS_dofs; ++i)
          {
            // Recenter unit cell on origin
            Tensor<1, dim, double> uSupp = vofLS_var.fe->unit_support_point(i)-uReCen;
            initial_solution (local_dof_indicies[system_fe
                                                 .component_to_system_index(vofLS_c_index, i)])
              = d-uSupp*normal;
          }
      }

    initial_solution.compress(VectorOperation::insert);

    sim.compute_current_constraints();
    sim.current_constraints.distribute(initial_solution);

    // solution.block(vof_blockidx) = initial_solution.block(vof_blockidx);
    solution.block(vofN_blockidx) = initial_solution.block(vofN_blockidx);
    solution.block(vofLS_blockidx) = initial_solution.block(vofLS_blockidx);

    sim.computing_timer.exit_section();
  }


  template <>
  void VoFHandler<3>::update_vof_normals (const VoFField<3> /*field*/,
                                          LinearAlgebra::BlockVector &/*solution*/)
  {
    Assert(false, ExcNotImplemented());
  }

  template <>
  void VoFHandler<2>::update_vof_composition (const typename Simulator<2>::AdvectionField composition_field,
                                              const VoFField<2> vof_field,
                                              LinearAlgebra::BlockVector &solution)
  {
    const int dim = 2;

    LinearAlgebra::BlockVector initial_solution;

    sim.computing_timer.enter_section("  Compute VoF compositions");

    initial_solution.reinit(sim.system_rhs, false);

    // Normal holding vars
    Point<dim> uReCen;

    for (unsigned int i=0; i<dim; ++i)
      uReCen[i] = 0.5;

    const FiniteElement<dim> &system_fe = this->get_fe();

    std::vector<types::global_dof_index> local_dof_indicies (system_fe.dofs_per_cell);

    const FEVariable<dim> &vof_var = vof_field.volume_fraction;
    const unsigned int vof_c_index = vof_var.first_component_index;
    const unsigned int vof_ind
      = system_fe.component_to_system_index(vof_c_index, 0);

    const FEVariable<dim> &vofN_var = vof_field.reconstruction;
    const unsigned int vofN_c_index = vofN_var.first_component_index;

    const unsigned int base_element = composition_field.base_element(this->introspection());
    const std::vector<Point<dim> > support_points = system_fe.base_element(base_element).get_unit_support_points();

    for (auto cell : this->get_dof_handler().active_cell_iterators ())
      {
        if (!cell->is_locally_owned())
          continue;

        cell->get_dof_indices (local_dof_indicies);
        const double cell_vof = solution(local_dof_indicies[vof_ind]);

        Tensor<1, dim, double> normal;

        for (unsigned int i=0; i<dim; ++i)
          normal[i] = solution(local_dof_indicies[system_fe
                                                  .component_to_system_index(vofN_c_index+i, 0)]);

        const double d = solution(local_dof_indicies[system_fe
                                                     .component_to_system_index(vofN_c_index+dim, 0)]);
        Tensor<1, dim, double> nnormal;
        double normall1n = 0.0;
        for (unsigned int i=0; i<dim; ++i)
          {
            normall1n += numbers::NumberTraits<double>::abs(normal[i]);
            nnormal[i] = 0.0;
          }
        if (normall1n > volume_fraction_threshold)
          {
            nnormal = normal / normall1n;
          }
        //Calculate correct factor to retain vol frac and [0,1] bound
        double fact = 2.0*(0.5-abs(cell_vof-0.5));
        for (unsigned int i=0; i<system_fe.base_element(base_element).dofs_per_cell; ++i)
          {
            const unsigned int system_local_dof
              = system_fe.component_to_system_index(composition_field.component_index(sim.introspection),
                                                    /*dof index within component*/i);

            Tensor<1, dim, double> uSupp = support_points[i]-uReCen;

            const double value = cell_vof - fact*(uSupp*nnormal);

            initial_solution(local_dof_indicies[system_local_dof]) = value;
          }

      }


    const unsigned int blockidx = composition_field.block_index(this->introspection());
    solution.block(blockidx) = initial_solution.block(blockidx);
    sim.computing_timer.exit_section();
  }

  template <>
  void VoFHandler<3>::update_vof_composition (const typename Simulator<3>::AdvectionField composition_field,
                                              const VoFField<3> vof_field,
                                              LinearAlgebra::BlockVector &solution)
  {
    Assert(false, ExcNotImplemented());
  }
}
