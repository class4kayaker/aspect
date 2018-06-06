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
#include <aspect/volume_of_fluid/handler.h>
#include <aspect/volume_of_fluid/utilities.h>

// #include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/parsed_function.h>

namespace aspect
{
  using namespace dealii;

  template <int dim>
  void VolumeOfFluidHandler<dim>::set_initial_volume_fractions ()
  {
    for (unsigned int f=0; f<n_volume_of_fluid_fields; ++f)
      {
        switch (initialization_data_type[f])
          {
            case VolumeOfFluid::VolumeOfFluidInputType::composition:
              initialize_from_composition_field (data[f], f);
              break;
            case VolumeOfFluid::VolumeOfFluidInputType::level_set:
              initialize_from_level_set (data[f], f);
              break;
            default:
              Assert(false, ExcNotImplemented ());
          }

        const unsigned int volume_of_fluidN_blockidx = data[f].reconstruction.block_index;
        const unsigned int volume_of_fluidLS_blockidx = data[f].level_set.block_index;
        update_volume_of_fluid_normals (data[f], sim.solution);
        sim.old_solution.block(volume_of_fluidN_blockidx) = sim.solution.block(volume_of_fluidN_blockidx);
        sim.old_old_solution.block(volume_of_fluidN_blockidx) = sim.solution.block(volume_of_fluidN_blockidx);
        sim.old_solution.block(volume_of_fluidLS_blockidx) = sim.solution.block(volume_of_fluidLS_blockidx);
        sim.old_old_solution.block(volume_of_fluidLS_blockidx) = sim.solution.block(volume_of_fluidLS_blockidx);

        // Update associated composition field
        const typename Simulator<dim>::AdvectionField composition_field = Simulator<dim>::AdvectionField::composition(data[f].composition_index);
        update_volume_of_fluid_composition (composition_field, data[f], sim.solution);
        const unsigned int volume_of_fluid_C_blockidx = composition_field.block_index(this->introspection());
        sim.old_solution.block(volume_of_fluid_C_blockidx) = sim.solution.block(volume_of_fluid_C_blockidx);
        sim.old_old_solution.block(volume_of_fluid_C_blockidx) = sim.solution.block(volume_of_fluid_C_blockidx);
      }
  }

  template <int dim>
  void VolumeOfFluidHandler<dim>::initialize_from_composition_field (const VolumeOfFluidField<dim> field, const unsigned int f_ind)
  {
    LinearAlgebra::BlockVector initial_solution;

    initial_solution.reinit(sim.system_rhs, false);

    const QIterated<dim> quadrature (QMidpoint<1>(), n_init_samples);
    FEValues<dim, dim> fe_init (this->get_mapping(), this->get_fe(), quadrature,
                                update_JxW_values | update_quadrature_points);

    std::vector<types::global_dof_index>
    local_dof_indicies (this->get_fe().dofs_per_cell);

    const FEVariable<dim> &volume_of_fluid_var = field.volume_fraction;
    const unsigned int component_index = volume_of_fluid_var.first_component_index;
    const unsigned int blockidx = volume_of_fluid_var.block_index;
    const unsigned int volume_of_fluid_ind
      = this->get_fe().component_to_system_index(component_index, 0);

    // Initialize state based on provided function
    for (auto cell : this->get_dof_handler().active_cell_iterators ())
      {
        if (!cell->is_locally_owned ())
          continue;

        // Calculate approximation for volume
        cell->get_dof_indices (local_dof_indicies);

        fe_init.reinit (cell);

        double volume_of_fluid_val = 0.0;
        double cell_vol = 0.0;

        for (unsigned int i = 0; i < fe_init.n_quadrature_points; ++i)
          {
            double ptvolume_of_fluid = this->get_initial_composition_manager().initial_composition(fe_init.quadrature_point(i), f_ind);
            volume_of_fluid_val += ptvolume_of_fluid * fe_init.JxW (i);
            cell_vol += fe_init.JxW(i);
          }

        initial_solution (local_dof_indicies[volume_of_fluid_ind]) = volume_of_fluid_val/cell_vol;
      }

    initial_solution.compress(VectorOperation::insert);

    sim.compute_current_constraints();
    sim.current_constraints.distribute(initial_solution);

    sim.solution.block(blockidx) = initial_solution.block(blockidx);
    sim.old_solution.block(blockidx) = initial_solution.block(blockidx);
    sim.old_old_solution.block(blockidx) = initial_solution.block(blockidx);
  }

  template <int dim>
  void VolumeOfFluidHandler<dim>::initialize_from_level_set (const VolumeOfFluidField<dim> field, const unsigned int f_ind)
  {
    LinearAlgebra::BlockVector initial_solution;

    initial_solution.reinit(sim.system_rhs, false);

    const QIterated<dim> quadrature (QMidpoint<1>(), n_init_samples);
    FEValues<dim, dim> fe_init (this->get_mapping(),
                                this->get_fe(),
                                quadrature,
                                update_JxW_values | update_quadrature_points);

    double h = 1.0/n_init_samples;

    std::vector<types::global_dof_index>
    local_dof_indicies (this->get_fe().dofs_per_cell);

    const FEVariable<dim> &volume_of_fluid_var = field.volume_fraction;
    const unsigned int component_index = volume_of_fluid_var.first_component_index;
    const unsigned int blockidx = volume_of_fluid_var.block_index;
    const unsigned int volume_of_fluid_ind
      = this->get_fe().component_to_system_index(component_index, 0);

    // Initialize state based on provided function
    for (auto cell : this->get_dof_handler().active_cell_iterators ())
      {
        if (!cell->is_locally_owned ())
          continue;

        // Calculate approximation for volume
        double cell_diam, d_func;
        cell->get_dof_indices (local_dof_indicies);

        cell_diam = cell->diameter();
        d_func = this->get_initial_composition_manager().initial_composition(cell->barycenter(), f_ind);
        fe_init.reinit (cell);

        double volume_of_fluid_val = 0.0;
        double cell_vol = 0.0;

        if (d_func <=-0.5*cell_diam)
          {
            volume_of_fluid_val = 0.0;
          }
        else
          {
            if (d_func >= 0.5*cell_diam)
              {
                volume_of_fluid_val = 1.0;
              }
            else
              {

                for (unsigned int i = 0; i < fe_init.n_quadrature_points; ++i)
                  {
                    double d = 0.0;
                    Tensor<1, dim, double> grad;
                    Point<dim> xU = quadrature.point (i);
                    for (unsigned int di = 0; di < dim; ++di)
                      {
                        Point<dim> xH, xL;
                        xH = xU;
                        xL = xU;
                        xH[di] += 0.5*h;
                        xL[di] -= 0.5*h;
                        double dH = this->get_initial_composition_manager().initial_composition(cell->intermediate_point(xH), f_ind);
                        double dL = this->get_initial_composition_manager().initial_composition(cell->intermediate_point(xL), f_ind);
                        grad[di] = (dL-dH);
                        d += (0.5/dim)*(dH+dL);
                      }
                    double ptvolume_of_fluid = VolumeOfFluid::compute_fluid_fraction<dim> (grad, d);
                    volume_of_fluid_val += ptvolume_of_fluid * fe_init.JxW (i);
                    cell_vol += fe_init.JxW (i);
                  }
                volume_of_fluid_val /= cell_vol;
              }
          }

        initial_solution (local_dof_indicies[volume_of_fluid_ind]) = volume_of_fluid_val;
      }

    initial_solution.compress(VectorOperation::insert);

    sim.compute_current_constraints();
    sim.current_constraints.distribute(initial_solution);

    sim.solution.block(blockidx) = initial_solution.block(blockidx);
    sim.old_solution.block(blockidx) = initial_solution.block(blockidx);
    sim.old_old_solution.block(blockidx) = initial_solution.block(blockidx);
  }
}

namespace aspect
{
#define INSTANTIATE(dim) \
  template void VolumeOfFluidHandler<dim>::set_initial_volume_fractions ();\
  template void VolumeOfFluidHandler<dim>::initialize_from_composition_field (const VolumeOfFluidField<dim> field, const unsigned int f_ind); \
  template void VolumeOfFluidHandler<dim>::initialize_from_level_set (const VolumeOfFluidField<dim> field, const unsigned int f_ind);

  ASPECT_INSTANTIATE(INSTANTIATE)
}
