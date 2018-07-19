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

#include <aspect/simulator_access.h>
#include <aspect/utilities.h>
#include <aspect/free_surface.h>
#include <aspect/volume_of_fluid/utilities.h>
#include <aspect/volume_of_fluid/handler.h>
#include <aspect/volume_of_fluid/field.h>
#include <aspect/volume_of_fluid/assembly.h>


namespace aspect
{
  namespace Assemblers
  {
    template <int dim>
    void VolumeOfFluidAssembler<dim>::local_assemble_volume_of_fluid_system (const VolumeOfFluidField<dim> field,
                                                                             const unsigned int calc_dir,
                                                                             const bool update_from_old,
                                                                             const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                                             internal::Assembly::Scratch::VolumeOfFluidSystem<dim> &scratch,
                                                                             internal::Assembly::CopyData::VolumeOfFluidSystem<dim> &data) const
    {
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;

      // also have the number of dofs that correspond just to the element for
      // the system we are currently trying to assemble
      const unsigned int volume_of_fluid_dofs_per_cell = data.local_dof_indices.size();

      Assert (volume_of_fluid_dofs_per_cell < scratch.finite_element_values.get_fe().dofs_per_cell, ExcInternalError());
      Assert (volume_of_fluid_dofs_per_cell < scratch.face_finite_element_values.get_fe().dofs_per_cell, ExcInternalError());
      Assert (scratch.phi_field.size() == volume_of_fluid_dofs_per_cell, ExcInternalError());

      // Need to have MappingCartesioan for current cell volume implementation to be correct

      const FiniteElement<dim> &main_fe = scratch.finite_element_values.get_fe();

      const unsigned int volume_of_fluidN_component = field.reconstruction.first_component_index;
      const FEValuesExtractors::Vector volume_of_fluidN_n = FEValuesExtractors::Vector(volume_of_fluidN_component);
      const FEValuesExtractors::Scalar volume_of_fluidN_d = FEValuesExtractors::Scalar(volume_of_fluidN_component+dim);

      const unsigned int solution_component = field.volume_fraction.first_component_index;
      const FEValuesExtractors::Scalar solution_field = field.volume_fraction.extractor_scalar();

      scratch.finite_element_values.reinit (cell);

      cell->get_dof_indices (scratch.local_dof_indices);
      for (unsigned int i=0; i<volume_of_fluid_dofs_per_cell; ++i)
        data.local_dof_indices[i] = scratch.local_dof_indices[main_fe.component_to_system_index(solution_component, i)];

      data.local_matrix = 0;
      data.local_rhs = 0;

      //loop over all possible subfaces of the cell, and reset corresponding rhs
      for (unsigned int f = 0; f < GeometryInfo<dim>::max_children_per_face * GeometryInfo<dim>::faces_per_cell; ++f)
        {
          data.local_face_rhs[f] = 0.0;
          data.local_face_matrices_ext_ext[f] = 0.0;
          data.face_contributions_mask[f] = false;
        }

      if (update_from_old)
        {
          scratch.finite_element_values[solution_field].get_function_values (this->get_old_solution(),
                                                                             scratch.old_field_values);

          scratch.finite_element_values[volume_of_fluidN_n].get_function_values (this->get_old_solution(),
                                                                                 scratch.cell_i_n_values);

          scratch.finite_element_values[volume_of_fluidN_d].get_function_values (this->get_old_solution(),
                                                                                 scratch.cell_i_d_values);
        }
      else
        {
          scratch.finite_element_values[solution_field].get_function_values (this->get_solution(),
                                                                             scratch.old_field_values);

          scratch.finite_element_values[volume_of_fluidN_n].get_function_values (this->get_solution(),
                                                                                 scratch.cell_i_n_values);

          scratch.finite_element_values[volume_of_fluidN_d].get_function_values (this->get_solution(),
                                                                                 scratch.cell_i_d_values);
        }

      scratch.volume = 0.0;

      for (unsigned int q = 0; q< n_q_points; ++q)
        {
          // Init FE field vals
          for (unsigned int k=0; k<volume_of_fluid_dofs_per_cell; ++k)
            scratch.phi_field[k] = scratch.finite_element_values[solution_field].value(main_fe.component_to_system_index(solution_component, k), q);

          for (unsigned int i = 0; i<volume_of_fluid_dofs_per_cell; ++i)
            {
              data.local_rhs[i] += scratch.old_field_values[q] *
                                   scratch.finite_element_values.JxW(q);
              for (unsigned int j=0; j<volume_of_fluid_dofs_per_cell; ++j)
                data.local_matrix (i, j) += scratch.phi_field[i] *
                                            scratch.phi_field[j] *
                                            scratch.finite_element_values.JxW(q);
            }
          scratch.volume += scratch.finite_element_values.JxW(q);
        }

      for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
        {
          // Obtain the normal direction for the face in question
          // Deall.II orders faces as dim*2+(face_direction_is_positive?1:0)
          const unsigned int face_normal_direction = face_no/2;

          if (face_normal_direction != calc_dir)
            continue;

          typename DoFHandler<dim>::face_iterator face = cell->face (face_no);

          if (!face->at_boundary())
            {
              this->local_assemble_internal_face_volume_of_fluid_system (field, calc_dir, update_from_old, cell, face_no, scratch, data);
            }
          else
            {
              this->local_assemble_boundary_face_volume_of_fluid_system (field, calc_dir, update_from_old, cell, face_no, scratch, data);
            }
        }
    }

    template <int dim>
    void VolumeOfFluidAssembler<dim>::local_assemble_boundary_face_volume_of_fluid_system (const VolumeOfFluidField<dim> field,
        const unsigned int calc_dir,
        const bool update_from_old,
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        const unsigned int face_no,
        internal::Assembly::Scratch::VolumeOfFluidSystem<dim> &scratch,
        internal::Assembly::CopyData::VolumeOfFluidSystem<dim> &data) const
    {
      const double volume_fraction_threshold = this->get_volume_of_fluid_handler().get_volume_fraction_threshold();

      const bool old_velocity_avail = (this->get_timestep_number() > 0);

      const typename DoFHandler<dim>::face_iterator face = cell->face(face_no);

      const unsigned int f_dim = face_no/2; // Obtain dimension
      const bool f_dir_pos = (face_no%2==1);

      const unsigned int n_f_q_points    = scratch.face_finite_element_values.n_quadrature_points;

      // Currently assuming cartesian mapping, so cell->measure()
      // works, and the neighbor volume cannot be computed easily using
      // a sum in this call.
      const double cell_vol = scratch.volume;

      // volume fraction and interface values are constants, so can set from first value
      const double cell_volume_of_fluid = scratch.old_field_values[0];
      const Tensor<1, dim, double> cell_i_normal = scratch.cell_i_n_values[0];
      const double cell_i_d = scratch.cell_i_d_values[0];

      // also have the number of dofs that correspond just to the element for
      // the system we are currently trying to assemble

      scratch.face_finite_element_values.reinit (cell, face_no);

      scratch.face_finite_element_values[this->introspection().extractors.velocities]
      .get_function_values (this->get_current_linearization_point(),
                            scratch.face_current_velocity_values);

      scratch.face_finite_element_values[this->introspection().extractors.velocities]
      .get_function_values (this->get_old_solution(),
                            scratch.face_old_velocity_values);

      if (this->get_parameters().free_surface_enabled)
        scratch.face_finite_element_values[this->introspection().extractors.velocities]
        .get_function_values (this->get_mesh_velocity(),
                              scratch.face_mesh_velocity_values);

      if (cell->has_periodic_neighbor (face_no))
        {
          // Periodic temperature/composition term: consider the corresponding periodic faces as the case of interior faces
          this->local_assemble_internal_face_volume_of_fluid_system(field, calc_dir, update_from_old, cell, face_no, scratch, data);
        }
      else
        {
          // Boundary is not periodic

          double face_flux = 0;
          double face_ls_d = 0;
          double face_ls_time_grad = 0;
          double boundary_fluid_flux = 0;

          // Using VolumeOfFluid so need to accumulate flux through face
          if ( this->get_fixed_composition_boundary_indicators().find(
                 face->boundary_id()
               )
               != this->get_fixed_composition_boundary_indicators().end())
            {
              for (unsigned int q=0; q<n_f_q_points; ++q)
                {

                  Tensor<1,dim> current_u = scratch.face_current_velocity_values[q];

                  //If old velocity available average to half timestep
                  if (old_velocity_avail)
                    current_u += 0.5*(scratch.face_old_velocity_values[q] -
                                      scratch.face_current_velocity_values[q]);

                  //Subtract off the mesh velocity for ALE corrections if necessary
                  if (this->get_parameters().free_surface_enabled)
                    current_u -= scratch.face_mesh_velocity_values[q];

                  face_flux += this->get_timestep() *
                               current_u *
                               scratch.face_finite_element_values.normal_vector(q) *
                               scratch.face_finite_element_values.JxW(q);

                }
            }
          else
            {
              for (unsigned int q=0; q<n_f_q_points; ++q)
                {

                  Tensor<1,dim> current_u = scratch.face_current_velocity_values[q];

                  //If old velocity available average to half timestep
                  if (old_velocity_avail)
                    current_u += 0.5*(scratch.face_old_velocity_values[q] -
                                      scratch.face_current_velocity_values[q]);

                  //Subtract off the mesh velocity for ALE corrections if necessary
                  if (this->get_parameters().free_surface_enabled)
                    current_u -= scratch.face_mesh_velocity_values[q];

                  face_flux += this->get_timestep() *
                               current_u *
                               scratch.face_finite_element_values.normal_vector(q) *
                               scratch.face_finite_element_values.JxW(q);

                  boundary_fluid_flux += this->get_boundary_composition_manager().boundary_composition(
                                           face->boundary_id(),
                                           scratch.face_finite_element_values.quadrature_point(q),
                                           field.composition_index) *
                                         current_u *
                                         scratch.face_finite_element_values.normal_vector(q) *
                                         scratch.face_finite_element_values.JxW(q);

                }
            }

          // Due to inability to reference this cell's values at the interface,
          // need to do explicit calculation
          if (f_dir_pos)
            {
              face_ls_d = cell_i_d - 0.5*cell_i_normal[f_dim];
              face_ls_time_grad = (face_flux/cell_vol)*cell_i_normal[f_dim];
            }
          else
            {
              face_ls_d = cell_i_d + 0.5*cell_i_normal[f_dim];
              face_ls_time_grad = -(face_flux/cell_vol)*cell_i_normal[f_dim];
            }

          // Calculate outward flux
          double flux_volume_of_fluid;
          if (std::abs(face_flux) < volume_fraction_threshold*cell_vol)
            {
              flux_volume_of_fluid = cell_volume_of_fluid;
            }
          else if (face_flux < 0.0) // edge is upwind
            {
              flux_volume_of_fluid = -boundary_fluid_flux/face_flux;
            }
          else // Cell is upwind, outflow boundary
            {
              flux_volume_of_fluid = VolumeOfFluid::calc_volume_of_fluid_flux_edge<dim> (f_dim,
                                                                                         face_ls_time_grad,
                                                                                         cell_i_normal,
                                                                                         face_ls_d);
            }

          // Add fluxes to RHS
          if (update_from_old)
            {
              data.local_matrix(0, 0) -= face_flux;
              data.local_rhs[0] -= (flux_volume_of_fluid) * face_flux;
            }
          else
            {
              data.local_rhs[0] -= (flux_volume_of_fluid-cell_volume_of_fluid) * face_flux;
            }
        }
    }

    template <int dim>
    void VolumeOfFluidAssembler<dim>::local_assemble_internal_face_volume_of_fluid_system (const VolumeOfFluidField<dim> field,
        const unsigned int calc_dir,
        bool update_from_old,
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        const unsigned int face_no,
        internal::Assembly::Scratch::VolumeOfFluidSystem<dim> &scratch,
        internal::Assembly::CopyData::VolumeOfFluidSystem<dim> &data) const
    {
      const double volume_fraction_threshold = this->get_volume_of_fluid_handler().get_volume_fraction_threshold();
      const bool old_velocity_avail = (this->get_timestep_number() > 0);

      const unsigned int f_dim = face_no/2; // Obtain dimension
      const bool f_dir_pos = (face_no%2==1);

      const unsigned int n_f_q_points    = scratch.face_finite_element_values.n_quadrature_points;

      // Currently assuming cartesian mapping, so cell->measure() works, we
      // also will need neighbor volume, which cannot be computed easily using
      // a sum in this call.
      const double cell_vol = scratch.volume;

      // vol fraction and interface values are constants, so can set from first value
      const double cell_volume_of_fluid = scratch.old_field_values[0];
      const Tensor<1, dim, double> cell_i_normal = scratch.cell_i_n_values[0];
      const double cell_i_d = scratch.cell_i_d_values[0];

      const FiniteElement<dim> &main_fe = scratch.finite_element_values.get_fe();

      // also have the number of dofs that correspond just to the element for
      // the system we are currently trying to assemble
      const unsigned int volume_of_fluid_dofs_per_cell = data.local_dof_indices.size();

      const unsigned int solution_component = field.volume_fraction.first_component_index;
      const FEValuesExtractors::Scalar solution_field = field.volume_fraction.extractor_scalar();

      const unsigned int volume_of_fluidN_component = field.reconstruction.first_component_index;
      const FEValuesExtractors::Vector volume_of_fluidN_n = FEValuesExtractors::Vector(volume_of_fluidN_component);
      const FEValuesExtractors::Scalar volume_of_fluidN_d = FEValuesExtractors::Scalar(volume_of_fluidN_component+dim);

      const typename DoFHandler<dim>::face_iterator face = cell->face (face_no);

      scratch.face_finite_element_values.reinit (cell, face_no);

      scratch.face_finite_element_values[this->introspection().extractors.velocities]
      .get_function_values (this->get_current_linearization_point(),
                            scratch.face_current_velocity_values);

      scratch.face_finite_element_values[this->introspection().extractors.velocities]
      .get_function_values (this->get_old_solution(),
                            scratch.face_old_velocity_values);

      if (this->get_parameters().free_surface_enabled)
        scratch.face_finite_element_values[this->introspection().extractors.velocities]
        .get_function_values (this->get_mesh_velocity(),
                              scratch.face_mesh_velocity_values);

      // interior face or periodic face - no contribution on RHS

      const typename DoFHandler<dim>::cell_iterator
      neighbor = cell->neighbor_or_periodic_neighbor (face_no);
      // note: "neighbor" defined above is NOT active_cell_iterator, so this includes cells that are refined
      // for example: cell with periodic boundary.
      Assert (neighbor.state() == IteratorState::valid,
              ExcInternalError());
      const bool cell_has_periodic_neighbor = cell->has_periodic_neighbor (face_no);
      const unsigned int neighbor_face_no = (cell_has_periodic_neighbor
                                             ?
                                             cell->periodic_neighbor_face_no(face_no)
                                             :
                                             cell->neighbor_face_no(face_no));

      const unsigned int n_f_dim = neighbor_face_no/2;
      const bool n_f_dir_pos = (neighbor_face_no%2==1);

      if ((!face->at_boundary() && !face->has_children())
          ||
          (face->at_boundary() && cell->periodic_neighbor_is_coarser(face_no))
          ||
          (face->at_boundary() && neighbor->level() == cell->level() && neighbor->active()))
        {
          if (neighbor->level () == cell->level () &&
              neighbor->active() &&
              (((neighbor->is_locally_owned()) && (cell->index() < neighbor->index()))
               ||
               ((!neighbor->is_locally_owned()) && (cell->subdomain_id() < neighbor->subdomain_id()))))
            {
              Assert (cell->is_locally_owned(), ExcInternalError());


              // Neighbor cell values

              scratch.neighbor_finite_element_values.reinit(neighbor);

              if (update_from_old)
                {
                  scratch.neighbor_finite_element_values[solution_field]
                  .get_function_values(this->get_old_solution(),
                                       scratch.neighbor_old_values);
                  scratch.neighbor_finite_element_values[volume_of_fluidN_n]
                  .get_function_values(this->get_old_solution(),
                                       scratch.neighbor_i_n_values);
                  scratch.neighbor_finite_element_values[volume_of_fluidN_d]
                  .get_function_values(this->get_old_solution(),
                                       scratch.neighbor_i_d_values);
                }
              else
                {
                  scratch.neighbor_finite_element_values[solution_field]
                  .get_function_values(this->get_solution(),
                                       scratch.neighbor_old_values);
                  scratch.neighbor_finite_element_values[volume_of_fluidN_n]
                  .get_function_values(this->get_solution(),
                                       scratch.neighbor_i_n_values);
                  scratch.neighbor_finite_element_values[volume_of_fluidN_d]
                  .get_function_values(this->get_solution(),
                                       scratch.neighbor_i_d_values);
                }

              // Currently assuming cartesian mapping, so cell->measure()
              // works, and the neighbor volume cannot be computed easily using
              // a sum in this call.
              const double neighbor_vol = neighbor->measure();
              const double neighbor_volume_of_fluid = scratch.neighbor_old_values[0];
              const Tensor<1, dim, double> neighbor_i_normal = scratch.neighbor_i_n_values[0];
              const double neighbor_i_d = scratch.neighbor_i_d_values[0];

              double face_flux = 0;
              double face_ls_d = 0;
              double face_ls_time_grad = 0;
              double n_face_ls_d = 0;
              double n_face_ls_time_grad =0;

              // Using VolumeOfFluid so need to accumulate flux through face
              for (unsigned int q=0; q<n_f_q_points; ++q)
                {

                  Tensor<1,dim> current_u = scratch.face_current_velocity_values[q];

                  //If old velocity available average to half timestep
                  if (old_velocity_avail)
                    current_u += 0.5*(scratch.face_old_velocity_values[q] -
                                      scratch.face_current_velocity_values[q]);

                  //Subtract off the mesh velocity for ALE corrections if necessary
                  if (this->get_parameters().free_surface_enabled)
                    current_u -= scratch.face_mesh_velocity_values[q];

                  face_flux += this->get_timestep() *
                               current_u *
                               scratch.face_finite_element_values.normal_vector(q) *
                               scratch.face_finite_element_values.JxW(q);

                }

              // Because the level set data is stored implicitly as a set of
              // generating data, the calculation for the relevant normal time
              // gradient and value of d at the face center must be computed
              // here
              if (f_dir_pos)
                {
                  face_ls_d = cell_i_d - 0.5*cell_i_normal[f_dim];
                  face_ls_time_grad = (face_flux/cell_vol)*cell_i_normal[f_dim];
                }
              else
                {
                  face_ls_d = cell_i_d + 0.5*cell_i_normal[f_dim];
                  face_ls_time_grad = -(face_flux/cell_vol)*cell_i_normal[f_dim];
                }

              if (n_f_dir_pos)
                {
                  n_face_ls_d = neighbor_i_d - 0.5*neighbor_i_normal[n_f_dim];
                  n_face_ls_time_grad = -(face_flux/neighbor_vol)*neighbor_i_normal[n_f_dim];
                }
              else
                {
                  n_face_ls_d = neighbor_i_d + 0.5*neighbor_i_normal[n_f_dim];
                  n_face_ls_time_grad = (face_flux/neighbor_vol)*neighbor_i_normal[n_f_dim];
                }

              // Calculate outward flux
              double flux_volume_of_fluid;
              if (std::abs(face_flux) < 0.5*volume_fraction_threshold*(cell_vol+neighbor_vol))
                {
                  flux_volume_of_fluid = 0.5*(cell_volume_of_fluid+neighbor_volume_of_fluid);
                }
              else if (face_flux < 0.0) // Neighbor is upwind
                {
                  flux_volume_of_fluid = VolumeOfFluid::calc_volume_of_fluid_flux_edge<dim> (n_f_dim,
                                                                                             n_face_ls_time_grad,
                                                                                             neighbor_i_normal,
                                                                                             n_face_ls_d);
                }
              else // This cell is upwind
                {
                  flux_volume_of_fluid = VolumeOfFluid::calc_volume_of_fluid_flux_edge<dim> (f_dim,
                                                                                             face_ls_time_grad,
                                                                                             cell_i_normal,
                                                                                             face_ls_d);
                }

              Assert (neighbor.state() == IteratorState::valid,
                      ExcInternalError());

              // No children, so can do simple approach
              Assert (cell->is_locally_owned(), ExcInternalError());
              //cell and neighbor are equal-sized, and cell has been chosen to assemble this face, so calculate from cell

              std::vector<types::global_dof_index> neighbor_dof_indices (main_fe.dofs_per_cell);
              // get all dof indices on the neighbor, then extract those
              // that correspond to the solution_field we are interested in
              neighbor->get_dof_indices (neighbor_dof_indices);

              const unsigned int f_rhs_ind = face_no * GeometryInfo<dim>::max_children_per_face;

              for (unsigned int i=0; i<volume_of_fluid_dofs_per_cell; ++i)
                data.neighbor_dof_indices[f_rhs_ind][i]
                  = neighbor_dof_indices[main_fe.component_to_system_index(solution_component, i)];

              data.face_contributions_mask[f_rhs_ind] = true;

              // fluxes to RHS
              if (update_from_old)
                {
                  data.local_matrix(0, 0) -= face_flux;
                  data.local_face_matrices_ext_ext[f_rhs_ind](0, 0) += face_flux;
                  data.local_rhs [0] -= (flux_volume_of_fluid) * face_flux;
                  data.local_face_rhs[f_rhs_ind][0] += (flux_volume_of_fluid) * face_flux;
                }
              else
                {
                  data.local_rhs [0] -= (flux_volume_of_fluid-cell_volume_of_fluid) * face_flux;
                  data.local_face_rhs[f_rhs_ind][0] += (flux_volume_of_fluid-neighbor_volume_of_fluid) * face_flux;
                }
            }
          else
            {
              /* neighbor is taking responsibility for assembly of this face, because
               * either (1) neighbor is coarser, or
               *        (2) neighbor is equally-sized and
               *           (a) neighbor is on a different subdomain, with lower subdmain_id(), or
               *           (b) neighbor is on the same subdomain and has lower index().
              */
            }
        }
      else // face->has_children() so always assemble from here
        {
          for (unsigned int subface_no=0; subface_no< face->number_of_children(); ++subface_no)
            {
              const typename DoFHandler<dim>::active_cell_iterator neighbor_child
                = ( cell_has_periodic_neighbor
                    ?
                    cell->periodic_neighbor_child_on_subface(face_no, subface_no)
                    :
                    cell->neighbor_child_on_subface (face_no, subface_no));

              // Neighbor cell values

              scratch.neighbor_finite_element_values.reinit(neighbor_child);

              if (update_from_old)
                {
                  scratch.neighbor_finite_element_values[solution_field]
                  .get_function_values(this->get_old_solution(),
                                       scratch.neighbor_old_values);
                  scratch.neighbor_finite_element_values[volume_of_fluidN_n]
                  .get_function_values(this->get_old_solution(),
                                       scratch.neighbor_i_n_values);
                  scratch.neighbor_finite_element_values[volume_of_fluidN_d]
                  .get_function_values(this->get_old_solution(),
                                       scratch.neighbor_i_d_values);
                }
              else
                {
                  scratch.neighbor_finite_element_values[solution_field]
                  .get_function_values(this->get_solution(),
                                       scratch.neighbor_old_values);
                  scratch.neighbor_finite_element_values[volume_of_fluidN_n]
                  .get_function_values(this->get_solution(),
                                       scratch.neighbor_i_n_values);
                  scratch.neighbor_finite_element_values[volume_of_fluidN_d]
                  .get_function_values(this->get_solution(),
                                       scratch.neighbor_i_d_values);
                }

              // Currently assuming cartesian mapping, so cell->measure()
              // works, and the neighbor volume cannot be computed easily using
              // a sum in this call.
              const double neighbor_vol = neighbor->measure();
              const double neighbor_volume_of_fluid = scratch.neighbor_old_values[0];
              // Unneeded neighbor data
              // const Tensor<1, dim, double> neighbor_i_normal = scratch.neighbor_i_n_values[0];
              // const double neighbor_i_d = scratch.neighbor_i_d_values[0];

              scratch.subface_finite_element_values.reinit (cell, face_no, subface_no);

              scratch.subface_finite_element_values[this->introspection().extractors.velocities]
              .get_function_values (this->get_current_linearization_point(),
                                    scratch.face_current_velocity_values);

              scratch.subface_finite_element_values[this->introspection().extractors.velocities]
              .get_function_values (this->get_old_solution(),
                                    scratch.face_old_velocity_values);

              if (this->get_parameters().free_surface_enabled)
                scratch.subface_finite_element_values[this->introspection().extractors.velocities]
                .get_function_values (this->get_mesh_velocity(),
                                      scratch.face_mesh_velocity_values);

              double face_flux = 0;
              // Unneeded interface flux data
              // double face_ls_d = 0;
              // double face_ls_time_grad = 0;

              // Using VolumeOfFluid so need to accumulate flux through face
              for (unsigned int q=0; q<n_f_q_points; ++q)
                {

                  Tensor<1,dim> current_u = scratch.face_current_velocity_values[q];

                  //If old velocity available average to half timestep
                  if (old_velocity_avail)
                    current_u += 0.5*(scratch.face_old_velocity_values[q] -
                                      scratch.face_current_velocity_values[q]);

                  //Subtract off the mesh velocity for ALE corrections if necessary
                  if (this->get_parameters().free_surface_enabled)
                    current_u -= scratch.face_mesh_velocity_values[q];

                  face_flux += this->get_timestep() *
                               current_u *
                               scratch.subface_finite_element_values.normal_vector(q) *
                               scratch.subface_finite_element_values.JxW(q);

                }

              std::vector<types::global_dof_index> neighbor_dof_indices (scratch.subface_finite_element_values.get_fe().dofs_per_cell);
              neighbor_child->get_dof_indices (neighbor_dof_indices);

              const unsigned int f_rhs_ind = face_no * GeometryInfo<dim>::max_children_per_face+subface_no;

              for (unsigned int i=0; i<volume_of_fluid_dofs_per_cell; ++i)
                data.neighbor_dof_indices[f_rhs_ind][i]
                  = neighbor_dof_indices[main_fe.component_to_system_index(solution_component, i)];

              data.face_contributions_mask[f_rhs_ind] = true;

              // fluxes to RHS
              double flux_volume_of_fluid = face_flux>0.0?cell_volume_of_fluid:neighbor_volume_of_fluid;
              if (std::abs(face_flux) < 0.5*volume_fraction_threshold*(cell_vol+neighbor_vol))
                {
                  flux_volume_of_fluid = 0.5*(cell_volume_of_fluid+neighbor_volume_of_fluid);
                }
              if (flux_volume_of_fluid < 0.0)
                flux_volume_of_fluid = 0.0;
              if (flux_volume_of_fluid > 1.0)
                flux_volume_of_fluid = 1.0;

              if (update_from_old)
                {
                  data.local_matrix(0, 0) -= face_flux;
                  data.local_face_matrices_ext_ext[f_rhs_ind](0, 0) += face_flux;
                  data.local_rhs [0] -= (flux_volume_of_fluid) * face_flux;
                  data.local_face_rhs[f_rhs_ind][0] += (flux_volume_of_fluid) * face_flux;
                }
              else
                {
                  data.local_rhs [0] -= (flux_volume_of_fluid-cell_volume_of_fluid) * face_flux;
                  data.local_face_rhs[f_rhs_ind][0] += (flux_volume_of_fluid-neighbor_volume_of_fluid) * face_flux;
                }

              AssertThrow(flux_volume_of_fluid < volume_fraction_threshold || flux_volume_of_fluid>1.0-volume_fraction_threshold,
                          ExcMessage("Face between multiple refinement levels with fluid interface producing fraction " +
                              Utilities::to_string(flux_volume_of_fluid) +
                              ". Levels: " +
                              Utilities::int_to_string(cell->level()) +
                              " and " +
                              Utilities::int_to_string(neighbor_child->level()) +
                              ". This case is not handled by the VOF assembler."));
            }
        }
    }
  }
}

namespace aspect
{
  namespace Assemblers
  {
#define INSTANTIATE(dim) \
  template class VolumeOfFluidAssembler<dim>;

    ASPECT_INSTANTIATE(INSTANTIATE)

  }
}
