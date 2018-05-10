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

#include <aspect/simulator_access.h>
#include <aspect/utilities.h>
#include <aspect/free_surface.h>
#include <aspect/vof/utilities.h>
#include <aspect/vof/field.h>
#include <aspect/vof/assembly.h>

//#include <deal.II/base/quadrature_lib.h>
//#include <deal.II/lac/full_matrix.h>
//#include <deal.II/lac/constraint_matrix.h>
//#include <deal.II/grid/tria_iterator.h>
//#include <deal.II/grid/filtered_iterator.h>
//#include <deal.II/dofs/dof_accessor.h>
//#include <deal.II/dofs/dof_tools.h>
//#include <deal.II/fe/fe_values.h>

namespace aspect
{
  namespace Assemblers
  {
    template <int dim>
    void VoFAssembler<dim>::set_vof_epsilon(const double value)
    {
      vof_epsilon = value;
    }

    template <int dim>
    void VoFAssembler<dim>::local_assemble_vof_system (const VoFField<dim> field,
                                                       const unsigned int calc_dir,
                                                       bool update_from_old,
                                                       const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                       internal::Assembly::Scratch::VoFSystem<dim> &scratch,
                                                       internal::Assembly::CopyData::VoFSystem<dim> &data) const
    {
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;

      // also have the number of dofs that correspond just to the element for
      // the system we are currently trying to assemble
      const unsigned int vof_dofs_per_cell = data.local_dof_indices.size();

      Assert (vof_dofs_per_cell < scratch.finite_element_values.get_fe().dofs_per_cell, ExcInternalError());
      Assert (vof_dofs_per_cell < scratch.face_finite_element_values.get_fe().dofs_per_cell, ExcInternalError());
      Assert (scratch.phi_field.size() == vof_dofs_per_cell, ExcInternalError());

      const FiniteElement<dim> &main_fe = scratch.finite_element_values.get_fe();

      const unsigned int vofN_component = field.reconstruction.first_component_index;
      const FEValuesExtractors::Vector vofN_n = FEValuesExtractors::Vector(vofN_component);
      const FEValuesExtractors::Scalar vofN_d = FEValuesExtractors::Scalar(vofN_component+dim);

      const unsigned int solution_component = field.volume_fraction.first_component_index;
      const FEValuesExtractors::Scalar solution_field = field.volume_fraction.extractor_scalar();

      scratch.finite_element_values.reinit (cell);

      cell->get_dof_indices (scratch.local_dof_indices);
      for (unsigned int i=0; i<vof_dofs_per_cell; ++i)
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

          scratch.finite_element_values[vofN_n].get_function_values (this->get_old_solution(),
                                                                     scratch.cell_i_n_values);

          scratch.finite_element_values[vofN_d].get_function_values (this->get_old_solution(),
                                                                     scratch.cell_i_d_values);
        }
      else
        {
          scratch.finite_element_values[solution_field].get_function_values (this->get_solution(),
                                                                             scratch.old_field_values);

          scratch.finite_element_values[vofN_n].get_function_values (this->get_solution(),
                                                                     scratch.cell_i_n_values);

          scratch.finite_element_values[vofN_d].get_function_values (this->get_solution(),
                                                                     scratch.cell_i_d_values);
        }

      // Obtain approximation to local interface
      for (unsigned int q = 0; q< n_q_points; ++q)
        {
          // Init FE field vals
          for (unsigned int k=0; k<vof_dofs_per_cell; ++k)
            scratch.phi_field[k] = scratch.finite_element_values[solution_field].value(main_fe.component_to_system_index(solution_component, k), q);

          // Init required local time
          for (unsigned int i = 0; i<vof_dofs_per_cell; ++i)
            {
              data.local_rhs[i] += scratch.old_field_values[q] *
                                   scratch.finite_element_values.JxW(q);
              for (unsigned int j=0; j<vof_dofs_per_cell; ++j)
                data.local_matrix (i, j) += scratch.phi_field[i] *
                                            scratch.phi_field[j] *
                                            scratch.finite_element_values.JxW(q);
            }
        }

      for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
        {
          const unsigned int f_dim = face_no/2; // Obtain dimension

          if (f_dim != calc_dir)
            continue;

          typename DoFHandler<dim>::face_iterator face = cell->face (face_no);

          if (!face->at_boundary())
            {
              this->local_assemble_internal_face_vof_system (field, calc_dir, update_from_old, cell, face_no, scratch, data);
            }
          else
            {
              this->local_assemble_boundary_face_vof_system (field, calc_dir, update_from_old, cell, face_no, scratch, data);
            }
        }
    }

    template <int dim>
    void VoFAssembler<dim>::local_assemble_boundary_face_vof_system (const VoFField<dim> field,
                                                                     const unsigned int calc_dir,
                                                                     bool update_from_old,
                                                                     const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                                     const unsigned int face_no,
                                                                     internal::Assembly::Scratch::VoFSystem<dim> &scratch,
                                                                     internal::Assembly::CopyData::VoFSystem<dim> &data) const
    {
      const bool old_velocity_avail = (this->get_timestep_number() > 0);

      const unsigned int f_dim = face_no/2; // Obtain dimension
      const bool f_dir_pos = (face_no%2==1);

      const unsigned int n_f_q_points    = scratch.face_finite_element_values.n_quadrature_points;

      // volume fraction and interface values are constants, so can set from first value
      const double cell_vol = cell->measure();
      const double cell_vof = scratch.old_field_values[0];
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
          this->local_assemble_internal_face_vof_system(field, calc_dir, update_from_old, cell, face_no, scratch, data);
        }
      else
        {
          // Boundary is not periodic

          double face_flux = 0;
          double face_ls_d = 0;
          double face_ls_time_grad = 0;

          // Using VoF so need to accumulate flux through face
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
          double flux_vof;
          if (std::abs(face_flux) < vof_epsilon*cell_vol)
            {
              flux_vof = cell_vof;
            }
          else if (face_flux < 0.0) // edge is upwind, currently assume zero inflow
            {
              flux_vof = 0.0;
            }
          else // Cell is upwind, outflow boundary
            {
              flux_vof = VolumeOfFluid::calc_vof_flux_edge<dim> (f_dim,
                                                                 face_ls_time_grad,
                                                                 cell_i_normal,
                                                                 face_ls_d);
            }

          //TODO: Handle non-zero inflow VoF boundary conditions

          // Add fluxes to RHS
          data.local_rhs[0] -= (flux_vof-cell_vof) * face_flux;
        }
    }

    template <int dim>
    void VoFAssembler<dim>::local_assemble_internal_face_vof_system (const VoFField<dim> field,
                                                                     const unsigned int calc_dir,
                                                                     bool update_from_old,
                                                                     const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                                     const unsigned int face_no,
                                                                     internal::Assembly::Scratch::VoFSystem<dim> &scratch,
                                                                     internal::Assembly::CopyData::VoFSystem<dim> &data) const
    {
      const bool old_velocity_avail = (this->get_timestep_number() > 0);

      const unsigned int f_dim = face_no/2; // Obtain dimension
      const bool f_dir_pos = (face_no%2==1);

      const unsigned int n_f_q_points    = scratch.face_finite_element_values.n_quadrature_points;

      // vol fraction and interface values are constants, so can set from first value
      const double cell_vol = cell->measure();
      const double cell_vof = scratch.old_field_values[0];
      const Tensor<1, dim, double> cell_i_normal = scratch.cell_i_n_values[0];
      const double cell_i_d = scratch.cell_i_d_values[0];

      const FiniteElement<dim> &main_fe = scratch.finite_element_values.get_fe();

      // also have the number of dofs that correspond just to the element for
      // the system we are currently trying to assemble
      const unsigned int vof_dofs_per_cell = data.local_dof_indices.size();

      const unsigned int solution_component = field.volume_fraction.first_component_index;
      const FEValuesExtractors::Scalar solution_field = field.volume_fraction.extractor_scalar();

      const unsigned int vofN_component = field.reconstruction.first_component_index;
      const FEValuesExtractors::Vector vofN_n = FEValuesExtractors::Vector(vofN_component);
      const FEValuesExtractors::Scalar vofN_d = FEValuesExtractors::Scalar(vofN_component+dim);

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
                  scratch.neighbor_finite_element_values[vofN_n]
                  .get_function_values(this->get_old_solution(),
                                       scratch.neighbor_i_n_values);
                  scratch.neighbor_finite_element_values[vofN_d]
                  .get_function_values(this->get_old_solution(),
                                       scratch.neighbor_i_d_values);
                }
              else
                {
                  scratch.neighbor_finite_element_values[solution_field]
                  .get_function_values(this->get_solution(),
                                       scratch.neighbor_old_values);
                  scratch.neighbor_finite_element_values[vofN_n]
                  .get_function_values(this->get_solution(),
                                       scratch.neighbor_i_n_values);
                  scratch.neighbor_finite_element_values[vofN_d]
                  .get_function_values(this->get_solution(),
                                       scratch.neighbor_i_d_values);
                }

              const double neighbor_vol = neighbor->measure();
              const double neighbor_vof = scratch.neighbor_old_values[0];
              const Tensor<1, dim, double> neighbor_i_normal = scratch.neighbor_i_n_values[0];
              const double neighbor_i_d = scratch.neighbor_i_d_values[0];

              double face_flux = 0;
              double face_ls_d = 0;
              double face_ls_time_grad = 0;
              double n_face_ls_d = 0;
              double n_face_ls_time_grad =0;

              // Using VoF so need to accumulate flux through face
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
              double flux_vof;
              if (std::abs(face_flux) < 0.5*vof_epsilon*(cell_vol+neighbor_vol))
                {
                  flux_vof = 0.5*(cell_vof+neighbor_vof);
                }
              else if (face_flux < 0.0) // Neighbor is upwind
                {
                  flux_vof = VolumeOfFluid::calc_vof_flux_edge<dim> (n_f_dim,
                                                                     n_face_ls_time_grad,
                                                                     neighbor_i_normal,
                                                                     n_face_ls_d);
                }
              else // This cell is upwind
                {
                  flux_vof = VolumeOfFluid::calc_vof_flux_edge<dim> (f_dim,
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

              for (unsigned int i=0; i<vof_dofs_per_cell; ++i)
                data.neighbor_dof_indices[f_rhs_ind][i]
                  = neighbor_dof_indices[main_fe.component_to_system_index(solution_component, i)];

              data.face_contributions_mask[f_rhs_ind] = true;

              // fluxes to RHS
              data.local_rhs [0] -= (flux_vof-cell_vof) * face_flux;
              data.local_face_rhs[f_rhs_ind][0] += (flux_vof-neighbor_vof) * face_flux;
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
                  scratch.neighbor_finite_element_values[vofN_n]
                  .get_function_values(this->get_old_solution(),
                                       scratch.neighbor_i_n_values);
                  scratch.neighbor_finite_element_values[vofN_d]
                  .get_function_values(this->get_old_solution(),
                                       scratch.neighbor_i_d_values);
                }
              else
                {
                  scratch.neighbor_finite_element_values[solution_field]
                  .get_function_values(this->get_solution(),
                                       scratch.neighbor_old_values);
                  scratch.neighbor_finite_element_values[vofN_n]
                  .get_function_values(this->get_solution(),
                                       scratch.neighbor_i_n_values);
                  scratch.neighbor_finite_element_values[vofN_d]
                  .get_function_values(this->get_solution(),
                                       scratch.neighbor_i_d_values);
                }

              const double neighbor_vol = neighbor->measure();
              const double neighbor_vof = scratch.neighbor_old_values[0];
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

              // Using VoF so need to accumulate flux through face
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

              for (unsigned int i=0; i<vof_dofs_per_cell; ++i)
                data.neighbor_dof_indices[f_rhs_ind][i]
                  = neighbor_dof_indices[main_fe.component_to_system_index(solution_component, i)];

              data.face_contributions_mask[f_rhs_ind] = true;

              // fluxes to RHS
              double flux_vof = cell_vof;
              if (std::abs(face_flux) < 0.5*vof_epsilon*(cell_vol+neighbor_vol))
                {
                  flux_vof = 0.5*(cell_vof+neighbor_vof);
                }
              if (flux_vof < 0.0)
                flux_vof = 0.0;
              if (flux_vof > 1.0)
                flux_vof = 1.0;

              data.local_rhs [0] -= (flux_vof-cell_vof) * face_flux;
              data.local_face_rhs[f_rhs_ind][0] += (flux_vof-neighbor_vof) * face_flux;

              // Limit to constant cases, otherwise announce error
              if (cell_vof > vof_epsilon && cell_vof<1.0-vof_epsilon)
                {
                  this->get_pcout() << "Cell at " << cell->center() << " " << cell_vof << std::endl
                                    << "\t" << face_flux/this->get_timestep()/cell_vol << std::endl
                                    << "\t" << cell_i_normal << ".x=" << cell_i_d << std::endl;
                  Assert(false, ExcNotImplemented());
                }
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
  template class VoFAssembler<dim>;

    ASPECT_INSTANTIATE(INSTANTIATE)

  }
}
