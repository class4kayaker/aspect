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

#include <aspect/simulator.h>
#include <aspect/utilities.h>
#include <aspect/free_surface.h>
#include <aspect/vof/handler.h>
#include <aspect/vof/utilities.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/signaling_nan.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>

namespace aspect
{
  namespace internal
  {
    namespace Assembly
    {
      namespace Scratch
      {
        template <int dim>
        VoFSystem<dim>::VoFSystem (const FiniteElement<dim> &finite_element,
                                   const FiniteElement<dim> &vof_element,
                                   const Mapping<dim>       &mapping,
                                   const Quadrature<dim>    &quadrature,
                                   const Quadrature<dim-1>  &face_quadrature)
          :
          finite_element_values (mapping,
                                 finite_element, quadrature,
                                 update_values |
                                 update_gradients |
                                 update_JxW_values),
          neighbor_finite_element_values (mapping,
                                          finite_element, quadrature,
                                          update_values |
                                          update_gradients |
                                          update_JxW_values),
          face_finite_element_values (mapping,
                                      finite_element, face_quadrature,
                                      update_values |
                                      update_gradients |
                                      update_normal_vectors |
                                      update_JxW_values),
          neighbor_face_finite_element_values (mapping,
                                               finite_element, face_quadrature,
                                               update_values |
                                               update_gradients |
                                               update_normal_vectors |
                                               update_JxW_values),
          subface_finite_element_values (mapping,
                                         finite_element, face_quadrature,
                                         update_values |
                                         update_gradients |
                                         update_normal_vectors |
                                         update_JxW_values),
          local_dof_indices(finite_element.dofs_per_cell),
          phi_field (vof_element.dofs_per_cell, numbers::signaling_nan<double>()),
          old_field_values (quadrature.size(), numbers::signaling_nan<double>()),
          cell_i_n_values (quadrature.size(), numbers::signaling_nan<Tensor<1, dim> > ()),
          cell_i_d_values (quadrature.size(), numbers::signaling_nan<double> ()),
          face_current_velocity_values (face_quadrature.size(), numbers::signaling_nan<Tensor<1, dim> >()),
          face_old_velocity_values (face_quadrature.size(), numbers::signaling_nan<Tensor<1, dim> >()),
          face_old_old_velocity_values (face_quadrature.size(), numbers::signaling_nan<Tensor<1, dim> >()),
          face_mesh_velocity_values (face_quadrature.size(), numbers::signaling_nan<Tensor<1, dim> >()),
          neighbor_old_values (face_quadrature.size(), numbers::signaling_nan<double>()),
          neighbor_i_n_values (face_quadrature.size(), numbers::signaling_nan<Tensor<1, dim> >()),
          neighbor_i_d_values (face_quadrature.size(), numbers::signaling_nan<double>())
        {}

        template <int dim>
        VoFSystem<dim>::VoFSystem (const VoFSystem &scratch)
          :
          finite_element_values (scratch.finite_element_values.get_mapping(),
                                 scratch.finite_element_values.get_fe(),
                                 scratch.finite_element_values.get_quadrature(),
                                 scratch.finite_element_values.get_update_flags()),
          neighbor_finite_element_values (scratch.neighbor_finite_element_values.get_mapping(),
                                          scratch.neighbor_finite_element_values.get_fe(),
                                          scratch.neighbor_finite_element_values.get_quadrature(),
                                          scratch.neighbor_finite_element_values.get_update_flags()),
          face_finite_element_values (scratch.face_finite_element_values.get_mapping(),
                                      scratch.face_finite_element_values.get_fe(),
                                      scratch.face_finite_element_values.get_quadrature(),
                                      scratch.face_finite_element_values.get_update_flags()),
          neighbor_face_finite_element_values (scratch.neighbor_face_finite_element_values.get_mapping(),
                                               scratch.neighbor_face_finite_element_values.get_fe(),
                                               scratch.neighbor_face_finite_element_values.get_quadrature(),
                                               scratch.neighbor_face_finite_element_values.get_update_flags()),
          subface_finite_element_values (scratch.subface_finite_element_values.get_mapping(),
                                         scratch.subface_finite_element_values.get_fe(),
                                         scratch.subface_finite_element_values.get_quadrature(),
                                         scratch.subface_finite_element_values.get_update_flags()),
          local_dof_indices (scratch.finite_element_values.get_fe().dofs_per_cell),
          phi_field (scratch.phi_field),
          old_field_values (scratch.old_field_values),
          cell_i_n_values (scratch.cell_i_n_values),
          cell_i_d_values (scratch.cell_i_d_values),
          face_current_velocity_values (scratch.face_current_velocity_values),
          face_old_velocity_values (scratch.face_old_velocity_values),
          face_old_old_velocity_values (scratch.face_old_old_velocity_values),
          face_mesh_velocity_values (scratch.face_mesh_velocity_values),
          neighbor_old_values (scratch.neighbor_old_values),
          neighbor_i_n_values (scratch.neighbor_i_n_values),
          neighbor_i_d_values (scratch.neighbor_i_d_values)
        {}
      }

      namespace CopyData
      {
        template <int dim>
        VoFSystem<dim>::VoFSystem(const FiniteElement<dim> &finite_element)
          :
          local_matrix (finite_element.dofs_per_cell,
                        finite_element.dofs_per_cell),
          local_rhs (finite_element.dofs_per_cell),
          local_dof_indices (finite_element.dofs_per_cell)
        {
          TableIndices<2> mat_size(finite_element.dofs_per_cell,
                                   finite_element.dofs_per_cell);
          for (unsigned int i=0;
               i < GeometryInfo<dim>::max_children_per_face *GeometryInfo<dim>::faces_per_cell;
               ++i)
            {
              face_contributions_mask[i] = false;
              local_face_rhs[i].reinit (finite_element.dofs_per_cell);
              local_face_matrices_ext_ext[i].reinit(mat_size);
              neighbor_dof_indices[i].resize(finite_element.dofs_per_cell);
            }
        }

        template<int dim>
        VoFSystem<dim>::VoFSystem(const VoFSystem &data)
          :
          local_matrix (data.local_matrix),
          local_rhs (data.local_rhs),
          local_face_rhs (data.local_face_rhs),
          local_face_matrices_ext_ext (data.local_face_matrices_ext_ext),
          local_dof_indices (data.local_dof_indices),
          neighbor_dof_indices (data.neighbor_dof_indices)
        {
          unsigned int dofs_per_cell = local_rhs.size();
          TableIndices<2> mat_size(dofs_per_cell,
                                   dofs_per_cell);
          for (unsigned int i=0;
               i < GeometryInfo<dim>::max_children_per_face *GeometryInfo<dim>::faces_per_cell;
               ++i)
            {
              face_contributions_mask[i] = false;
              local_face_rhs[i].reinit (dofs_per_cell);
              local_face_matrices_ext_ext[i].reinit(mat_size);
              neighbor_dof_indices[i].resize(dofs_per_cell);
            }
        }
      }
    }
  }

  template <int dim>
  void VoFHandler<dim>::local_assemble_vof_system (const VoFField<dim> field,
                                                   const unsigned int calc_dir,
                                                   bool update_from_old,
                                                   const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                   internal::Assembly::Scratch::VoFSystem<dim> &scratch,
                                                   internal::Assembly::CopyData::VoFSystem<dim> &data)
  {
    const bool old_velocity_avail = (this->get_timestep_number() > 0);
    const bool old_old_velocity_avail = (this->get_timestep_number() > 1);

    const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;
    const unsigned int n_f_q_points    = scratch.face_finite_element_values.n_quadrature_points;

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

    const unsigned int solution_component = field.fraction.first_component_index;
    const FEValuesExtractors::Scalar solution_field = field.fraction.extractor_scalar();
    const Quadrature<dim> &quadrature = scratch.finite_element_values.get_quadrature();

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

    double face_flux;
    double dflux = 0.0;

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
  void VoFHandler<dim>::local_assemble_internal_face_vof_system (const VoFField<dim> field,
                                                                 const unsigned int calc_dir,
                                                                 bool update_from_old,
                                                                 const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                                 const unsigned int face_no,
                                                                 internal::Assembly::Scratch::VoFSystem<dim> &scratch,
                                                                 internal::Assembly::CopyData::VoFSystem<dim> &data)
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

    const unsigned int solution_component = field.fraction.first_component_index;
    const FEValuesExtractors::Scalar solution_field = field.fraction.extractor_scalar();

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
            data.local_rhs [0] -= flux_vof * face_flux;
            data.local_matrix (0, 0) -= face_flux;
            data.local_face_rhs[f_rhs_ind][0] += flux_vof * face_flux;
            data.local_face_matrices_ext_ext[f_rhs_ind] (0, 0) += face_flux;
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
            const Tensor<1, dim, double> neighbor_i_normal = scratch.neighbor_i_n_values[0];
            const double neighbor_i_d = scratch.neighbor_i_d_values[0];

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

            data.local_rhs [0] -= flux_vof * face_flux;
            data.local_matrix (0, 0) -= face_flux;
            data.local_face_rhs[f_rhs_ind][0] += flux_vof * face_flux;
            data.local_face_matrices_ext_ext[f_rhs_ind] (0, 0) += face_flux;

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


  template <int dim>
  void VoFHandler<dim>::local_assemble_boundary_face_vof_system (const VoFField<dim> field,
                                                                 const unsigned int calc_dir,
                                                                 bool update_from_old,
                                                                 const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                                 const unsigned int face_no,
                                                                 internal::Assembly::Scratch::VoFSystem<dim> &scratch,
                                                                 internal::Assembly::CopyData::VoFSystem<dim> &data)
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

    const unsigned int solution_component = field.fraction.first_component_index;
    const FEValuesExtractors::Scalar solution_field = field.fraction.extractor_scalar();

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
        if (face_flux < 0.0)
          {
            flux_vof = 0.0;
          }
        else if (face_flux < vof_epsilon*cell_vol)
          {
            face_flux=0.0;
            flux_vof = 0.0;
          }
        else
          {
            flux_vof = VolumeOfFluid::calc_vof_flux_edge<dim> (f_dim,
                                                               face_ls_time_grad,
                                                               cell_i_normal,
                                                               face_ls_d);
          }

        //TODO: Handle non-zero inflow VoF boundary conditions

        // Add fluxes to RHS
        data.local_rhs[0] -= flux_vof * face_flux;
        data.local_matrix(0, 0) -= face_flux;
      }
  }


  template <int dim>
  void VoFHandler<dim>::assemble_vof_system (const VoFField<dim> field,
                                             unsigned int dir,
                                             bool update_from_old)
  {
    sim.computing_timer.enter_section ("   Assemble VoF system");
    const unsigned int block_idx = field.fraction.block_index;
    sim.system_matrix.block(block_idx, block_idx) = 0;
    sim.system_rhs = 0;

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    const FiniteElement<dim> &vof_fe = (*field.fraction.fe);

    WorkStream::
    run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                     sim.dof_handler.begin_active()),
         CellFilter (IteratorFilters::LocallyOwnedCell(),
                     sim.dof_handler.end()),
         std_cxx11::bind (&VoFHandler<dim>::
                          local_assemble_vof_system,
                          this,
                          field,
                          dir,
                          update_from_old,
                          std_cxx11::_1,
                          std_cxx11::_2,
                          std_cxx11::_3),
         std_cxx11::bind (&VoFHandler<dim>::
                          copy_local_to_global_vof_system,
                          this,
                          std_cxx11::_1),
         // we have to assemble the term u.grad phi_i * phi_j, which is
         // of total polynomial degree
         //   stokes_deg - 1
         // (or similar for comp_deg). this suggests using a Gauss
         // quadrature formula of order
         //   stokes_deg/2
         // rounded up. do so. (note that x/2 rounded up
         // equals (x+1)/2 using integer division.)
         //
         // (note: we need to get at the advection element in
         // use for the scratch and copy objects below. the
         // base element for the compositional fields exists
         // only once, with multiplicity, so only query
         // introspection.block_indices.compositional_fields[0]
         // instead of subscripting with the correct compositional
         // field index.)
         internal::Assembly::Scratch::
         VoFSystem<dim> (sim.finite_element,
                         vof_fe,
                         *sim.mapping,
                         QGauss<dim>((sim.parameters.stokes_velocity_degree+1)/2),
                         QGauss<dim-1>((sim.parameters.stokes_velocity_degree+1)/2)),
         internal::Assembly::CopyData::
         VoFSystem<dim> (vof_fe));

    sim.system_matrix.compress(VectorOperation::add);
    sim.system_rhs.compress(VectorOperation::add);

    sim.computing_timer.exit_section ();
  }

  template <int dim>
  void VoFHandler<dim>::copy_local_to_global_vof_system (const internal::Assembly::CopyData::VoFSystem<dim> &data)
  {
    // copy entries into the global matrix. note that these local contributions
    // only correspond to the advection dofs, as assembled above
    sim.current_constraints.distribute_local_to_global (data.local_matrix,
                                                        data.local_rhs,
                                                        data.local_dof_indices,
                                                        sim.system_matrix,
                                                        sim.system_rhs);

    /* In the following, we copy DG contributions element by element. This
     * is allowed since there are no constraints imposed on discontinuous fields.
     */
    for (unsigned int f=0; f<GeometryInfo<dim>::max_children_per_face
         * GeometryInfo<dim>::faces_per_cell; ++f)
      {
        if (data.face_contributions_mask[f])
          {
            for (unsigned int i=0; i<data.neighbor_dof_indices[f].size(); ++i)
              {
                sim.system_rhs(data.neighbor_dof_indices[f][i]) += data.local_face_rhs[f][i];
                for (unsigned int j=0; j< data.neighbor_dof_indices[f].size(); ++j)
                  {
                    sim.system_matrix.add(data.neighbor_dof_indices[f][i],
                                          data.neighbor_dof_indices[f][j],
                                          data.local_face_matrices_ext_ext[f](i,j));
                  }
              }
          }
      }
  }
}

namespace aspect
{
#define INSTANTIATE(dim) \
  template void VoFHandler<dim>::assemble_vof_system (const VoFField<dim> field, \
                                                      unsigned int dir, \
                                                      bool update_from_old); \
  template void VoFHandler<dim>::local_assemble_vof_system (const VoFField<dim> field, \
                                                            const unsigned int calc_dir, \
                                                            bool update_from_old, \
                                                            const typename DoFHandler<dim>::active_cell_iterator &cell, \
                                                            internal::Assembly::Scratch::VoFSystem<dim> &scratch, \
                                                            internal::Assembly::CopyData::VoFSystem<dim> &data); \
  template void VoFHandler<dim>::copy_local_to_global_vof_system (const internal::Assembly::CopyData::VoFSystem<dim> &data);


  ASPECT_INSTANTIATE(INSTANTIATE)
}
