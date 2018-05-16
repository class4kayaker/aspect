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

#include <aspect/simulator.h>
#include <aspect/utilities.h>
#include <aspect/free_surface.h>
#include <aspect/volume_of_fluid/handler.h>
#include <aspect/volume_of_fluid/utilities.h>
#include <aspect/volume_of_fluid/assembly.h>

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
                                   const FiniteElement<dim> &volume_of_fluid_element,
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
          phi_field (volume_of_fluid_element.dofs_per_cell, numbers::signaling_nan<double>()),
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
  void VoFHandler<dim>::assemble_volume_of_fluid_system (const VoFField<dim> field,
                                             const unsigned int dir,
                                             const bool update_from_old)
  {
    sim.computing_timer.enter_section ("   Assemble VoF system");
    const unsigned int block_idx = field.volume_fraction.block_index;
    sim.system_matrix.block(block_idx, block_idx) = 0;
    sim.system_rhs = 0;

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    const FiniteElement<dim> &volume_of_fluid_fe = (*field.volume_fraction.fe);

    WorkStream::
    run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                     this->get_dof_handler().begin_active()),
         CellFilter (IteratorFilters::LocallyOwnedCell(),
                     this->get_dof_handler().end()),
         std_cxx11::bind (&Assemblers::VoFAssembler<dim>::
                          local_assemble_volume_of_fluid_system,
                          assembler,
                          field,
                          dir,
                          update_from_old,
                          std_cxx11::_1,
                          std_cxx11::_2,
                          std_cxx11::_3),
         std_cxx11::bind (&VoFHandler<dim>::
                          copy_local_to_global_volume_of_fluid_system,
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
         VoFSystem<dim> (this->get_fe(),
                         volume_of_fluid_fe,
                         this->get_mapping(),
                         QGauss<dim>((this->get_parameters().stokes_velocity_degree+1)/2),
                         QGauss<dim-1>((this->get_parameters().stokes_velocity_degree+1)/2)),
         internal::Assembly::CopyData::
         VoFSystem<dim> (volume_of_fluid_fe));

    sim.system_matrix.compress(VectorOperation::add);
    sim.system_rhs.compress(VectorOperation::add);

    sim.computing_timer.exit_section ();
  }

  template <int dim>
  void VoFHandler<dim>::copy_local_to_global_volume_of_fluid_system (const internal::Assembly::CopyData::VoFSystem<dim> &data)
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
  template void VoFHandler<dim>::assemble_volume_of_fluid_system (const VoFField<dim> field, \
                                                      unsigned int dir, \
                                                      bool update_from_old); \
  template void VoFHandler<dim>::copy_local_to_global_volume_of_fluid_system (const internal::Assembly::CopyData::VoFSystem<dim> &data);


  ASPECT_INSTANTIATE(INSTANTIATE)
}
