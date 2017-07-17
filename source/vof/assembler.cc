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
    void VoFAssembler<dim>::local_assemble_vof_system (const VoFField<dim> field,
                                                       const unsigned int calc_dir,
                                                       bool update_from_old,
                                                       const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                       internal::Assembly::Scratch::VoFSystem<dim> &scratch,
                                                       internal::Assembly::CopyData::VoFSystem<dim> &data)
    {
    }

    template <int dim>
    void VoFAssembler<dim>::local_assemble_boundary_face_vof_system (const VoFField<dim> field,
                                                                     const unsigned int calc_dir,
                                                                     bool update_from_old,
                                                                     const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                                     const unsigned int face_no,
                                                                     internal::Assembly::Scratch::VoFSystem<dim> &scratch,
                                                                     internal::Assembly::CopyData::VoFSystem<dim> &data)
    {
    }

    template <int dim>
    void VoFAssembler<dim>::local_assemble_internal_face_vof_system (const VoFField<dim> field,
                                                                     const unsigned int calc_dir,
                                                                     bool update_from_old,
                                                                     const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                                     const unsigned int face_no,
                                                                     internal::Assembly::Scratch::VoFSystem<dim> &scratch,
                                                                     internal::Assembly::CopyData::VoFSystem<dim> &data)
    {
    }
  }
}

namespace aspect
{
  namespace Assemblers
  {
#define INSTANTIATE(dim) \
  template void VoFAssembler<dim>::local_assemble_internal_face_vof_system (const VoFField<dim> field, \
                                                                            const unsigned int calc_dir, \
                                                                            bool update_from_old, \
                                                                            const typename DoFHandler<dim>::active_cell_iterator &cell, \
                                                                            const unsigned int face_no, \
                                                                            internal::Assembly::Scratch::VoFSystem<dim> &scratch, \
                                                                            internal::Assembly::CopyData::VoFSystem<dim> &data);

    ASPECT_INSTANTIATE(INSTANTIATE)

  }
}
