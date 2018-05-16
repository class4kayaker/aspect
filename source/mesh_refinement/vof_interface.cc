/*
  Copyright (C) 2011 - 2018 by the authors of the ASPECT code.

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


#include <aspect/mesh_refinement/vof_interface.h>
#include <aspect/vof/handler.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/geometry_info.h>

#include <deal.II/numerics/error_estimator.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_tools.h>

namespace aspect
{
  namespace MeshRefinement
  {

    template <int dim>
    void
    VoFInterface<dim>::tag_additional_cells() const
    {

      //Break early if DoFs do not exist
      if (this->get_dof_handler().n_dofs() == 0) return;


      const QMidpoint<dim> qMidC;

      // Create a map from vertices to adjacent cells
      const std::vector<std::set<typename Triangulation<dim>::active_cell_iterator> >
      vertex_to_cells(GridTools::vertex_to_cell_map(this->get_triangulation()));

//        std::vector<std::set<typename parallel::distributed::Triangulation<dim>::active_cell_iterator> > vert_cell_map =
//                vertex_to_cells(GridTools::vertex_to_cell_map(this->get_dof_handler().get_triangulation()));

      std::set<typename Triangulation<dim>::active_cell_iterator> marked_cells;
      FEValues<dim> fe_values (this->get_mapping(),
                               this->get_fe(),
                               qMidC,
                               update_values |
                               update_quadrature_points);
      for (unsigned int f=0; f<this->get_vof_handler().get_n_fields(); ++f)
        {

          const FEValuesExtractors::Scalar vof_field = this->get_vof_handler().field_struct_for_field_index(f)
                                                       .volume_fraction.extractor_scalar();
          std::vector<double> vof_q_values(qMidC.size());

          const double voleps = this->get_vof_handler().get_volume_fraction_threshold();

          typename DoFHandler<dim>::active_cell_iterator cell = this->get_dof_handler().begin_active(),
                                                         endc = this->get_dof_handler().end();
          for (; cell != endc; ++cell)
            {
              // Skip if not local
              if (!cell->is_locally_owned())
                continue;

              bool mark = false;

              // Get cell vof
              fe_values.reinit(cell);
              fe_values[vof_field].get_function_values(this->get_solution(),
                                                       vof_q_values);
              double cell_vof = vof_q_values[0];

              // Handle overshoots
              if (cell_vof > 1.0)
                cell_vof = 1.0;

              if (cell_vof < 0.0)
                cell_vof = 0.0;

              // Check if at interface
              if (cell_vof > voleps && cell_vof < (1.0 - voleps))
                {
                  mark = true;
                }

              // ha
              if (!mark)
                {
                  for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                    {
                      const bool cell_has_periodic_neighbor = cell->has_periodic_neighbor(f);
                      const typename DoFHandler<dim>::face_iterator face = cell->face(f);

                      if (face->at_boundary() && !cell_has_periodic_neighbor)
                        continue;

                      const typename DoFHandler<dim>::cell_iterator neighbor = cell->neighbor_or_periodic_neighbor(f);

                      if ((!face->at_boundary() && !face->has_children())
                          ||
                          (face->at_boundary() && cell->periodic_neighbor_is_coarser(f))
                          ||
                          (face->at_boundary() && neighbor->level() == cell->level() && neighbor->active()))
                        {
                          if (neighbor->active())
                            {
                              if (neighbor==endc) continue;
                              fe_values.reinit(neighbor);
                              fe_values[vof_field].get_function_values(this->get_solution(),
                                                                       vof_q_values);

                              double neighbor_vof = vof_q_values[0];

                              if (neighbor_vof > 1.0)
                                neighbor_vof = 1.0;

                              if (neighbor_vof < 0.0)
                                neighbor_vof = 0.0;

                              if (std::abs(neighbor_vof-cell_vof)>voleps)
                                {
                                  mark = true;
                                  break;
                                }
                            }
                          else
                            {
                              this->get_pcout() << "Error " << cell->index();
                            }
                        }
                      else
                        {
                          for (unsigned int sf=0; sf < face->number_of_children(); ++sf)
                            {
                              const typename DoFHandler<dim>::active_cell_iterator neighbor_sub =
                                (cell_has_periodic_neighbor
                                 ?
                                 cell->periodic_neighbor_child_on_subface(f, sf)
                                 :
                                 cell->neighbor_child_on_subface(f, sf));

                              if (neighbor_sub==endc) continue;
                              fe_values.reinit(neighbor_sub);
                              fe_values[vof_field].get_function_values(this->get_solution(),
                                                                       vof_q_values);

                              double neighbor_vof = vof_q_values[0];

                              if (neighbor_vof > 1.0)
                                neighbor_vof = 1.0;

                              if (neighbor_vof < 0.0)
                                neighbor_vof = 0.0;

                              if (std::abs(neighbor_vof-cell_vof)>voleps)
                                {
                                  mark = true;
                                  break;
                                }

                            }
                          if (mark)
                            break;
                        }
                    }
                }

              if (mark)
                {
                  // Fractional volume
                  marked_cells.insert(cell);
                  cell->clear_coarsen_flag ();
                  cell->set_refine_flag ();
                }

            }
        }

      // Now mark for refinement all cells that are a neighbor of a cell that contains the interface

      std::set<typename Triangulation<dim>::active_cell_iterator> marked_cells_and_neighbors = marked_cells;
      typename std::set<typename parallel::distributed::Triangulation<dim>::active_cell_iterator>::const_iterator mcells = marked_cells.begin(),
                                                                                                                  endmc = marked_cells.end();
      for (; mcells != endmc; mcells++)
        {
          typename parallel::distributed::Triangulation<dim>::active_cell_iterator mcell = *mcells;
          for (unsigned int vertex_index=0; vertex_index<GeometryInfo<dim>::vertices_per_cell; ++vertex_index)
            {
              std::set<typename Triangulation<dim>::active_cell_iterator> neighbor_cells = vertex_to_cells[mcell->vertex_index(vertex_index)];
              typename std::set<typename Triangulation<dim>::active_cell_iterator>::const_iterator neighbor_cell = neighbor_cells.begin(),
                                                                                                   end_neighbor_cell_index= neighbor_cells.end();
              for (; neighbor_cell!=end_neighbor_cell_index; neighbor_cell++)
                {
                  typename Triangulation<dim>::active_cell_iterator itr_tmp = *neighbor_cell;
                  if (itr_tmp->active())
                    {
                      itr_tmp->clear_coarsen_flag ();
                      itr_tmp->set_refine_flag ();
                      marked_cells_and_neighbors.insert(itr_tmp);
                    }
                }
            }

          // Check for periodic neighbors, and refine if existing
          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
            {
              if ( mcell->has_periodic_neighbor(f))
                {
                  typename Triangulation<dim>::cell_iterator itr_tmp = mcell->periodic_neighbor(f);
                  if (itr_tmp->active())
                    {
                      itr_tmp->clear_coarsen_flag ();
                      itr_tmp->set_refine_flag ();
                      marked_cells_and_neighbors.insert(itr_tmp);
                    }
                }
            }
        }

      if (strict_refinement)
        {
          typename DoFHandler<dim>::active_cell_iterator
          cell = this->get_dof_handler().begin_active(),
          endc = this->get_dof_handler().end();
          for (; cell != endc; ++cell)
            {
              if (cell->is_locally_owned())
                {
                  if (marked_cells_and_neighbors.find(cell) != marked_cells_and_neighbors.end())
                    {
                      //Refinement already requested
                    }
                  else
                    {
                      if (cell->active())
                        {
                          cell->set_coarsen_flag();
                          cell->clear_refine_flag();
                        }
                    }
                }
            }
        }

    }

    template <int dim>
    void
    VoFInterface<dim>::
    declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Mesh refinement");
      {
        prm.enter_subsection("VOF Interface");
        {
          prm.declare_entry("Strict refinement", "false",
                            Patterns::Bool(),
                            "If true, then explicitly coarsen any cells not "
                            "neighboring the VoF interface.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    VoFInterface<dim>::parse_parameters (ParameterHandler &prm)
    {
      //TODO: Add check for vof active
      AssertThrow(this->get_parameters().vof_tracking_enabled,
                  ExcMessage("The 'vof boundary' mesh refinement strategy requires that the 'Use VoF tracking' parameter is enabled."));

      prm.enter_subsection("Mesh refinement");
      {
        prm.enter_subsection("VOF Interface");
        {
          strict_refinement = prm.get_bool("Strict refinement");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


  }
}

// explicit instantiations
namespace aspect
{
  namespace MeshRefinement
  {
    ASPECT_REGISTER_MESH_REFINEMENT_CRITERION(VoFInterface,
                                              "vof interface",
                                              "A class that implements a mesh refinement criterion, which "
                                              "ensures a minimum level of refinement near the VoF interface boundary.")
  }
}
