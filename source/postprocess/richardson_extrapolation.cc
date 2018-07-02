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


#include <aspect/postprocess/richardson_extrapolation.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>

#include <aspect/global.h>
#include <math.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>

#include <aspect/simulator.h>
#include <aspect/utilities.h>

namespace aspect
{
  namespace Postprocess
  {

    template <int dim>
    void
    RichardsonExtrapolation<dim>::initialize()
    {
      AssertThrow(dim==2,
                  ExcMessage("Richardson Extrapolation is currently only functional for dim=2."));
      AssertThrow((this->get_parameters().initial_adaptive_refinement == 0 ||
                   this->get_parameters().adaptive_refinement_interval == 0),
                  ExcMessage("Richardson Extrapolation is not compatible with AMR."));

      coarse_file_name = this->get_output_directory()
                         + coarse_file_name + "_"
                         + Utilities::int_to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()))
                         + ".dat";
      refined_file_name = this->get_output_directory()
                          + refined_file_name + "_"
                          + Utilities::int_to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()))
                          + ".dat";
    }

    template <int dim>
    void
    RichardsonExtrapolation<dim>::write_out_coarse_data()
    {
      std::ofstream interpolated_data_stream;
      interpolated_data_stream.open(coarse_file_name, std::ios_base::out);
      interpolated_data_stream << std::setprecision(14);

      /**
      * Compute the Legendre gauss points at level 2 indirection.
      **/
      //QGaussLobatto<1> base_quadrature(this->get_stokes_velocity_degree() + 1);
      QGauss<1> base_quadrature(this->get_stokes_velocity_degree() + 1);
      QIterated<dim> quadrature_rule (base_quadrature, 1);

      FEValues<dim> fe_values(this->get_mapping(),
                              this->get_fe(),
                              quadrature_rule,
                              update_values |
                              update_quadrature_points |
                              update_JxW_values);

      const FEValuesExtractors::Scalar &extractor_pressure = this->introspection().extractors.pressure;
      const FEValuesExtractors::Scalar &extractor_temperature = this->introspection().extractors.temperature;
      const FEValuesExtractors::Vector &extractor_velocity = this->introspection().extractors.velocities;

      // Declaring an iterator over all active cells on local mpi process
      typename DoFHandler<dim>::active_cell_iterator cell = this->get_dof_handler().begin_active();
      for (; cell != this->get_dof_handler().end();
           ++cell)
        {
          if (!(cell->is_locally_owned()))
            continue;

          fe_values.reinit(cell);
          // Each vector is of the same length.
          const std::vector<Point<dim>>  quadrature_points = fe_values.get_quadrature_points();

          std::vector<double> interpolated_temperature(quadrature_points.size());
          std::vector<double> interpolated_pressure(quadrature_points.size());
          std::vector<Tensor<1,dim>> interpolated_velocity(quadrature_points.size());
          std::vector<std::vector<double>> interpolated_compositional_fields(this->n_compositional_fields(),
                                                                             std::vector<double>(quadrature_points.size()));

          fe_values[extractor_pressure].get_function_values(this->get_solution(), interpolated_pressure);
          fe_values[extractor_temperature].get_function_values(this->get_solution(), interpolated_temperature);
          fe_values[extractor_velocity].get_function_values(this->get_solution(), interpolated_velocity);

          typename std::vector<std::vector<double>>::iterator itr_compositional_fields = interpolated_compositional_fields.begin();

          unsigned int index = 0;
          for (; itr_compositional_fields != interpolated_compositional_fields.end(); itr_compositional_fields++)
            {
              fe_values[this->introspection().extractors.compositional_fields[index]].get_function_values(
                this->get_solution(),
                *itr_compositional_fields);
              index++;
            }

          typename std::vector<double>::const_iterator itr_temperature = interpolated_temperature.begin();
          typename std::vector<double>::const_iterator itr_pressure = interpolated_pressure.begin();
          typename std::vector<Tensor<1,dim>>::const_iterator itr_velocity = interpolated_velocity.begin();

          typename std::vector<Point<dim>>::const_iterator itr_quadrature_points = quadrature_points.begin();

          unsigned int quadrature_point_index = 0;
          for (; itr_quadrature_points != quadrature_points.end();
               itr_quadrature_points++, itr_velocity++, itr_pressure++, itr_temperature++, quadrature_point_index++)
            {
              interpolated_data_stream << std::scientific << std::setprecision(15)
                                       << (*itr_quadrature_points)[0] << "\t" << (*itr_quadrature_points)[1]
                                       << "\t" << (*itr_velocity)[0] << "\t" << (*itr_velocity)[1]
                                       << "\t" << *itr_pressure << "\t" << *itr_temperature;
              if (this->n_compositional_fields() != 0)
                {
                  unsigned int count = 0;
                  itr_compositional_fields = interpolated_compositional_fields.begin();
                  for (; itr_compositional_fields != interpolated_compositional_fields.end(); itr_compositional_fields++, count++)
                    interpolated_data_stream << "\t" << (*itr_compositional_fields)[quadrature_point_index] << "\t";
                }
              interpolated_data_stream << std::endl;
            }
        }

      interpolated_data_stream.close();
    }

    template <int dim>
    void
    RichardsonExtrapolation<dim>::write_out_refined_data()
    {
      std::ofstream interpolated_data_stream;
      interpolated_data_stream.open(refined_file_name, std::ios_base::out);
      interpolated_data_stream << std::setprecision(14);

      /**
      * Compute the Legendre gauss points at level 2 indirection.
      **/
      //QGaussLobatto<1> base_quadrature(this->get_stokes_velocity_degree() + 1);
      QGauss<1> base_quadrature(this->get_stokes_velocity_degree() + 1);
      QIterated<dim> quadrature_rule (base_quadrature, 2);

      FEValues<dim> fe_values(this->get_mapping(),
                              this->get_fe(),
                              quadrature_rule,
                              update_values |
                              update_quadrature_points |
                              update_JxW_values);

      const FEValuesExtractors::Scalar &extractor_pressure = this->introspection().extractors.pressure;
      const FEValuesExtractors::Scalar &extractor_temperature = this->introspection().extractors.temperature;
      const FEValuesExtractors::Vector &extractor_velocity = this->introspection().extractors.velocities;

      // Declaring an iterator over all active cells on local mpi process
      typename DoFHandler<dim>::active_cell_iterator cell = this->get_dof_handler().begin_active();
      for (; cell != this->get_dof_handler().end();
           ++cell)
        {
          if (!(cell->is_locally_owned()))
            continue;

          fe_values.reinit(cell);
          // Each vector is of the same length.
          const std::vector<Point<dim>>  quadrature_points = fe_values.get_quadrature_points();

          std::vector<double> interpolated_temperature(quadrature_points.size());
          std::vector<double> interpolated_pressure(quadrature_points.size());
          std::vector<Tensor<1,dim>> interpolated_velocity(quadrature_points.size());
          std::vector<std::vector<double>> interpolated_compositional_fields(this->n_compositional_fields(),
                                                                             std::vector<double>(quadrature_points.size()));

          fe_values[extractor_pressure].get_function_values(this->get_solution(), interpolated_pressure);
          fe_values[extractor_temperature].get_function_values(this->get_solution(), interpolated_temperature);
          fe_values[extractor_velocity].get_function_values(this->get_solution(), interpolated_velocity);

          typename std::vector<std::vector<double>>::iterator itr_compositional_fields = interpolated_compositional_fields.begin();

          unsigned int index = 0;
          for (; itr_compositional_fields != interpolated_compositional_fields.end(); itr_compositional_fields++)
            {
              fe_values[this->introspection().extractors.compositional_fields[index]].get_function_values(
                this->get_solution(),
                *itr_compositional_fields);
              index++;
            }

          typename std::vector<double>::const_iterator itr_temperature = interpolated_temperature.begin();
          typename std::vector<double>::const_iterator itr_pressure = interpolated_pressure.begin();
          typename std::vector<Tensor<1,dim>>::const_iterator itr_velocity = interpolated_velocity.begin();

          typename std::vector<Point<dim>>::const_iterator itr_quadrature_points = quadrature_points.begin();

          unsigned int quadrature_point_index = 0;
          for (; itr_quadrature_points != quadrature_points.end();
               itr_quadrature_points++, itr_velocity++, itr_pressure++, itr_temperature++, quadrature_point_index++)
            {
              interpolated_data_stream << std::scientific << std::setprecision(15)
                                       << (*itr_quadrature_points)[0] << "\t" << (*itr_quadrature_points)[1]
                                       << "\t" << (*itr_velocity)[0] << "\t" << (*itr_velocity)[1]
                                       << "\t" << *itr_pressure << "\t" << *itr_temperature;
              if (this->n_compositional_fields() != 0)
                {
                  unsigned int count = 0;
                  itr_compositional_fields = interpolated_compositional_fields.begin();
                  for (; itr_compositional_fields != interpolated_compositional_fields.end(); itr_compositional_fields++, count++)
                    interpolated_data_stream << "\t" << (*itr_compositional_fields)[quadrature_point_index] << "\t";
                }
              interpolated_data_stream << std::endl;
            }
        }

      interpolated_data_stream.close();
    }

    template <int dim>
    std::pair<std::string,std::string>
    RichardsonExtrapolation<dim>::execute (TableHandler &)
    {
      if ( this->get_time() == end_time)
        {
          // Write out the current solution, both current and interpolated at a higher resolved mesh.
          write_out_coarse_data();
          write_out_refined_data();

          return std::pair<std::string, std::string> ("State data written to : ",
                                                      coarse_file_name + ", " + refined_file_name);
        }
      else
        {
          return std::pair<std::string, std::string> ();
        }
    }

    template <int dim>
    void
    RichardsonExtrapolation<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Richardson extrapolation");
        {
          prm.declare_entry("File name prefix for interpolated data", "interpolated_data",
                            Patterns::Anything (),
                            "The file name containing the interpolated solution at the nodal values "
                            "at the current grid resolution+1.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    RichardsonExtrapolation<dim>::parse_parameters (ParameterHandler &prm)
    {
      bool use_years = prm.get_bool ("Use years in output instead of seconds");

      // Use the existing end time parameter
      //
      // read end time from parameter file. if it is to be interpreted
      // in years rather than seconds, then do the conversion
      end_time = prm.get_double ("End time");
      if ( use_years == true)
        end_time *= year_in_seconds;

      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Richardson extrapolation");
        {
          std::string file_prefix = prm.get("File name prefix for interpolated data");
          coarse_file_name = file_prefix + "_" + Utilities::int_to_string(this->get_parameters().initial_global_refinement) + "_c";
          refined_file_name = file_prefix + "_" + Utilities::int_to_string(this->get_parameters().initial_global_refinement) + "_r";
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
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(RichardsonExtrapolation,
                                  "richardson extrapolation",
                                  "A postprocessor called Richardson Extrapolation.")
  }
}
