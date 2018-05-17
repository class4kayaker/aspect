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

#include <aspect/postprocess/visualization/volume_of_fluid_values.h>
#include <aspect/simulator_access.h>
#include <aspect/volume_of_fluid/handler.h>


namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      template <int dim>
      VolumeOfFluidValues<dim>::
      VolumeOfFluidValues ()
        :
        DataPostprocessor<dim> ()
      {}


      template <int dim>
      std::vector<std::string>
      VolumeOfFluidValues<dim>::
      get_names () const
      {
        return volume_of_fluid_names;
      }


      template <int dim>
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
      VolumeOfFluidValues<dim>::
      get_data_component_interpretation () const
      {
        return interp;
      }


      template <int dim>
      UpdateFlags
      VolumeOfFluidValues<dim>::
      get_needed_update_flags () const
      {
        return update_values;
      }


      template <int dim>
      void
      VolumeOfFluidValues<dim>::
      evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                            std::vector<Vector<double>> &computed_quantities) const
      {
        const unsigned int n_quadrature_points = input_data.solution_values.size();
        Assert (computed_quantities.size() == n_quadrature_points, ExcInternalError ());
        Assert (input_data.solution_values[0].size() == this->introspection().n_components, ExcInternalError ());

        unsigned int out_per_field=1;
        if (include_contour)
          out_per_field += 1;
        if (include_normal)
          out_per_field += dim;

        for (unsigned int f=0; f<this->get_volume_of_fluid_handler().get_n_fields(); ++f)
          {
            VolumeOfFluidField<dim> field = this->get_volume_of_fluid_handler().field_struct_for_field_index(f);

            const FEVariable<dim> &volume_of_fluid_var = field.volume_fraction;
            const unsigned int volume_of_fluid_ind = volume_of_fluid_var.first_component_index;
            const FEVariable<dim> &volume_of_fluidLS_var = field.level_set;
            const unsigned int volume_of_fluidLS_ind = volume_of_fluidLS_var.first_component_index;

            for (unsigned int q=0; q<n_quadrature_points; ++q)
              {
                unsigned int out_ind = f*out_per_field;
                computed_quantities[q][out_ind] = input_data.solution_values[q][volume_of_fluid_ind];
                ++out_ind;
                if (include_contour)
                  {
                    computed_quantities[q][out_ind] = input_data.solution_values[q][volume_of_fluidLS_ind];
                    ++out_ind;
                  }

                if (include_normal)
                  {
                    Tensor<1, dim, double> normal = -input_data.solution_gradients[q][volume_of_fluidLS_ind];
                    for (unsigned int i = 0; i<dim; ++i)
                      {
                        computed_quantities[q][out_ind] = normal[i];
                        ++out_ind;
                      }
                  }
              }
          }
      }


      template <int dim>
      void
      VolumeOfFluidValues<dim>::declare_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Visualization");
          {
            prm.enter_subsection("Volume of Fluid");
            {
              prm.declare_entry("Include internal reconstruction contour", "false",
                                Patterns::Bool (),
                                "Include the internal level set data use to save reconstructed interfaces");

              prm.declare_entry("Include normals", "false",
                                Patterns::Bool (),
                                "Include internal normal data in output (DEBUG)");
            }
            prm.leave_subsection();
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }


      template <int dim>
      void
      VolumeOfFluidValues<dim>::parse_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Visualization");
          {
            prm.enter_subsection("Volume of Fluid");
            {
              include_contour = prm.get_bool("Include internal reconstruction contour");
              include_normal = prm.get_bool("Include normals");

              for (unsigned int f=0; f<this->get_volume_of_fluid_handler().get_n_fields(); ++f)
                {
                  std::string field_name = this->get_volume_of_fluid_handler().name_for_field_index(f);
                  volume_of_fluid_names.push_back("volume_fraction_"+field_name);
                  interp.push_back(DataComponentInterpretation::component_is_scalar);

                  if (include_contour)
                    {
                      volume_of_fluid_names.push_back("volume_of_fluid_contour_"+field_name);
                      interp.push_back(DataComponentInterpretation::component_is_scalar);
                    }

                  if (include_normal)
                    {
                      for (unsigned int i=0; i<dim; ++i)
                        {
                          volume_of_fluid_names.push_back("volume_of_fluid_interface_normal_"+field_name);
                          interp.push_back(DataComponentInterpretation::component_is_part_of_vector);
                        }
                    }
                }
            }
            prm.leave_subsection();
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      ASPECT_REGISTER_VISUALIZATION_POSTPROCESSOR(VolumeOfFluidValues,
                                                  "volume of fluid values",
                                                  "A visualization output object that outputs the  volume_of_fluid data."
                                                  "Names are given in Postprocess/Visualization/Volume of fluid")
    }
  }
}
