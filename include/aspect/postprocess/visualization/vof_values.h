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


#ifndef __aspect__postprocess_visualization_vof_values_h
#define __aspect__postprocess_visualization_vof_values_h

#include <aspect/postprocess/visualization.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      /**
       * A class derived that implements a function that provides volume of
       * fluid interface tracking data for graphical output.
       */
      template <int dim>
      class VoFValues
        : public DataPostprocessor<dim>,
          public SimulatorAccess<dim>,
          public Interface<dim>
      {
        public:
          /**
           * Standard constructor
           */
          VoFValues ();

          /**
           * Get the list of names for the components that will be produced by
           * this postprocessor
           */
          virtual
          std::vector<std::string>
          get_names () const;

          /**
           * Get the list of component interpretations for the components that
           * will be produced by this postprocessor
           */
          virtual
          std::vector<DataComponentInterpretation::DataComponentInterpretation>
          get_data_component_interpretation () const;

          /**
           * Get required update flags
           */
          virtual
          UpdateFlags
          get_needed_update_flags () const;

          /**
           * Produce that data based on provided solution data
           */
          virtual
          void
          evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                                std::vector<Vector<double>> &computed_quantities) const;

          /**
           * Declare the parameters this class takes through input files.
           */
          static
          void
          declare_parameters (ParameterHandler &prm);

          /**
           * Read the parameters this class declares from the parameter file.
           */
          virtual
          void
          parse_parameters (ParameterHandler &prm);

        private:
          /**
           * Stored list of names for the produced components
           */
          std::vector<std::string> vof_names;

          /**
           * Stored list of interpretations for produced components
           */
          std::vector<DataComponentInterpretation::DataComponentInterpretation> interp;

          /**
           * If true, the data output will include a field that has the
           * reconstructed fluid interface as the zero contour
           */
          bool include_contour;

          /**
           * If true, the data output will include the normal vector for the reconstructed interface
           */
          bool include_normal;
      };
    }
  }
}

#endif
