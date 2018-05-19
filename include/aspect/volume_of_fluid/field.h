/*
  Copyright (C) 2011 - 2017 by the authors of the ASPECT code.

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

#ifndef __aspect_volume_of_fluid_field_h
#define __aspect_volume_of_fluid_field_h

#include <aspect/fe_variable_collection.h>

using namespace dealii;

namespace aspect
{
  /**
   * Structure to package the relevant data (both state and cached) in a single
   * location for access.
   */
  template<int dim>
  struct VolumeOfFluidField
  {
    /**
     * Initialize the structure with FEVariables to hold the required
     * information that must be available on all cells.
     */
    VolumeOfFluidField(const FEVariable<dim> &volume_fraction,
                       const FEVariable<dim> &reconstruction,
                       const FEVariable<dim> &level_set,
                       const unsigned int composition_index);

    /**
     * Field to hold the current volume fraction.
     */
    const FEVariable<dim> &volume_fraction;

    /**
     * Field to hold the cached interface reconstruction.
     */
    const FEVariable<dim> &reconstruction;

    /**
     * Field to expose reconstructed interface as a zero-contour to output in
     * visualization plugin.
     */
    const FEVariable<dim> &level_set;

    /**
     * Field to hold the index of the associated composition field
     */
    const unsigned int composition_index;

  };
}

#endif
