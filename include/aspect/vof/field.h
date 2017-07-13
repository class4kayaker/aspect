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

#ifndef __aspect_vof_field_h
#define __aspect_vof_field_h

#include <aspect/fe_variable_collection.h>

using namespace dealii;

namespace aspect
{
  template<int dim>
  struct VoFField
  {
    /**
     * Initialize th
     */
    VoFField(const FEVariable<dim> &fraction,
             const FEVariable<dim> &reconstruction,
             const FEVariable<dim> &level_set,
             const std::string c_field_name);

    const FEVariable<dim> &fraction, &reconstruction, &level_set;

    const std::string c_field_name;
  };
}

#endif
