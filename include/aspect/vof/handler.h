/*
  Copyright (C) 2011 - 2016 by the authors of the ASPECT code.

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

#ifndef __aspect_vof_handler_h
#define __aspect_vof_handler_h

#include <aspect/simulator.h>
#include <aspect/simulator_access.h>
#include <aspect/vof_initial_conditions/interface.h>
#include <aspect/vof/field.h>
#include <aspect/vof/assembly.h>

using namespace dealii;

namespace aspect
{
  /**
   * A member class that isolates the functions and variables that deal
   * with the volume of fluid implementation. If Volume of Fluid interface
   * tracking is not active, there is no instantiation of this class at
   * all.
   */
  template <int dim>
  class VoFHandler : public SimulatorAccess<dim>
  {
    public:
      // Construtor
      VoFHandler(Simulator<dim> &sim, ParameterHandler &prm);

      void edit_finite_element_variables (std::vector<VariableDeclaration<dim> > &vars);

      // Parameter handling
      static
      void declare_parameters (ParameterHandler &prm);

      void parse_parameters (ParameterHandler &prm);

      // Get VoF data
      unsigned int get_n_fields() const;
      const std::string get_field_name(unsigned int i) const;
      const VoFField<dim> &get_field(unsigned int i) const;
      double get_vof_epsilon() const;
      unsigned int get_vof_field(std::string composition_fieldname) const;

      // initialiation
      void initialize (ParameterHandler &prm);

      // Functions for initialization of state
      void set_initial_vofs ();
      void init_vof_compos (const VoFField<dim> field, const unsigned int f_ind);
      void init_vof_ls (const VoFField<dim> field, const unsigned int f_ind);

      // Do interface reconstruction
      void update_vof_normals (const VoFField<dim> field,
                               LinearAlgebra::BlockVector &solution);
      void update_vof_composition (const typename Simulator<dim>::AdvectionField composition_field,
                                   const VoFField<dim> vof_field,
                                   LinearAlgebra::BlockVector &solution);

      // Logic to handle dimensionally split update
      void do_vof_update ();

      // Assembly
      void assemble_vof_system (const VoFField<dim> field,
                                unsigned int dir,
                                bool update_from_old);
      /**
       * Function to copy assembled data to final system. Requires access to
       * the full matrix, so must be in this class.
       */
      void copy_local_to_global_vof_system (const internal::Assembly::CopyData::VoFSystem<dim> &data);
      // Solver
      void solve_vof_system (const VoFField<dim> field);


    private:
      // Parent simulator
      Simulator<dim> &sim;

      //Initial conditions
      const std_cxx11::unique_ptr<VoFInitialConditions::Interface<dim> >      vof_initial_conditions;

      // Update assembler
      Assemblers::VoFAssembler<dim> assembler;

      // Store
      unsigned int n_vof_fields;
      std::vector<VoFField<dim>> data;

      // Minimal considered volume fraction
      double vof_epsilon;

      double vof_solver_tolerance;

      std::vector<std::string> vof_field_names;
      std::map<std::string, unsigned int> vof_composition_map_index;

      // Order for split update
      bool vof_dir_order_dsc;

      friend class Simulator<dim>;
      friend class SimulatorAccess<dim>;
  };

}

#endif
