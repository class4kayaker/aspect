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
#include <aspect/global.h>
#include <aspect/vof/handler.h>
#include <aspect/vof/assembly.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_cartesian.h>

using namespace dealii;

namespace aspect
{

  template <int dim>
  VoFField<dim>::VoFField(const FEVariable<dim> &fraction,
                          const FEVariable<dim> &reconstruction,
                          const FEVariable<dim> &level_set)
    : fraction (fraction),
      reconstruction (reconstruction),
      level_set (level_set)
  {}

  template <int dim>
  VoFHandler<dim>::VoFHandler (Simulator<dim> &simulator,
                               ParameterHandler &prm)
    : sim (simulator),
      vof_initial_conditions (VoFInitialConditions::create_initial_conditions<dim>(prm)),
      assembler (),
      vof_reconstruct_epsilon(1e-13)
  {
    this->initialize_simulator(sim);
    assembler.initialize_simulator(sim);
    parse_parameters (prm);
    assembler.set_vof_epsilon(vof_epsilon);

    this->get_signals().edit_finite_element_variables.connect(std_cxx11::bind(&aspect::VoFHandler<dim>::edit_finite_element_variables,
                                                                              std_cxx11::ref(*this),
                                                                              std_cxx11::_1));
    this->get_signals().post_set_initial_state.connect(std_cxx11::bind(&aspect::VoFHandler<dim>::set_initial_vofs,
                                                                       std_cxx11::ref(*this)));
  }

  template <int dim>
  void
  VoFHandler<dim>::edit_finite_element_variables (std::vector<VariableDeclaration<dim> > &vars)
  {
    for (unsigned int f=0; f<n_vof_fields; ++f)
      {
        vars.push_back(VariableDeclaration<dim>("vof_"+vof_field_names[f],
                                                std_cxx11::shared_ptr<FiniteElement<dim>>(
                                                  new FE_DGQ<dim>(0)),
                                                1,
                                                1));

        vars.push_back(VariableDeclaration<dim>("vofN_"+vof_field_names[f],
                                                std_cxx11::shared_ptr<FiniteElement<dim>>(
                                                  new FE_DGQ<dim>(0)),
                                                dim+1,
                                                1));

        vars.push_back(VariableDeclaration<dim>("vofLS_"+vof_field_names[f],
                                                std_cxx11::shared_ptr<FiniteElement<dim>>(
                                                  new FE_DGQ<dim>(1)),
                                                1,
                                                1));
      }
  }

  template <int dim>
  void
  VoFHandler<dim>::declare_parameters (ParameterHandler &prm)
  {
    prm.enter_subsection ("Volume of Fluid");
    {
      prm.declare_entry ("Number of fields", "1",
                         Patterns::Integer(0),
                         "The number of fields to be handled using Volume of Fluid interface tracking.");

      prm.declare_entry ("Volume fraction threshold", "1e-6",
                         Patterns::Double (0, 1),
                         "Minimum significant volume. VOFs below this considered to be zero.");

      prm.declare_entry ("Volume of Fluid solver tolerance", "1e-12",
                         Patterns::Double(0,1),
                         "The relative tolerance up to which the linear system for "
                         "the Volume of Fluid system gets solved. See 'linear solver "
                         "tolerance' for more details.");

      prm.declare_entry ("Volume of Fluid field names", "",
                         Patterns::List(Patterns::Anything()),
                         "User-defined names for Volume of Fluid fields.");

      prm.declare_entry ("Volume of Fluid composition mapping", "",
                         Patterns::Map(Patterns::Anything(), Patterns::Anything()),
                         "Links between composition and Volume of Fluid fields in composition:VoF form");
    }
    prm.leave_subsection ();
  }

  template <int dim>
  void
  VoFHandler<dim>::parse_parameters (ParameterHandler &prm)
  {
    prm.enter_subsection ("Volume of Fluid");
    {
      vof_epsilon = prm.get_double("Volume fraction threshold");

      vof_solver_tolerance = prm.get_double("Volume of Fluid solver tolerance");

      n_vof_fields = prm.get_integer("Number of fields");

      vof_field_names = Utilities::split_string_list (prm.get("Volume of Fluid field names"));
      AssertThrow((vof_field_names.size() == 0) ||
                  (vof_field_names.size() == n_vof_fields),
                  ExcMessage("The length of the list of names for the Volume of Fluid fields "
                             "needs to either be empty or have length equal to the "
                             "number of compositional fields."));

      // check that names use only allowed characters, are not empty strings, and are unique
      for (unsigned int i=0; i<vof_field_names.size(); ++i)
        {
          Assert (vof_field_names[i].find_first_not_of("abcdefghijklmnopqrstuvwxyz"
                                                       "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                                       "0123456789_") == std::string::npos,
                  ExcMessage("Invalid character in field " + vof_field_names[i] + ". "
                             "Names of Volume of Fluid fields should consist of a "
                             "combination of letters, numbers and underscores."));
          Assert (vof_field_names[i].size() > 0,
                  ExcMessage("Invalid name of field " + vof_field_names[i] + ". "
                             "Names of Volume of Fluid fields need to be non-empty."));
          for (unsigned int j=0; j<i; ++j)
            Assert (vof_field_names[i] != vof_field_names[j],
                    ExcMessage("Names of Volume of Fluid fields have to be unique! " + vof_field_names[i] +
                               " is used more than once."));
        }

      // default names if not empty
      if (vof_field_names.size()==0)
        {
          for (unsigned int i=0; i<n_vof_fields; ++i)
            vof_field_names.push_back("F_" + Utilities::int_to_string(i+1));
        }

      const std::vector<std::string> x_vof_composition_vars =
        Utilities::split_string_list
        (prm.get ("Volume of Fluid composition mapping"));

      if (x_vof_composition_vars.size()>0 && !this->get_parameters().use_discontinuous_composition_discretization)
        {
          AssertThrow(false, ExcMessage("Volume of Fluid composition field not implemented for continuous composition."));
        }

      for (std::vector<std::string>::const_iterator p = x_vof_composition_vars.begin();
           p != x_vof_composition_vars.end(); ++p)
        {
          const std::vector<std::string> split_parts = Utilities::split_string_list(*p, ':');
          AssertThrow (split_parts.size() == 2,
                       ExcMessage("The format for Volume of Fluid composition mappings requires that each entry has the form"
                                  "`composition:Vof field', but there does not appear to be a colon in the entry <" + *p + ">."));

          const std::string composition_field = split_parts[0];
          const std::string vof_field = split_parts[1];

          // Check composition_field exists
          bool field_exists=false;

          for (unsigned int i=0; i<this->get_parameters().n_compositional_fields; ++i)
            {
              field_exists = field_exists ||
                             (composition_field==this->get_parameters().names_of_compositional_fields[i]);
            }

          Assert(field_exists, ExcMessage("Composition field variable " +
                                          composition_field +
                                          " does not exist."));

          // Check vof_field exists
          field_exists=false;
          unsigned int vof_field_index = n_vof_fields;

          for (unsigned int i=0; i<n_vof_fields; ++i)
            {
              if (vof_field == vof_field_names[i])
                {
                  field_exists = true;
                  vof_field_index = i;
                  break;
                }
            }

          Assert(field_exists, ExcMessage("Volume of Fluid field variable " +
                                          vof_field +
                                          " does not exist."));

          // Ensure no duplicate mapping to a composition field
          Assert (vof_composition_map_index.count(composition_field) == 0,
                  ExcMessage("Volume of Fluid composition field mappings have to be unique! " + composition_field +
                             " is used more than once."));


          // Add to mappings
          vof_composition_map_index[composition_field] = vof_field_index;
        }
    }
    prm.leave_subsection ();
  }

  template <int dim>
  void
  VoFHandler<dim>::initialize (ParameterHandler &prm)
  {
    // Do checks on required assumptions
    AssertThrow(dim==2,
                ExcMessage("Volume of Fluid Interface Tracking is currently only functional for dim=2."));

    AssertThrow(this->get_parameters().CFL_number < 1.0,
                ExcMessage("Volume of Fluid Interface Tracking requires CFL < 1."));

    AssertThrow(!this->get_material_model().is_compressible(),
                ExcMessage("Volume of Fluid Interface Tracking currently assumes incompressiblity."));

    AssertThrow(dynamic_cast<const MappingCartesian<dim> *>(&(this->get_mapping())),
                ExcMessage("Volume of Fluid Interface Tracking currently requires Cartesian Mappings"));

    AssertThrow(!this->get_parameters().free_surface_enabled,
                ExcMessage("Volume of Fluid Interface Tracking is currently incompatible with the Free Surface implementation."));

    AssertThrow(!this->get_parameters().include_melt_transport,
                ExcMessage("Volume of Fluid Interface Tracking has not been tested with melt transport yet, so inclusion of both is currently disabled."))

    // Check for correct mapping

    if ( this->get_parameters().initial_adaptive_refinement > 0 ||
         this->get_parameters().adaptive_refinement_interval > 0 )
      {
        // AMR active so check refinement strategy includes 'vof boundary'
        // Need to identify method of accessing list of refinement strategies
      }

    // Gather data

    for (unsigned int f=0; f<n_vof_fields; ++f)
      {
        data.push_back(VoFField<dim>(this->introspection().variable("vof_"+vof_field_names[f]),
                                     this->introspection().variable("vofN_"+vof_field_names[f]),
                                     this->introspection().variable("vofLS_"+vof_field_names[f])));
      }

    // Do initial conditions setup
    if (SimulatorAccess<dim> *sim_a = dynamic_cast<SimulatorAccess<dim>*>(vof_initial_conditions.get()))
      sim_a->initialize_simulator (sim);
    if (vof_initial_conditions.get())
      {
        vof_initial_conditions->parse_parameters (prm);
        vof_initial_conditions->initialize ();
      }

  }

  template <int dim>
  unsigned int VoFHandler<dim>::get_n_fields() const
  {
    return n_vof_fields;
  }

  template <int dim>
  const std::string VoFHandler<dim>::get_field_name(unsigned int field) const
  {
    Assert(field < n_vof_fields,
           ExcMessage("Invalid field index"));
    return vof_field_names[field];
  }

  template <int dim>
  double VoFHandler<dim>::get_vof_epsilon() const
  {
    return vof_epsilon;
  }

  template <int dim>
  const VoFField<dim> &VoFHandler<dim>::get_field(unsigned int field) const
  {
    Assert(field < n_vof_fields,
           ExcMessage("Invalid field index"));
    return data[field];
  }

  template <int dim>
  unsigned int VoFHandler<dim>::get_vof_field(std::string composition_fieldname) const
  {
    if (vof_composition_map_index.count(composition_fieldname) ==0)
      return n_vof_fields;
    return vof_composition_map_index.at(composition_fieldname);
  }

  template <int dim>
  void VoFHandler<dim>::do_vof_update ()
  {
    for (unsigned int f=0; f<n_vof_fields; ++f)
      {
        const unsigned int vof_block_idx = data[f].fraction.block_index;
        const unsigned int vofN_block_idx = data[f].reconstruction.block_index;

        // Reset current base to values at beginning of timestep

        // Due to dimensionally split formulation, use strang splitting
        // TODO: Reformulate for unsplit (may require flux limiter)
        bool update_from_old = true;
        for (unsigned int dir = 0; dir < dim; ++dir)
          {
            // Update base to intermediate solution
            if (!vof_dir_order_dsc)
              {
                assemble_vof_system(data[f], dir, update_from_old);
              }
            else
              {
                assemble_vof_system(data[f], dim-dir-1, update_from_old);
              }
            solve_vof_system (data[f]);
            // Copy current candidate normals.
            // primarily useful for exact linear translation
            sim.solution.block(vofN_block_idx) = sim.old_solution.block(vofN_block_idx);
            update_vof_normals (data[f], sim.solution);

            sim.current_linearization_point.block(vof_block_idx) = sim.solution.block(vof_block_idx);
            sim.current_linearization_point.block(vofN_block_idx) = sim.solution.block(vofN_block_idx);
            update_from_old = false;
          }
      }
    for (std::map<std::string, unsigned int>::const_iterator iter=vof_composition_map_index.begin();
         iter!=vof_composition_map_index.end(); ++iter)
      {
        const unsigned int c_var_index = this->introspection().compositional_index_for_name(iter->first);
        const typename Simulator<dim>::AdvectionField adv_f = Simulator<dim>::AdvectionField::composition(c_var_index);
        const VoFField<dim> vof_f= get_field(iter->second);
        update_vof_composition(adv_f, vof_f, sim.solution);
      }
    // change dimension iteration order
    vof_dir_order_dsc = !vof_dir_order_dsc;
  }
}

namespace aspect
{
#define INSTANTIATE(dim) \
  template class VoFHandler<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)
}
