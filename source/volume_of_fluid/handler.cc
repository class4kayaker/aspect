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
#include <aspect/global.h>
#include <aspect/volume_of_fluid/handler.h>
#include <aspect/volume_of_fluid/assembly.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_cartesian.h>

using namespace dealii;

namespace aspect
{

  template <int dim>
  VolumeOfFluidField<dim>::VolumeOfFluidField(const FEVariable<dim> &volume_fraction,
                          const FEVariable<dim> &reconstruction,
                          const FEVariable<dim> &level_set)
    : volume_fraction (volume_fraction),
      reconstruction (reconstruction),
      level_set (level_set)
  {}

  template <int dim>
  VolumeOfFluidHandler<dim>::VolumeOfFluidHandler (Simulator<dim> &simulator,
                               ParameterHandler &prm)
    : sim (simulator),
      volume_of_fluid_initial_conditions (VolumeOfFluidInitialConditions::create_initial_conditions<dim>(prm)),
      assembler (),
      direction_order_descending(false)
  {
    this->initialize_simulator(sim);
    assembler.initialize_simulator(sim);
    parse_parameters (prm);
    assembler.set_volume_fraction_threshold(volume_fraction_threshold);

    this->get_signals().edit_finite_element_variables.connect(std_cxx11::bind(&aspect::VolumeOfFluidHandler<dim>::edit_finite_element_variables,
                                                                              std_cxx11::ref(*this),
                                                                              std_cxx11::_1));
    this->get_signals().post_set_initial_state.connect(std_cxx11::bind(&aspect::VolumeOfFluidHandler<dim>::set_initial_volume_of_fluids,
                                                                       std_cxx11::ref(*this)));
  }

  template <int dim>
  void
  VolumeOfFluidHandler<dim>::edit_finite_element_variables (std::vector<VariableDeclaration<dim> > &vars)
  {
    for (unsigned int f=0; f<n_volume_of_fluid_fields; ++f)
      {
        // Add declaration for volume fraction field
        vars.push_back(VariableDeclaration<dim>("volume_fraction_"+volume_of_fluid_field_names[f],
                                                std_cxx11::shared_ptr<FiniteElement<dim>>(
                                                  new FE_DGQ<dim>(0)),
                                                1,
                                                1));

        // Add declaration for reconstructed interface cache
        vars.push_back(VariableDeclaration<dim>("volume_of_fluid_interface_reconstruction_"+volume_of_fluid_field_names[f],
                                                std_cxx11::shared_ptr<FiniteElement<dim>>(
                                                  new FE_DGQ<dim>(0)),
                                                dim+1,
                                                1));

        vars.push_back(VariableDeclaration<dim>("volume_of_fluid_contour_"+volume_of_fluid_field_names[f],
                                                std_cxx11::shared_ptr<FiniteElement<dim>>(
                                                  new FE_DGQ<dim>(1)),
                                                1,
                                                1));
      }
  }

  template <int dim>
  void
  VolumeOfFluidHandler<dim>::declare_parameters (ParameterHandler &prm)
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
                         "The relative tolerance up to which the linear system"
                         "for the Volume of Fluid system gets solved. See"
                         "'Solver parameters/Composition solver tolerance'"
                         "for more details.");

      prm.declare_entry ("Volume of Fluid field names", "",
                         Patterns::List(Patterns::Anything()),
                         "User-defined names for Volume of Fluid fields.");

      prm.declare_entry ("Volume of Fluid composition mapping", "",
                         Patterns::Map(Patterns::Anything(), Patterns::Anything()),
                         "Links between composition and Volume of Fluid fields in composition:VolumeOfFluid form");
    }
    prm.leave_subsection ();
  }

  template <int dim>
  void
  VolumeOfFluidHandler<dim>::parse_parameters (ParameterHandler &prm)
  {
    prm.enter_subsection ("Volume of Fluid");
    {
      volume_fraction_threshold = prm.get_double("Volume fraction threshold");

      volume_of_fluid_solver_tolerance = prm.get_double("Volume of Fluid solver tolerance");

      n_volume_of_fluid_fields = prm.get_integer("Number of fields");

      volume_of_fluid_field_names = Utilities::split_string_list (prm.get("Volume of Fluid field names"));
      AssertThrow((volume_of_fluid_field_names.size() == 0) ||
                  (volume_of_fluid_field_names.size() == n_volume_of_fluid_fields),
                  ExcMessage("The length of the list of names for the Volume of Fluid fields "
                             "needs to either be empty or have length equal to the "
                             "number of compositional fields."));

      // check that names use only allowed characters, are not empty strings, and are unique
      for (unsigned int i=0; i<volume_of_fluid_field_names.size(); ++i)
        {
          Assert (volume_of_fluid_field_names[i].find_first_not_of("abcdefghijklmnopqrstuvwxyz"
                                                       "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                                       "0123456789_") == std::string::npos,
                  ExcMessage("Invalid character in field " + volume_of_fluid_field_names[i] + ". "
                             "Names of Volume of Fluid fields should consist of a "
                             "combination of letters, numbers and underscores."));
          Assert (volume_of_fluid_field_names[i].size() > 0,
                  ExcMessage("Invalid name of field " + volume_of_fluid_field_names[i] + ". "
                             "Names of Volume of Fluid fields need to be non-empty."));
          for (unsigned int j=0; j<i; ++j)
            Assert (volume_of_fluid_field_names[i] != volume_of_fluid_field_names[j],
                    ExcMessage("Names of Volume of Fluid fields have to be unique! " + volume_of_fluid_field_names[i] +
                               " is used more than once."));
        }

      // default names if not empty
      if (volume_of_fluid_field_names.size()==0)
        {
          for (unsigned int i=0; i<n_volume_of_fluid_fields; ++i)
            volume_of_fluid_field_names.push_back("F_" + Utilities::int_to_string(i+1));
        }

      const std::vector<std::string> x_volume_of_fluid_composition_vars =
        Utilities::split_string_list
        (prm.get ("Volume of Fluid composition mapping"));

      if (x_volume_of_fluid_composition_vars.size()>0 && !this->get_parameters().use_discontinuous_composition_discretization)
        {
          AssertThrow(false, ExcMessage("Volume of Fluid composition field not implemented for continuous composition."));
        }

      for (std::vector<std::string>::const_iterator p = x_volume_of_fluid_composition_vars.begin();
           p != x_volume_of_fluid_composition_vars.end(); ++p)
        {
          const std::vector<std::string> split_parts = Utilities::split_string_list(*p, ':');
          AssertThrow (split_parts.size() == 2,
                       ExcMessage("The format for Volume of Fluid composition mappings requires that each entry has the form"
                                  "`composition:Vof field', but there does not appear to be a colon in the entry <" + *p + ">."));

          const std::string composition_field = split_parts[0];
          const std::string volume_of_fluid_field = split_parts[1];

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

          // Check volume_of_fluid_field exists
          field_exists=false;
          unsigned int volume_of_fluid_field_index = n_volume_of_fluid_fields;

          for (unsigned int i=0; i<n_volume_of_fluid_fields; ++i)
            {
              if (volume_of_fluid_field == volume_of_fluid_field_names[i])
                {
                  field_exists = true;
                  volume_of_fluid_field_index = i;
                  break;
                }
            }

          Assert(field_exists, ExcMessage("Volume of Fluid field variable " +
                                          volume_of_fluid_field +
                                          " does not exist."));

          // Ensure no duplicate mapping to a composition field
          Assert (volume_of_fluid_composition_map_index.count(composition_field) == 0,
                  ExcMessage("Volume of Fluid composition field mappings have to be unique! " + composition_field +
                             " is used more than once."));

          // Add to mappings
          volume_of_fluid_composition_map_index[composition_field] = volume_of_fluid_field_index;
        }
    }
    prm.leave_subsection ();
  }

  template <int dim>
  void
  VolumeOfFluidHandler<dim>::initialize (ParameterHandler &prm)
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

    if ( this->get_parameters().initial_adaptive_refinement > 0 ||
         this->get_parameters().adaptive_refinement_interval > 0 )
      {
        prm.enter_subsection("Mesh refinement");
        {
          std::vector<std::string> plugin_names
            = Utilities::split_string_list(prm.get("Strategy"));

          bool has_volume_of_fluid_strategy = false;
          for (unsigned int name=0; name<plugin_names.size(); ++name)
            {
              has_volume_of_fluid_strategy = (plugin_names[name] == "volume of fluid interface") || has_volume_of_fluid_strategy;
            }
          AssertThrow(has_volume_of_fluid_strategy,
                      ExcMessage("Volume of Fluid Interface Tracking requires that the 'volume of fluid interface' strategy be used for AMR"));
        }
        prm.leave_subsection();

        AssertThrow(this->get_parameters().adaptive_refinement_interval <(1/this->get_parameters().CFL_number),
                    ExcMessage("Volume of Fluid Interface Tracking requires that the AMR interval be less than 1/CFL_number"));
      }

    // Gather the created volume fraction data into a structure for easier programmatic referencing

    for (unsigned int f=0; f<n_volume_of_fluid_fields; ++f)
      {
        data.push_back(VolumeOfFluidField<dim>(this->introspection().variable("volume_fraction_"+volume_of_fluid_field_names[f]),
                                     this->introspection().variable("volume_of_fluid_interface_reconstruction_"+volume_of_fluid_field_names[f]),
                                     this->introspection().variable("volume_of_fluid_contour_"+volume_of_fluid_field_names[f])));
      }

    // Do initial conditions setup
    if (SimulatorAccess<dim> *sim_a = dynamic_cast<SimulatorAccess<dim>*>(volume_of_fluid_initial_conditions.get()))
      sim_a->initialize_simulator (sim);
    volume_of_fluid_initial_conditions->parse_parameters (prm);
    volume_of_fluid_initial_conditions->initialize ();

  }

  template <int dim>
  unsigned int VolumeOfFluidHandler<dim>::get_n_fields() const
  {
    return n_volume_of_fluid_fields;
  }

  template <int dim>
  const std::string VolumeOfFluidHandler<dim>::name_for_field_index(unsigned int field) const
  {
    Assert(field < n_volume_of_fluid_fields,
           ExcMessage("Invalid field index"));
    return volume_of_fluid_field_names[field];
  }

  template <int dim>
  double VolumeOfFluidHandler<dim>::get_volume_fraction_threshold() const
  {
    return volume_fraction_threshold;
  }

  template <int dim>
  const VolumeOfFluidField<dim> &VolumeOfFluidHandler<dim>::field_struct_for_field_index(unsigned int field) const
  {
    Assert(field < n_volume_of_fluid_fields,
           ExcMessage("Invalid field index"));
    return data[field];
  }

  template <int dim>
  unsigned int VolumeOfFluidHandler<dim>::field_index_for_name(std::string composition_fieldname) const
  {
    if (volume_of_fluid_composition_map_index.count(composition_fieldname) ==0)
      return n_volume_of_fluid_fields;
    return volume_of_fluid_composition_map_index.at(composition_fieldname);
  }

  template <int dim>
  void VolumeOfFluidHandler<dim>::do_volume_of_fluid_update ()
  {
    for (unsigned int f=0; f<n_volume_of_fluid_fields; ++f)
      {
        const unsigned int volume_of_fluid_block_idx = data[f].volume_fraction.block_index;
        const unsigned int volume_of_fluidN_block_idx = data[f].reconstruction.block_index;

        // Due to dimensionally split formulation, use Strang (second-order dimensional) splitting
        for (unsigned int direction = 0; direction < dim; ++direction)
          {
            // Only reference old_solution for data from prior substep if this is the first
            // substep for dimensional splitting
            bool update_from_old = (direction == 0);
            // Update base to intermediate solution
            if (!direction_order_descending)
              {
                assemble_volume_of_fluid_system(data[f], direction, update_from_old);
              }
            else
              {
                assemble_volume_of_fluid_system(data[f], dim-direction-1, update_from_old);
              }
            solve_volume_of_fluid_system (data[f]);
            // Copy current candidate normals.
            // primarily useful for exact linear translation
            sim.solution.block(volume_of_fluidN_block_idx) = sim.old_solution.block(volume_of_fluidN_block_idx);
            update_volume_of_fluid_normals (data[f], sim.solution);

            sim.current_linearization_point.block(volume_of_fluid_block_idx) = sim.solution.block(volume_of_fluid_block_idx);
            sim.current_linearization_point.block(volume_of_fluidN_block_idx) = sim.solution.block(volume_of_fluidN_block_idx);
          }
      }
    for (std::map<std::string, unsigned int>::const_iterator iter=volume_of_fluid_composition_map_index.begin();
         iter!=volume_of_fluid_composition_map_index.end(); ++iter)
      {
        const unsigned int composition_index = this->introspection().compositional_index_for_name(iter->first);
        const typename Simulator<dim>::AdvectionField advection_field = Simulator<dim>::AdvectionField::composition(composition_index);
        const VolumeOfFluidField<dim> volume_of_fluid_f= field_struct_for_field_index(iter->second);
        update_volume_of_fluid_composition(advection_field, volume_of_fluid_f, sim.solution);
      }
    // change dimension iteration order
    direction_order_descending = !direction_order_descending;
  }
}

namespace aspect
{
#define INSTANTIATE(dim) \
  template class VolumeOfFluidHandler<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)
}
