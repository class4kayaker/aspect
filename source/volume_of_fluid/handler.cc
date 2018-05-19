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
#include <aspect/parameters.h>
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
      assembler ()
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
      prm.declare_entry ("Volume fraction threshold", "1e-6",
                         Patterns::Double (0, 1),
                         "Minimum significant volume. VOFs below this considered to be zero.");

      prm.declare_entry ("Volume of Fluid solver tolerance", "1e-12",
                         Patterns::Double(0,1),
                         "The relative tolerance up to which the linear system"
                         "for the Volume of Fluid system gets solved. See"
                         "'Solver parameters/Composition solver tolerance'"
                         "for more details.");
    }
    prm.leave_subsection ();
  }

  template <int dim>
  void
  VolumeOfFluidHandler<dim>::parse_parameters (ParameterHandler &prm)
  {
    // Get parameter data
    n_volume_of_fluid_fields = 0;
    std::vector<std::string> names_of_compositional_fields = this->get_parameters().names_of_compositional_fields;
    std::vector<typename Parameters<dim>::AdvectionFieldMethod::Kind> compositional_field_methods = this->get_parameters().compositional_field_methods;

    for (unsigned int i=0; i<names_of_compositional_fields.size(); ++i)
      {
        if (compositional_field_methods[i] == Parameters<dim>::AdvectionFieldMethod::volume_of_fluid)
          {
            // Add this field as the next volume of fluid field
            volume_of_fluid_field_names.push_back(names_of_compositional_fields[i]);
            volume_of_fluid_composition_map_index[i] = n_volume_of_fluid_fields;
            ++n_volume_of_fluid_fields;
          }
      }

    prm.enter_subsection ("Volume of Fluid");
    {
      volume_fraction_threshold = prm.get_double("Volume fraction threshold");

      volume_of_fluid_solver_tolerance = prm.get_double("Volume of Fluid solver tolerance");
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
    const unsigned int composition_index = this->introspection().compositional_index_for_name(composition_fieldname);
    if (volume_of_fluid_composition_map_index.count(composition_index) ==0)
      return n_volume_of_fluid_fields;
    return volume_of_fluid_composition_map_index.at(composition_index);
  }

  template <int dim>
  void VolumeOfFluidHandler<dim>::do_volume_of_fluid_update (const typename Simulator<dim>::AdvectionField advection_field)
  {
    const bool direction_order_descending = (this->get_timestep_number() % 2) == 1;
    const VolumeOfFluidField<dim> volume_of_fluid_field = data[volume_of_fluid_composition_map_index[advection_field.field_index()]];

    const unsigned int volume_of_fluid_block_idx = volume_of_fluid_field.volume_fraction.block_index;
    const unsigned int volume_of_fluidN_block_idx = volume_of_fluid_field.reconstruction.block_index;

    // Due to dimensionally split formulation, use Strang (second-order dimensional) splitting
    for (unsigned int direction = 0; direction < dim; ++direction)
      {
        // Only reference old_solution for data from prior substep if this is the first
        // substep for dimensional splitting
        bool update_from_old = (direction == 0);
        // Update base to intermediate solution
        if (!direction_order_descending)
          {
            assemble_volume_of_fluid_system(volume_of_fluid_field, direction, update_from_old);
          }
        else
          {
            assemble_volume_of_fluid_system(volume_of_fluid_field, dim-direction-1, update_from_old);
          }
        solve_volume_of_fluid_system (volume_of_fluid_field);
        // Copy current candidate normals.
        // primarily useful for exact linear translation
        sim.solution.block(volume_of_fluidN_block_idx) = sim.old_solution.block(volume_of_fluidN_block_idx);
        update_volume_of_fluid_normals (volume_of_fluid_field, sim.solution);

        sim.current_linearization_point.block(volume_of_fluid_block_idx) = sim.solution.block(volume_of_fluid_block_idx);
        sim.current_linearization_point.block(volume_of_fluidN_block_idx) = sim.solution.block(volume_of_fluidN_block_idx);
      }
    update_volume_of_fluid_composition(advection_field, volume_of_fluid_field, sim.solution);
  }
}

namespace aspect
{
#define INSTANTIATE(dim) \
  template class VolumeOfFluidHandler<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)
}
