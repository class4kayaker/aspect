/*
Copyright (C) 2011 - 2015 by the authors of the ASPECT code.

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



#include <aspect/global.h>
#include <aspect/simulator_access.h>
#include <aspect/postprocess/interface.h>
#include <aspect/vof/utilities.h>
#include <aspect/vof/handler.h>

// Deal II includes
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>


namespace aspect
{
  namespace Postprocess
  {
    using namespace dealii;

    template <int dim>
    class VoFMMSErr : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        VoFMMSErr();

        double get_next_t (double curr_time, double interval);

        /**
         * Evaluate the solution for some velocity statistics.
         */
        virtual
        std::pair<std::string,std::string>
        execute (TableHandler &);

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
        // Error calculation functions
        static std::vector<std::string> error_names ();
        static std::vector<std::string> error_abrev ();
        std::vector<double> calc_error_ls (Function<dim> &func,
                                           unsigned int n_samp,
                                           unsigned int f_ind);

        // Required variables
        bool initialized;
        double err_interval;
        double next_err_t;
        unsigned int n_e_samp;

        /**
         * Level set function for true solution
         */
        std_cxx11::unique_ptr<Functions::ParsedFunction<dim>> trueSolLS;
    };

    template <int dim>
    VoFMMSErr<dim>::VoFMMSErr ()
      : initialized(false),
        err_interval (std::numeric_limits<double>::quiet_NaN ()),
        next_err_t (std::numeric_limits<double>::quiet_NaN ()),
        n_e_samp (1)
    {
    }

    template <int dim>
    double VoFMMSErr<dim>::get_next_t (double curr_time,
                                       double interval)
    {
      int i = curr_time / interval;
      return (i + 1) * interval;
    }

    template <int dim>
    std::pair<std::string, std::string>
    VoFMMSErr<dim>::execute (TableHandler &)
    {
      std::string result_string = "";
      std::string label_string = "";
      if (!initialized)
        {
          initialized = true;
          next_err_t = this->get_time ();
        }

      if (this->get_time () >= next_err_t)
        {
          std::vector<std::string> err_abrev = error_abrev ();
          double curr_time = this->get_time();
          if (this->convert_output_to_years())
            curr_time /= year_in_seconds;
          trueSolLS->set_time(curr_time);

          std::ostringstream label_stream;
          std::ostringstream err_str;
          label_stream << "VoF Calculation(";

          unsigned int n_err = err_abrev.size();
          std::vector<std::string>::iterator it = err_abrev.begin();
          for (; it!=err_abrev.end(); )
            {
              label_stream << *it;
              if (++it!=err_abrev.end())
                label_stream << "/";
            }
          label_stream << "):";
          label_string = label_stream.str();

          unsigned int n_vof_fields = this->get_vof_handler().get_n_fields();
          std::vector<double> local_err_vals (n_vof_fields*n_err);
          for (unsigned int f=0; f<n_vof_fields; ++f)
            {
              std::vector<double> l_err_vals_f = calc_error_ls (*trueSolLS, n_e_samp, f);
              for (unsigned int i=0; i<n_err; ++i)
                local_err_vals[f*n_err+i] = l_err_vals_f[i];
            }

          std::vector<double> global_err_vals(n_vof_fields*n_err);
          Utilities::MPI::sum (local_err_vals, this->get_mpi_communicator(), global_err_vals);

          for (unsigned int f=0; f<n_vof_fields; ++f)
            {
              for (unsigned int i=0; i<n_err; ++i)
                {
                  err_str << std::scientific << std::setprecision (8)
                          << global_err_vals[f*n_err+i];
                  if (i+1<n_err)
                    err_str << "/";
                }
              if (f+1<n_vof_fields)
                err_str << "//";
            }
          result_string = err_str.str ();
          next_err_t = get_next_t (this->get_time (), err_interval);
        }
      return std::make_pair (label_string, result_string);
    }

    template <int dim>
    void
    VoFMMSErr<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("VoF MMS");
        {
          prm.declare_entry ("Time between error estimates", "1e8",
                             Patterns::Double (0),
                             "Time interval between error estimates.");

          prm.declare_entry ("Number error samples", "3",
                             Patterns::Integer (1),
                             "Number of sampled points per dimension when estimating error");

          prm.enter_subsection ("True LS");
          {
            Functions::ParsedFunction<dim>::declare_parameters (prm, 1);
          }
          prm.leave_subsection ();

        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    VoFMMSErr<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("VoF MMS");
        {
          n_e_samp = prm.get_integer ("Number error samples");

          err_interval = prm.get_double ("Time between error estimates");
          if (this->convert_output_to_years())
            err_interval *= year_in_seconds;
          prm.enter_subsection ("True LS");
          {
            trueSolLS.reset(new Functions::ParsedFunction<dim>(this->get_vof_handler().get_n_fields()));
            trueSolLS->parse_parameters (prm);
          }
          prm.leave_subsection ();
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    std::vector<std::string> VoFMMSErr<dim>::error_abrev ()
    {
      std::vector<std::string> names (1, "IEstL1");
      names.push_back ("FEstL1");
      return names;
    }

    template <int dim>
    std::vector<double> VoFMMSErr<dim>::calc_error_ls (Function<dim> &func,
                                                       unsigned int n_samp,
                                                       unsigned int f_ind)
    {
      const LinearAlgebra::BlockVector &solution = this->get_solution();

      double I_err_est = 0.0;
      double F_err_est = 0.0;
      double h = 1.0 / n_samp;

      Point<dim> xU;
      Tensor<1, dim, double> normal;
      double d;

      const DoFHandler<dim> &dof_handler = this->get_dof_handler();
      const FiniteElement<dim> &finite_element = this->get_fe();
      const unsigned int dofs_per_cell = finite_element.dofs_per_cell;
      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

      const QIterated<dim> quadrature (QMidpoint<1>(), n_samp);
      FEValues<dim> fe_err (this->get_mapping(), finite_element, quadrature,
                            update_JxW_values |
                            update_quadrature_points);

      const unsigned int vof_index = this->get_vof_handler().field_struct_for_field_index(f_ind)
          .volume_fraction.first_component_index;
      const unsigned int vofN_c_index = this->get_vof_handler().field_struct_for_field_index(f_ind)
          .reconstruction.first_component_index;

      // Initialize state based on provided function
      for (auto cell : dof_handler.active_cell_iterators ())
        {
          if (!cell->is_locally_owned ())
            continue;

          double cell_vof, cell_vol;
          double cell_diam;
          double d_func;

          // Obtain data for this cell and neighbors
          cell->get_dof_indices (local_dof_indices);
          for (unsigned int i=0; i<dim; ++i)
            normal[i] = solution(local_dof_indices[finite_element.component_to_system_index(vofN_c_index+i, 0)]);
          d = solution(local_dof_indices[finite_element.component_to_system_index(vofN_c_index+dim, 0)]);
          cell_vof = solution(local_dof_indices[finite_element.component_to_system_index(vof_index, 0)]);

          // Calculate approximation for volume
          fe_err.reinit (cell);

          cell_vol = cell->measure ();
          cell_diam = cell->diameter();
          d_func = func.value(cell->barycenter(), f_ind);
          double nnorm1 = 0;
          for (unsigned int i = 0; i < dim; ++i)
            {
              nnorm1 += abs (normal[i]);
            }

          if (abs (d) >= 0.5 * nnorm1 &&
              abs (d_func) >= 0.5 * cell_diam &&
              d * d_func >= 0.0)
            {
              continue;
            }

          Tensor<1, dim, double> grad_t;
          double d_t;
          double val = 0.0;
          double vof_reinit = 0.0;
          for (unsigned int i = 0; i < fe_err.n_quadrature_points; ++i)
            {
              d_t = 0.0;
              xU = quadrature.point (i);
              for (unsigned int di = 0; di < dim; ++di)
                {
                  Point<dim> xH, xL;
                  xH = xU;
                  xL = xU;
                  xH[di] += 0.5 * h;
                  xL[di] -= 0.5 * h;
                  double dH = func.value(cell->intermediate_point(xH), f_ind);
                  double dL = func.value(cell->intermediate_point(xL), f_ind);
                  grad_t[di] = (dL - dH);
                  d_t += (0.5 / dim) * (dH + dL);
                }
              double ptvof_t = VolumeOfFluid::compute_fluid_fraction<dim> (grad_t, d_t);
              vof_reinit += ptvof_t *(fe_err.JxW (i) / cell_vol);
              for (unsigned int di = 0; di < dim; ++di)
                {
                  xU[di] -= 0.5;
                }
              double dot = normal * xU;
              double ptvof_e = VolumeOfFluid::compute_fluid_fraction<dim> (h * normal,
                                                                           (d - dot));
              double diff = abs (ptvof_t - ptvof_e);
              val += diff * fe_err.JxW (i);
            }
          I_err_est += val;
          F_err_est += abs (cell_vof - vof_reinit) * cell_vol;
        }

      std::vector<double> errors(1, I_err_est);
      errors.push_back (F_err_est);
      return errors;
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(VoFMMSErr,
                                  "vof mms",
                                  "A postprocessor that approximates the interface error.")
  }
}
